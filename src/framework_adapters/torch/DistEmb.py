"""Define sparse embedding and optimizer."""

import torch as th
import torch.distributed as dist

from DistTensor import DistTensor
import dist_utils

from utils import XLOG


class DistEmbedding:
    """
    Parameters
    ----------
    num_embeddings : int
        The number of embeddings. Currently, the number of embeddings has to be the same as
        the number of nodes or the number of edges.
    embedding_dim : int
        The dimension size of embeddings.
    name : str, optional
        The name of the embeddings. The name can uniquely identify embeddings in a system
        so that another DistEmbedding object can referent to the same embeddings.
    init_func : callable, optional
        The function to create the initial data. If the init function is not provided,
        the values of the embeddings are initialized to zero.
    part_policy : PartitionPolicy, optional
        The partition policy that assigns embeddings to different machines in the cluster.
        Currently, it only supports node partition policy or edge partition policy.
        The system determines the right partition policy automatically.

    Examples
    --------
    >>> def initializer(shape, dtype):
            arr = th.zeros(shape, dtype=dtype)
            arr.uniform_(-1, 1)
            return arr
    >>> emb = dgl.distributed.DistEmbedding(g.num_nodes(), 10, init_func=initializer)
    >>> optimizer = dgl.distributed.optim.SparseAdagrad([emb], lr=0.001)
    >>> for blocks in dataloader:
    ...     feats = emb(nids)
    ...     loss = F.sum(feats + 1, 0)
    ...     loss.backward()
    ...     optimizer.step()

    Note
    ----
    When a ``DistEmbedding``  object is used when the deep learning framework is recording
    the forward computation, users have to invoke
    py:meth:`~dgl.distributed.optim.SparseAdagrad.step` afterwards. Otherwise, there will be
    some memory leak.
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        name=None,
        init_func=None,
        part_policy=None,
    ):
        self._tensor = DistTensor(
            (num_embeddings, embedding_dim),
            th.float32,
            name,
            init_func=init_func,
            part_policy=part_policy,
        )
        self._trace = []
        self._hand_grad = []
        self._name = name
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        """
        # Check whether it is multi-gpu/distributed training or not
        if th.distributed.is_initialized():
            self._rank = th.distributed.get_rank()
            self._world_size = th.distributed.get_world_size()
        # [TODO] The following code is clearly wrong but changing it to "raise DGLError"
        # actually fails unit test.  ???
        # else:
        #     assert 'th.distributed should be initialized'
        """
        self._optm_state = None  # track optimizer state
        self._part_policy = part_policy

    def __call__(self, idx, device=th.device("cpu"), record_trace=True):
        """
        node_ids : th.tensor
            Index of the embeddings to collect.
        device : th.device
            Target device to put the collected embeddings.

        Returns
        -------
        Tensor
            The requested node embeddings
        """
        idx = dist_utils.toTensor(dist_utils.toindex(idx))
        emb = self._tensor[idx].to(device, non_blocking=True)

        if True or th.is_grad_enabled():
            emb = dist_utils.attach_grad(emb)
            if record_trace:
                self._trace.append((idx.to(device, non_blocking=True), emb))
                print(f"self._trace.append(({idx}, {emb}))")
        return emb

    def record_grad(self, idx, grad):
        # XLOG.debug(f"record_grad {idx} {grad}")
        assert len(idx) == len(grad)
        self._hand_grad.append((idx, grad))

    def reset_trace(self):
        """Reset the traced data."""
        self._trace = []
        self._hand_grad = []
    
    def get_grad(self):
        return self._hand_grad

    def set_data(self, data: th.Tensor):
        assert data.shape == (self.num_embeddings,
                              self.embedding_dim), "Shape does not match!"
        self.weight[th.arange(0, self.num_embeddings,
                              dtype=th.int64, device='cpu')] = data

    @property
    def part_policy(self):
        """Return the partition policy

        Returns
        -------
        PartitionPolicy
            partition policy
        """
        return self._part_policy

    @property
    def name(self):
        """Return the name of the embeddings

        Returns
        -------
        str
            The name of the embeddings
        """
        return self._tensor.tensor_name

    @property
    def data_name(self):
        """Return the data name of the embeddings

        Returns
        -------
        str
            The data name of the embeddings
        """
        return self._tensor._name

    @property
    def kvstore(self):
        """Return the kvstore client

        Returns
        -------
        KVClient
            The kvstore client
        """
        return self._tensor.kvstore

    @property
    def num_embeddings(self):
        """Return the number of embeddings

        Returns
        -------
        int
            The number of embeddings
        """
        return self._num_embeddings

    @property
    def embedding_dim(self):
        """Return the dimension of embeddings

        Returns
        -------
        int
            The dimension of embeddings
        """
        return self._embedding_dim

    @property
    def shape(self):
        return (self.num_embeddings, self.embedding_dim)

    @property
    def optm_state(self):
        """Return the optimizer related state tensor.

        Returns
        -------
        tuple of torch.Tensor
            The optimizer related state.
        """
        return self._optm_state

    @property
    def weight(self):
        """Return the tensor storing the node embeddings

        Returns
        -------
        torch.Tensor
            The tensor storing the node embeddings
        """
        return self._tensor

    def get_shm_tensor(self):
        return self.weight.get_shm_tensor()