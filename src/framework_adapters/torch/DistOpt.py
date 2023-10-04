"""Node embedding optimizers for distributed training"""
import abc
import warnings
from abc import abstractmethod
from os.path import exists
from DistEmb import DistEmbedding
from DistTensor import DistTensor
import torch.distributed as dist
import utils
import torch as th
import logging

EMB_STATES = "emb_states"
WORLD_SIZE = "world_size"
IDS = "ids"
PARAMS = "params"
STATES = "states"
NODE_PART_POLICY = "node"
EDGE_PART_POLICY = "edge"


class DistSparseGradOptimizer(abc.ABC):
    r"""The abstract dist sparse optimizer.

    Note: dgl dist sparse optimizer only work with dgl.distributed.DistEmbedding

    Parameters
    ----------
    params : list of DistEmbedding
        The list of DistEmbedding.
    lr : float
        The learning rate.
    """

    def __init__(self, params, lr):
        self._params = params
        self._lr = lr
        self._rank = None
        self._world_size = None
        self._shared_cache = {}
        self._clean_grad = False
        self._opt_meta = {}
        self._state = {}
        # collect all hyper parameters for save
        self._defaults = {}
        if th.distributed.is_initialized():
            self._rank = th.distributed.get_rank()
            self._world_size = th.distributed.get_world_size()
        else:
            self._rank = 0
            self._world_size = 1

    def local_state_dict(self):
        """Return the state pertaining to current rank of the optimizer.

        Returns
        -------
        dict
            Local state dict
            Example Dict of Adagrad Optimizer:
            .. code-block:: json

            {
                "params": {
                    "_lr": 0.01,
                    "_eps": "1e-8",
                    "world_size": 2
                },
                "emb_states": {
                    "emb_name1": {
                        "ids": [0, 2, 4, 6 ,8 ,10], ## tensor,
                        "emb_name1_sum": [0.1 , 0.2, 0.5, 0.1, 0.2] ## tensor,
                    },
                    "emb_name2": {
                        "ids": [0, 2, 4, 6 ,8 ,10], ## tensor,
                        "emb_name2_sum": [0.3 , 0.2, 0.4, 0.5, 0.2] ## tensor,
                    }
                }
            }

            :param json: json object

        See Also
        --------
        load_local_state_dict
        """
        local_state_dict = {}
        local_state_dict[EMB_STATES] = {}
        local_state_dict[PARAMS] = {WORLD_SIZE: self._world_size}
        for emb in self._params:
            trainers_per_machine = self._world_size // max(
                1  # , dgl.distributed.get_num_machines()
            )
            emb_state_dict = {}
            part_policy = (
                emb.part_policy if emb.part_policy else emb.weight.part_policy
            )
            idx = self._get_local_ids(part_policy)
            if trainers_per_machine > 1:
                kv_idx_split = (idx % trainers_per_machine).long()
                local_rank = self._rank % trainers_per_machine
                mask = kv_idx_split == local_rank
                idx = utils.boolean_mask(idx, mask)
            emb_state_dict.update({IDS: idx})
            emb_state = {}
            states = (
                list(self._state[emb.name])
                if isinstance(self._state[emb.name], tuple)
                else [self._state[emb.name]]
            )
            emb_state = {state.name: state[idx] for state in states}
            emb_state_dict.update({STATES: emb_state})
            local_state_dict[EMB_STATES].update({emb.name: emb_state_dict})
        local_state_dict[PARAMS].update(self._defaults)
        return local_state_dict

    def load_local_state_dict(self, local_state_dict):
        """Load the local state from the input state_dict,
        updating the optimizer as needed.

        Parameters
        ----------
        local_state_dict : dict
            Optimizer state; should be an object returned
            from a call to local_state_dict().

        See Also
        --------
        local_state_dict
        """
        for emb_name, emb_state in local_state_dict[EMB_STATES].items():
            idx = emb_state[IDS]
            # As state of an embedding of different optimizers can be a single
            # DistTensor(Adagrad) or a tuple(Adam) of that, converting it to list for
            # consistency. The list contains reference(s) to original DistTensor(s).
            states = (
                list(self._state[emb_name])
                if isinstance(self._state[emb_name], tuple)
                else [self._state[emb_name]]
            )
            if len(emb_state[STATES]) != len(states):
                raise ValueError(
                    f"loaded state dict has a different number of states"
                    f" of embedding {emb_name}"
                )
            name_to_index = {
                state.name: index for index, state in enumerate(states)
            }
            for name, state in emb_state[STATES].items():
                if name not in name_to_index:
                    raise ValueError(
                        "loaded state dict contains a state {name}"
                        "that can't be found in the optimizer states"
                    )
                state_idx = name_to_index[name]
                state = state.to(
                    th.device("cpu"), states[name_to_index[name]].dtype
                )
                states[state_idx][idx] = state
        self._defaults.update(local_state_dict[PARAMS])
        self.__dict__.update(local_state_dict[PARAMS])

    def save(self, f):
        """Save the local state_dict to disk on per rank.

        Saved dict contains 2 parts:

        * 'params': hyper parameters of the optimizer.
        * 'emb_states': partial optimizer states, each embedding contains 2 items:
            1. ```ids```: global id of the nodes/edges stored in this rank.
            2. ```states```: state data corrseponding to ```ids```.

        NOTE: This needs to be called on all ranks.

        Parameters
        ----------
        f : Union[str, os.PathLike]
            The path of the file to save to.

        See Also
        --------
        load
        """
        if self._world_size > 1:
            th.distributed.barrier()
        f = f if isinstance(f, str) else str(f, "UTF-8")
        f = f"{f}_{self._rank}"
        th.save(self.local_state_dict(), f)
        if self._world_size > 1:
            th.distributed.barrier()

    def load(self, f):
        """Load the local state of the optimizer from the file on per rank.

        NOTE: This needs to be called on all ranks.

        Parameters
        ----------
        f : Union[str, os.PathLike]
            The path of the file to load from.

        See Also
        --------
        save
        """
        if self._world_size > 1:
            th.distributed.barrier()
        f = f if isinstance(f, str) else str(f, "UTF-8")
        f_attach_rank = f"{f}_{self._rank}"
        # Don't throw error here to support device number scale-out
        # after reloading, but make sure your hyper parameter is same
        # as before because new added local optimizers will be filled
        # in nothing
        if not exists(f_attach_rank):
            warnings.warn(
                f"File {f_attach_rank} can't be found, load nothing.")
        else:
            old_world_size = self._load_state_from(f_attach_rank)
            # Device number scale-in
            if self._world_size < old_world_size:
                for rank in range(
                    self._rank + self._world_size,
                    old_world_size,
                    self._world_size,
                ):
                    self._load_state_from(f"{f}_{rank}")
        if self._world_size > 1:
            th.distributed.barrier()

    def _load_state_from(self, f):
        local_state_dict = th.load(f)
        world_size = local_state_dict[PARAMS].pop(WORLD_SIZE)
        self.load_local_state_dict(local_state_dict)
        return world_size

    def _get_local_ids(self, part_policy):
        if EDGE_PART_POLICY in part_policy.policy_str:
            return part_policy.partition_book.partid2eids(
                part_policy.part_id, part_policy.type_name
            )
        elif NODE_PART_POLICY in part_policy.policy_str:
            return part_policy._partition_book.partid2nids(
                part_policy.part_id, part_policy.type_name
            )
        else:
            raise RuntimeError(
                "Cannot support policy: %s " % part_policy.policy_str
            )

    def step(self):
        """The step function.

        The step function is invoked at the end of every batch to push the gradients
        of the embeddings involved in a mini-batch to DGL's servers and update the embeddings.
        """
        with th.no_grad():
            device = th.device("cpu")
            for emb in self._params:
                name = emb.weight.name
                idics = []
                grads = []
                assert len(emb._trace) == 0
                for trace in emb._trace:
                    idx, embbed_value = trace
                    if embbed_value.grad is None:
                        assert len(idx) == 0
                    else:
                        idics.append(idx)
                        grads.append(embbed_value.grad.data)

                for each_hand_record_grad in emb._hand_grad:
                    each_idx, each_grad = each_hand_record_grad
                    if each_grad is not None:
                        idics.append(each_idx)
                        grads.append(each_grad)
                    else:
                        assert len(idx) == 0

                if len(idics) == 0:
                    return
                elif len(idics) == 1:
                    idics = idics[0]
                    grads = grads[0]
                else:
                    idics = th.cat(idics, dim=0)
                    grads = th.cat(grads, dim=0)
                
                self.update(
                    idics,
                    grads,
                    emb,
                )
                
            if self._clean_grad:
                # clean gradient track
                for emb in self._params:
                    emb.reset_trace()
                self._clean_grad = False


        """
        # synchronized gradient update
        if self._world_size > 1:
            th.distributed.barrier()
        """

    @abstractmethod
    def update(self, idx, grad, emb):
        """Update embeddings in a sparse manner
        Sparse embeddings are updated in mini batches. We maintain gradient states for
        each embedding so they can be updated separately.

        Parameters
        ----------
        idx : tensor
            Index of the embeddings to be updated.
        grad : tensor
            Gradient of each embedding.
        emb : dgl.distributed.DistEmbedding
            Sparse node embedding to update.
        """

    def zero_grad(self):
        """clean grad cache"""
        self._clean_grad = True
        # xmh add the following
        for emb in self._params:
            emb.reset_trace()


def initializer(shape, dtype):
    """Sparse optimizer state initializer

    Parameters
    ----------
    shape : tuple of ints
        The shape of the state tensor
    dtype : torch dtype
        The data type of the state tensor
    """
    arr = th.zeros(shape, dtype=dtype)
    return arr


class SparseAdagrad(DistSparseGradOptimizer):
    r"""Distributed Node embedding optimizer using the Adagrad algorithm.

    This optimizer implements a distributed sparse version of Adagrad algorithm for
    optimizing :class:`dgl.distributed.DistEmbedding`. Being sparse means it only updates
    the embeddings whose gradients have updates, which are usually a very
    small portion of the total embeddings.

    Adagrad maintains a :math:`G_{t,i,j}` for every parameter in the embeddings, where
    :math:`G_{t,i,j}=G_{t-1,i,j} + g_{t,i,j}^2` and :math:`g_{t,i,j}` is the gradient of
    the dimension :math:`j` of embedding :math:`i` at step :math:`t`.

    NOTE: The support of sparse Adagrad optimizer is experimental.

    Parameters
    ----------
    params : list[dgl.distributed.DistEmbedding]
        The list of dgl.distributed.DistEmbedding.
    lr : float
        The learning rate.
    eps : float, Optional
        The term added to the denominator to improve numerical stability
        Default: 1e-10
    """

    def __init__(self, params, lr, eps=1e-10):
        super(SparseAdagrad, self).__init__(params, lr)
        self._eps = eps
        self._defaults = {"_lr": lr, "_eps": eps}
        # We need to register a state sum for each embedding in the kvstore.
        for emb in params:
            assert isinstance(
                emb, DistEmbedding
            ), "SparseAdagrad only supports dgl.distributed.DistEmbedding"

            name = emb.name + "_sum"
            state = DistTensor(
                (emb.num_embeddings, emb.embedding_dim),
                th.float32,
                name,
                init_func=initializer,
                part_policy=emb.part_policy,
                persistent=True,
                is_gdata=False,
            )
            assert (
                emb.name not in self._state
            ), "{} already registered in the optimizer".format(emb.name)
            self._state[emb.name] = state

    def update(self, idx, grad, emb):
        """Update embeddings in a sparse manner
        Sparse embeddings are updated in mini batches. We maintain gradient states for
        each embedding so they can be updated separately.

        Parameters
        ----------
        idx : tensor
            Index of the embeddings to be updated.
        grad : tensor
            Gradient of each embedding.
        emb : dgl.distributed.DistEmbedding
            Sparse embedding to update.
        """
        eps = self._eps
        clr = self._lr

        state_dev = th.device("cpu")
        exec_dev = grad.device

        # only perform async copies cpu -> gpu, or gpu-> gpu, but block
        # when copying to the cpu, so as to ensure the copy is finished
        # before operating on the data on the cpu
        state_block = state_dev == th.device("cpu") and exec_dev != state_dev

        # the update is non-linear so indices must be unique
        grad_indices, inverse, cnt = th.unique(
            idx, return_inverse=True, return_counts=True
        )
        grad_values = th.zeros(
            (grad_indices.shape[0], grad.shape[1]), device=exec_dev
        )
        grad_values.index_add_(0, inverse, grad)
        grad_values = grad_values / cnt.unsqueeze(1)
        grad_sum = grad_values * grad_values

        # update grad state
        grad_state = self._state[emb.name][grad_indices].to(exec_dev)
        grad_state += grad_sum
        grad_state_dst = grad_state.to(state_dev, non_blocking=True)
        if state_block:
            # use events to try and overlap CPU and GPU as much as possible
            update_event = th.cuda.Event()
            update_event.record()

        # update emb
        std_values = grad_state.sqrt_().add_(eps)
        tmp = clr * grad_values / std_values
        tmp_dst = tmp.to(state_dev, non_blocking=True)

        if state_block:
            std_event = th.cuda.Event()
            std_event.record()
            # wait for our transfers from exec_dev to state_dev to finish
            # before we can use them
            update_event.wait()
        self._state[emb.name][grad_indices] = grad_state_dst

        if state_block:
            # wait for the transfer of std_values to finish before we
            # can use it
            std_event.wait()
        emb._tensor[grad_indices] -= tmp_dst


class SparseSGD(DistSparseGradOptimizer):
    def __init__(self, params, lr, eps=1e-10):
        super(SparseSGD, self).__init__(params, lr)
        self._eps = eps
        self._defaults = {"_lr": lr, "_eps": eps}
        # We need to register a state sum for each embedding in the kvstore.
        for emb in params:
            assert isinstance(
                emb, DistEmbedding
            ), "SparseSGD only supports dgl.distributed.DistEmbedding"

            name = emb.name + "_sum"



    def update(self, idx, grad, emb):
        """Update embeddings in a sparse manner
        Sparse embeddings are updated in mini batches. We maintain gradient states for
        each embedding so they can be updated separately.

        Parameters
        ----------
        idx : tensor
            Index of the embeddings to be updated.
        grad : tensor
            Gradient of each embedding.
        emb : dgl.distributed.DistEmbedding
            Sparse embedding to update.
        """
        eps = self._eps
        clr = self._lr

        if len(idx) == 0:
            return


        state_dev = th.device("cpu")
        exec_dev = grad.device

        # tmp = - clr * grad
        # if tmp.device != state_dev:
        #     tmp = tmp.to(state_dev)
        # emb.get_shm_tensor().index_add(0, idx, tmp)

        assert exec_dev == th.device("cpu")
        # the update is non-linear so indices must be unique
        grad_indices, inverse, cnt = th.unique(
            idx, return_inverse=True, return_counts=True
        )
        grad_values = th.zeros(
            (grad_indices.shape[0], grad.shape[1]), device=exec_dev
        )
        grad_values.index_add_(0, inverse, grad)
        grad_values = grad_values / cnt.unsqueeze(1)
        # update emb
        tmp = clr * grad_values
        tmp_dst = tmp.to(state_dev, non_blocking=True)
        # logging.debug(f"OPT: grad_indices={grad_indices}, tmp_dst={tmp_dst}")
        emb._tensor[grad_indices] = emb._tensor[grad_indices] - tmp_dst




class SparseRowWiseAdaGrad(DistSparseGradOptimizer):
    def __init__(self, params, lr, eps=1e-10):
        super(SparseRowWiseAdaGrad, self).__init__(params, lr)
        self._eps = eps
        self._defaults = {"_lr": lr, "_eps": eps}
        # We need to register a state sum for each embedding in the kvstore.
        for emb in params:
            assert isinstance(
                emb, DistEmbedding
            ), "SparseRowWiseAdaGrad only supports dgl.distributed.DistEmbedding"

            name = emb.name + "_SparseRowWiseAdaGrad_sum"

            state = DistTensor(
                (emb.num_embeddings,),
                th.float32,
                name,
                init_func=initializer,
                part_policy=emb.part_policy,
                persistent=True,
                is_gdata=False,
            )
            assert (
                emb.name not in self._state
            ), "{} already registered in the optimizer".format(emb.name)
            self._state[emb.name] = state


    def update(self, idxs, grads, emb):
        """Update embeddings in a sparse manner
        Sparse embeddings are updated in mini batches. We maintain gradient states for
        each embedding so they can be updated separately.

        Parameters
        ----------
        idx : tensor
            Index of the embeddings to be updated.
        grad : tensor
            Gradient of each embedding.
        emb : dgl.distributed.DistEmbedding
            Sparse embedding to update.
        """
        if len(idxs) == 0:
            return

        grads = grads.cuda()
        assert idxs.is_cpu
        assert grads.is_cuda
        
        eps = self._eps
        clr = self._lr
        state_dev = th.device("cpu")
        exec_dev = grads.device
        state_block = state_dev == th.device("cpu") and exec_dev != state_dev

        grad_sum = (grads * grads).mean(1)

        if state_dev != idxs.device:
            idxs = idxs.to(state_dev)
        if state_dev != grad_sum.device:
            grad_sum = grad_sum.to(state_dev)
        
        self._state[emb.name].get_shm_tensor().index_add_(0, idxs, grad_sum)
        
        std = self._state[emb.name].get_shm_tensor()[idxs]
        std = std.cuda()
        std_values = std.sqrt_().add_(self._eps).unsqueeze(1)
        
        tmp = - clr * grads / std_values

        if tmp.device != state_dev:
            tmp = tmp.to(state_dev)
        emb.get_shm_tensor().index_add(0, idxs, tmp)