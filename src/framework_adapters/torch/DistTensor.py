import torch.distributed as dist
from utils import XLOG
from PsKvstore import get_kvstore, kvinit
import dist_utils
import torch as th
DIST_TENSOR_ID = 0


def _default_init_data(tensor, shape, dtype):
    return tensor.zero_()


class DistTensor:
    def __init__(
        self,
        shape,
        dtype,
        name=None,
        init_func=None,
        part_policy=None,
        persistent=False,
        is_gdata=True,
        attach=True,
    ):

        assert type(shape) is tuple
        self.kvstore = get_kvstore()
        assert (
            self.kvstore is not None
        ), "Distributed module is not initialized. Please call dgl.distributed.initialize."
        self._shape = shape
        self._dtype = dtype
        self._attach = attach
        self._is_gdata = is_gdata
        self._part_policy = part_policy

        if init_func is None:
            init_func = _default_init_data
        exist_names = self.kvstore.data_name_list()
        # If a user doesn't provide a name, we generate a name ourselves.
        # We need to generate the name in a deterministic way.
        if name is None:
            raise NotImplementedError

        self._tensor_name = name
        self._name = name
        self._persistent = persistent

        if self._name not in exist_names:
            self._owner = True
            self.kvstore.init_data(
                self._name, shape, dtype, part_policy, init_func, is_gdata
            )
        else:
            self._owner = False
            dtype1, shape1, _ = self.kvstore.get_data_meta(self._name)
            assert (
                dtype == dtype1
            ), "The dtype does not match with the existing tensor"
            assert (
                shape == shape1
            ), "The shape does not match with the existing tensor"

    def __del__(self):
        if not self._persistent and self._owner:
            self.kvstore.Delete(self._name)

    def __getitem__(self, idx):
        idx = dist_utils.toindex(idx)
        # idx = idx.tousertensor()
        result = self.kvstore.Get(name=self._name, id_tensor=idx)
        # return th.tensor(result.reshape((len(idx), self.shape[1])))
        return result.reshape((len(idx), self.shape[1]))

    def __setitem__(self, idx, val):
        idx = dist_utils.toindex(idx)
        self.kvstore.Put(name=self._name, id_tensor=idx, data_tensor=val)

    def get_shm_tensor(self):
        return self.kvstore.tensor_store[self._name]

    @property
    def kvstore_key(self):
        """Return the key string of this DistTensor in the associated KVStore."""
        return self._name

    @property
    def name(self):
        """Return the name of the distributed tensor

        Returns
        -------
        str
            The name of the tensor.
        """
        # return self._detach_group_id(self._name)
        return self._name

    @property
    def tensor_name(self):
        """Return the tensor name

        Returns
        -------
        str
            The name of the tensor.
        """
        return self._tensor_name
        # return self._detach_group_id(self._tensor_name)

    @property
    def local_partition(self):
        """Return the local partition of this DistTensor."""
        return self.kvstore.data_store[self._name]

    """
    def __or__(self, other):
        new_dist_tensor = DistTensor(
            self._shape,
            self._dtype,
            part_policy=self._part_policy,
            persistent=self._persistent,
            is_gdata=self._is_gdata,
            attach=self._attach,
        )
        kvstore = self.kvstore
        kvstore.union(self._name, other._name, new_dist_tensor._name)
        return new_dist_tensor
    """

    def __len__(self):
        return self._shape[0]

    @property
    def part_policy(self):
        """Return the partition policy

        Returns
        -------
        PartitionPolicy
            The partition policy of the distributed tensor.
        """
        return self._part_policy

    @property
    def shape(self):
        """Return the shape of the distributed tensor.

        Returns
        -------
        tuple
            The shape of the distributed tensor.
        """
        return self._shape

    def dim(self):
        return len(self.shape)

    @property
    def dtype(self):
        """Return the data type of the distributed tensor.

        Returns
        ------
        dtype
            The data type of the tensor.
        """
        return self._dtype

    def to_dense_tensor(self):
        assert self.dim() == 2
        return self.kvstore.GetRowTensor(self._name)


if __name__ == "__main__":
    kvinit()
    a = DistTensor((10, 3), th.float32, name='a', persistent=True)
    b = DistTensor((10, 3), th.float32, name='b', persistent=True)
    idx = th.LongTensor([1, 2, 3])
    a[1] = th.Tensor([1, 2, 3])
    a[2] = th.Tensor([4, 5, 6])
    a[3] = th.Tensor([7, 8, 9])

    print(a[idx])
    assert th.equal(a[idx], th.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

    a[idx] = th.Tensor([[4, 5, 6], [7, 8, 9], [1, 2, 3]])
    print(a[idx])
    assert th.equal(a[idx], th.Tensor([[4, 5, 6], [7, 8, 9], [1, 2, 3]]))
