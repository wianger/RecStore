import dgl.backend as F
from DistOpt import SparseSGD, SparseAdagrad, SparseRowWiseAdaGrad
from cache_emb_factory import CacheEmbFactory
from DistEmb import DistEmbedding
from .tensor_models import ExternalEmbedding
import torch as th
import torch.nn as nn
import torch.nn.functional as functional
import torch.nn.init as INIT


import sys

sys.path.append("/home/xieminhui/RecStore/src/framework_adapters/torch")


is_custom_external_emb_initialized = False


class CustomExternalEmbedding:
    """Sparse Embedding for Knowledge Graph
    It is used to store both entity embeddings and relation embeddings.

    Parameters
    ----------
    args :
        Global configs.
    num : int
        Number of embeddings.
    dim : int
        Embedding dimention size.
    device : th.device
        Device to store the embedding.
    """

    def __init__(self, args, num, dim, emb_name):
        self.gpu = args.gpu
        self.args = args
        self.emb = DistEmbedding(num, dim, name=emb_name)

        global is_custom_external_emb_initialized
        assert not is_custom_external_emb_initialized
        is_custom_external_emb_initialized = True

        self.num = num
        self.trace = []

        # dummy LR, only register the tensor state of OSP
        _ = SparseRowWiseAdaGrad([self.emb], lr=self.args.lr)

        self.has_cross_rel = False
        self.debug_externel_emb = ExternalEmbedding(args, num, dim, F.cpu())

    def prepare_per_worker_process(self):
        args = {
            "cache_ratio": self.args.cache_ratio,
            "kForwardItersPerStep": self.args.kForwardItersPerStep,
            "backwardMode": self.args.backwardMode,
            "backgrad_init": self.args.backgrad_init,

        }
        self.cached_emb = CacheEmbFactory.New(
            self.args.cached_emb_type, self.emb, args)
        self.dist_opt = SparseRowWiseAdaGrad([self.emb], lr=self.args.lr)
        self.rank = th.distributed.get_rank()

    def init(self, emb_init):
        """Initializing the embeddings.

        Parameters
        ----------
        emb_init : float
            The intial embedding range should be [-emb_init, emb_init].
        """
        shm_tensor = self.emb.weight
        INIT.uniform_(shm_tensor, -emb_init, emb_init)

    def setup_cross_rels(self, cross_rels, global_relation_emb):
        cpu_bitmap = th.zeros((self.num,), dtype=th.bool)
        for i, rel in enumerate(cross_rels):
            cpu_bitmap[rel] = 1
        self.cpu_bitmap = cpu_bitmap
        self.has_cross_rel = True
        self.global_relation_emb = global_relation_emb

    def get_noncross_idx(self, idx):
        cpu_mask = self.cpu_bitmap[idx]
        gpu_mask = ~cpu_mask
        return idx[gpu_mask]

    def share_memory(self):
        """Use torch.tensor.share_memory_() to allow cross process tensor access
        """
        pass

    def __call__(self, idx, gpu_id=-1, trace=True):
        """ Return sliced tensor.

        Parameters
        ----------
        idx : th.tensor
            Slicing index
        gpu_id : int
            Which gpu to put sliced data in.
        trace : bool
            If True, trace the computation. This is required in training.
            If False, do not trace the computation.
            Default: True
        """
        if self.has_cross_rel:
            assert False
            cpu_idx = idx.cpu()
            cpu_mask = self.cpu_bitmap[cpu_idx]
            cpu_idx = cpu_idx[cpu_mask]
            cpu_idx = th.unique(cpu_idx)
            if cpu_idx.shape[0] != 0:
                cpu_emb = self.global_relation_emb.emb[cpu_idx]
                self.emb[cpu_idx] = cpu_emb.cuda(gpu_id)

        idx = idx.cuda()
        s = self.cached_emb.forward(idx, trace)
        if gpu_id >= 0:
            s = s.cuda(gpu_id)

        if trace:
            self.trace.append((idx, s))
        data = s
        return data

    def update(self, gpu_id=-1):
        """ Update embeddings in a sparse manner
        Sparse embeddings are updated in mini batches. we maintains gradient states for
        each embedding so they can be updated separately.

        Parameters
        ----------
        gpu_id : int
            Which gpu to accelerate the calculation. if -1 is provided, cpu is used.
        """

        # for each in self.emb._hand_grad:
        #     self.debug_externel_emb.trace.append(each)
        # from pyinstrument import Profiler
        # from torch.profiler import profile, record_function, ProfilerActivity

        # for _ in range(10):
        #     with record_function("vanilla"):
        #         self.debug_externel_emb.debug_xmh_update(gpu_id)
        #     ###############################
        #     with record_function("xmh"):
        #         self.dist_opt.step()

        #     if self.rank == 0:
        #         prof.step()

        self.dist_opt.step()
        self.trace.clear()
        self.dist_opt.zero_grad()
        return

    def create_async_update(self):
        """Set up the async update subprocess.
        """
        assert False
        self.async_q = Queue(1)
        self.async_p = mp.Process(
            target=async_update, args=(self.args, self, self.async_q))
        self.async_p.start()

    def finish_async_update(self):
        """Notify the async update subprocess to quit.
        """
        assert False
        self.async_q.put((None, None, None))
        self.async_p.join()

    def curr_emb(self):
        """Return embeddings in trace.
        """
        data = [data for _, data in self.trace]
        return th.cat(data, 0)

    def save(self, path, name):
        """Save embeddings.

        Parameters
        ----------
        path : str
            Directory to save the embedding.
        name : str
            Embedding name.
        """
        pass
        assert False
        # file_name = os.path.join(path, name+'.npy')
        # np.save(file_name, self.emb.cpu().detach().numpy())

    def load(self, path, name):
        """Load embeddings.

        Parameters
        ----------
        path : str
            Directory to load the embedding.
        name : str
            Embedding name.
        """
        pass
        assert False
        # file_name = os.path.join(path, name+'.npy')
        # self.emb = th.Tensor(np.load(file_name))
