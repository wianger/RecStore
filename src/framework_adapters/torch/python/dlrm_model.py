import argparse
import torch.nn as nn
import torch
import numpy as np
import sys
import extend_distributed as ext_dist
from torch.nn.parallel.replicate import replicate
from torch.autograd.profiler import record_function
import datetime
import json
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel.scatter_gather import gather, scatter
from torch.nn.parallel.parallel_apply import parallel_apply

import time
# dataloader
import dlrm_data_pytorch as dp

try:
    from internals import (
        fbDataLoader,
        fbInputBatchFormatter,
    )
    has_internal_libs = True
except ImportError:
    has_internal_libs = False


def time_wrap(use_gpu):
    if use_gpu:
        torch.cuda.synchronize()
    return time.time()


def dlrm_wrap(X, lS_o, lS_i, use_gpu, device, ndevices=1):
    with record_function("DLRM forward"):
        if use_gpu:  # .cuda()
            # lS_i can be either a list of tensors or a stacked tensor.
            # Handle each case below:
            if ndevices == 1:
                lS_i = (
                    [S_i.to(device) for S_i in lS_i]
                    if isinstance(lS_i, list)
                    else lS_i.to(device)
                )
                lS_o = (
                    [S_o.to(device) for S_o in lS_o]
                    if isinstance(lS_o, list)
                    else lS_o.to(device)
                )
        return dlrm(X.to(device), lS_o, lS_i)


def unpack_batch(b):
    if args.data_generation == "internal":
        return fbInputBatchFormatter(b, args.data_size)
    else:
        # Experiment with unweighted samples
        return b[0], b[1], b[2], b[3], torch.ones(b[3].size()), None

def loss_fn_wrap(Z, T, use_gpu, device):
    with record_function("DLRM loss compute"):
        if args.loss_function == "mse" or args.loss_function == "bce":
            return dlrm.loss_fn(Z, T.to(device))
        elif args.loss_function == "wbce":
            loss_ws_ = dlrm.loss_ws[T.data.view(-1).long()].view_as(T).to(device)
            loss_fn_ = dlrm.loss_fn(Z, T.to(device))
            loss_sc_ = loss_ws_ * loss_fn_
            return loss_sc_.mean()
        
class DLRM_Net(nn.Module):
    def create_mlp(self, ln, sigmoid_layer):
        # build MLP layer by layer
        layers = nn.ModuleList()
        for i in range(0, ln.size - 1):
            n = ln[i]
            m = ln[i + 1]

            # construct fully connected operator
            LL = nn.Linear(int(n), int(m), bias=True)

            # initialize the weights
            # with torch.no_grad():
            # custom Xavier input, output or two-sided fill
            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            # approach 1
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)
            # approach 2
            # LL.weight.data.copy_(torch.tensor(W))
            # LL.bias.data.copy_(torch.tensor(bt))
            # approach 3
            # LL.weight = Parameter(torch.tensor(W),requires_grad=True)
            # LL.bias = Parameter(torch.tensor(bt),requires_grad=True)
            layers.append(LL)

            # construct sigmoid or relu operator
            if i == sigmoid_layer:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())

        # approach 1: use ModuleList
        # return layers
        # approach 2: use Sequential container to wrap all layers
        return torch.nn.Sequential(*layers)

    def create_emb(self, m, ln, weighted_pooling=None):
        emb_l = nn.ModuleList()
        v_W_l = []
        for i in range(0, ln.size):
            if ext_dist.my_size > 1:
                if i not in self.local_emb_indices:
                    continue
            n = ln[i]

            EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)
            # initialize embeddings
            # nn.init.uniform_(EE.weight, a=-np.sqrt(1 / n), b=np.sqrt(1 / n))
            W = np.random.uniform(
                low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
            ).astype(np.float32)
            # approach 1
            EE.weight.data = torch.tensor(W, requires_grad=True)
            # approach 2
            # EE.weight.data.copy_(torch.tensor(W))
            # approach 3
            # EE.weight = Parameter(torch.tensor(W),requires_grad=True)

            if weighted_pooling is None:
                v_W_l.append(None)
            else:
                v_W_l.append(torch.ones(n, dtype=torch.float32))
            emb_l.append(EE)
        return emb_l, v_W_l

    def __init__(
        self,
        m_spa=None,
        ln_emb=None,
        ln_bot=None,
        ln_top=None,
        arch_interaction_op=None,
        arch_interaction_itself=False,
        sigmoid_bot=-1,
        sigmoid_top=-1,
        sync_dense_params=True,
        loss_threshold=0.0,
        ndevices=-1,
        qr_flag=False,
        qr_operation="mult",
        qr_collisions=0,
        qr_threshold=200,
        md_flag=False,
        md_threshold=200,
        weighted_pooling=None,
        loss_function="bce",
    ):
        super(DLRM_Net, self).__init__()

        if (
            (m_spa is not None)
            and (ln_emb is not None)
            and (ln_bot is not None)
            and (ln_top is not None)
            and (arch_interaction_op is not None)
        ):

            # save arguments
            self.ndevices = ndevices
            self.output_d = 0
            self.parallel_model_batch_size = -1
            self.parallel_model_is_not_prepared = True
            self.arch_interaction_op = arch_interaction_op
            self.arch_interaction_itself = arch_interaction_itself
            self.sync_dense_params = sync_dense_params
            self.loss_threshold = loss_threshold
            self.loss_function = loss_function
            if weighted_pooling is not None and weighted_pooling != "fixed":
                self.weighted_pooling = "learned"
            else:
                self.weighted_pooling = weighted_pooling
            # create variables for QR embedding if applicable
            self.qr_flag = qr_flag
            if self.qr_flag:
                self.qr_collisions = qr_collisions
                self.qr_operation = qr_operation
                self.qr_threshold = qr_threshold
            # create variables for MD embedding if applicable
            self.md_flag = md_flag
            if self.md_flag:
                self.md_threshold = md_threshold

            # If running distributed, get local slice of embedding tables
            if ext_dist.my_size > 1:
                n_emb = len(ln_emb)
                if n_emb < ext_dist.my_size:
                    sys.exit(
                        "only (%d) sparse features for (%d) devices, table partitions will fail"
                        % (n_emb, ext_dist.my_size)
                    )
                self.n_global_emb = n_emb
                self.n_local_emb, self.n_emb_per_rank = ext_dist.get_split_lengths(
                    n_emb
                )
                self.local_emb_slice = ext_dist.get_my_slice(n_emb)
                self.local_emb_indices = list(range(n_emb))[
                    self.local_emb_slice]

            # create operators
            if ndevices <= 1:
                self.emb_l, w_list = self.create_emb(
                    m_spa, ln_emb, weighted_pooling)
                if self.weighted_pooling == "learned":
                    self.v_W_l = nn.ParameterList()
                    for w in w_list:
                        self.v_W_l.append(Parameter(w))
                else:
                    self.v_W_l = w_list
            self.bot_l = self.create_mlp(ln_bot, sigmoid_bot)
            self.top_l = self.create_mlp(ln_top, sigmoid_top)

            # quantization
            self.quantize_emb = False
            self.emb_l_q = []
            self.quantize_bits = 32

            # specify the loss function
            if self.loss_function == "mse":
                self.loss_fn = torch.nn.MSELoss(reduction="mean")
            elif self.loss_function == "bce":
                self.loss_fn = torch.nn.BCELoss(reduction="mean")
            elif self.loss_function == "wbce":
                self.loss_ws = torch.tensor(
                    np.fromstring(args.loss_weights, dtype=float, sep="-")
                )
                self.loss_fn = torch.nn.BCELoss(reduction="none")
            else:
                sys.exit(
                    "ERROR: --loss-function=" + self.loss_function + " is not supported"
                )

    def apply_mlp(self, x, layers):
        # approach 1: use ModuleList
        # for layer in layers:
        #     x = layer(x)
        # return x
        # approach 2: use Sequential container to wrap all layers
        return layers(x)

    def apply_emb(self, lS_o, lS_i, emb_l, v_W_l):
        # WARNING: notice that we are processing the batch at once. We implicitly
        # assume that the data is laid out such that:
        # 1. each embedding is indexed with a group of sparse indices,
        #   corresponding to a single lookup
        # 2. for each embedding the lookups are further organized into a batch
        # 3. for a list of embedding tables there is a list of batched lookups

        ly = []
        for k, sparse_index_group_batch in enumerate(lS_i):
            sparse_offset_group_batch = lS_o[k]

            # embedding lookup
            # We are using EmbeddingBag, which implicitly uses sum operator.
            # The embeddings are represented as tall matrices, with sum
            # happening vertically across 0 axis, resulting in a row vector
            # E = emb_l[k]

            if v_W_l[k] is not None:
                per_sample_weights = v_W_l[k].gather(
                    0, sparse_index_group_batch)
            else:
                per_sample_weights = None

            if self.quantize_emb:
                assert False
                s1 = self.emb_l_q[k].element_size() * \
                    self.emb_l_q[k].nelement()
                s2 = self.emb_l_q[k].element_size() * \
                    self.emb_l_q[k].nelement()
                print("quantized emb sizes:", s1, s2)

                if self.quantize_bits == 4:
                    QV = ops.quantized.embedding_bag_4bit_rowwise_offsets(
                        self.emb_l_q[k],
                        sparse_index_group_batch,
                        sparse_offset_group_batch,
                        per_sample_weights=per_sample_weights,
                    )
                elif self.quantize_bits == 8:
                    QV = ops.quantized.embedding_bag_byte_rowwise_offsets(
                        self.emb_l_q[k],
                        sparse_index_group_batch,
                        sparse_offset_group_batch,
                        per_sample_weights=per_sample_weights,
                    )

                ly.append(QV)
            else:
                E = emb_l[k]
                V = E(
                    sparse_index_group_batch,
                    sparse_offset_group_batch,
                    per_sample_weights=per_sample_weights,
                )

                ly.append(V)

        # print(ly)
        return ly

    def interact_features(self, x, ly):

        if self.arch_interaction_op == "dot":
            # concatenate dense and sparse features
            (batch_size, d) = x.shape
            T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
            # perform a dot product
            Z = torch.bmm(T, torch.transpose(T, 1, 2))
            # append dense feature with the interactions (into a row vector)
            # approach 1: all
            # Zflat = Z.view((batch_size, -1))
            # approach 2: unique
            _, ni, nj = Z.shape
            # approach 1: tril_indices
            # offset = 0 if self.arch_interaction_itself else -1
            # li, lj = torch.tril_indices(ni, nj, offset=offset)
            # approach 2: custom
            offset = 1 if self.arch_interaction_itself else 0
            li = torch.tensor([i for i in range(ni)
                              for j in range(i + offset)])
            lj = torch.tensor([j for i in range(nj)
                              for j in range(i + offset)])
            Zflat = Z[:, li, lj]
            # concatenate dense features and interactions
            R = torch.cat([x] + [Zflat], dim=1)
        elif self.arch_interaction_op == "cat":
            # concatenation features (into a row vector)
            R = torch.cat([x] + ly, dim=1)
        else:
            sys.exit(
                "ERROR: --arch-interaction-op="
                + self.arch_interaction_op
                + " is not supported"
            )

        return R

    def forward(self, dense_x, lS_o, lS_i):
        if ext_dist.my_size > 1:
            # multi-node multi-device run
            return self.distributed_forward(dense_x, lS_o, lS_i)
        elif self.ndevices <= 1:
            # single device run
            return self.sequential_forward(dense_x, lS_o, lS_i)
        else:
            # single-node multi-device run
            return self.parallel_forward(dense_x, lS_o, lS_i)

    def distributed_forward(self, dense_x, lS_o, lS_i):
        batch_size = dense_x.size()[0]
        # WARNING: # of ranks must be <= batch size in distributed_forward call
        if batch_size < ext_dist.my_size:
            sys.exit(
                "ERROR: batch_size (%d) must be larger than number of ranks (%d)"
                % (batch_size, ext_dist.my_size)
            )
        if batch_size % ext_dist.my_size != 0:
            sys.exit(
                "ERROR: batch_size %d can not split across %d ranks evenly"
                % (batch_size, ext_dist.my_size)
            )

        dense_x = dense_x[ext_dist.get_my_slice(batch_size)]
        lS_o = lS_o[self.local_emb_slice]
        lS_i = lS_i[self.local_emb_slice]

        if (len(self.emb_l) != len(lS_o)) or (len(self.emb_l) != len(lS_i)):
            sys.exit(
                "ERROR: corrupted model input detected in distributed_forward call"
            )

        # embeddings
        with record_function("DLRM embedding forward"):
            ly = self.apply_emb(lS_o, lS_i, self.emb_l, self.v_W_l)

        # WARNING: Note that at this point we have the result of the embedding lookup
        # for the entire batch on each rank. We would like to obtain partial results
        # corresponding to all embedding lookups, but part of the batch on each rank.
        # Therefore, matching the distribution of output of bottom mlp, so that both
        # could be used for subsequent interactions on each device.
        if len(self.emb_l) != len(ly):
            sys.exit(
                "ERROR: corrupted intermediate result in distributed_forward call")

        a2a_req = ext_dist.alltoall(ly, self.n_emb_per_rank)

        with record_function("DLRM bottom nlp forward"):
            x = self.apply_mlp(dense_x, self.bot_l)

        ly = a2a_req.wait()
        ly = list(ly)

        # interactions
        with record_function("DLRM interaction forward"):
            z = self.interact_features(x, ly)

        # top mlp
        with record_function("DLRM top nlp forward"):
            p = self.apply_mlp(z, self.top_l)

        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z = torch.clamp(p, min=self.loss_threshold,
                            max=(1.0 - self.loss_threshold))
        else:
            z = p

        return z

    def sequential_forward(self, dense_x, lS_o, lS_i):
        # process dense features (using bottom mlp), resulting in a row vector
        
        print("dense_x", dense_x.shape)
        print("lS_o", lS_o.shape)
        for S_i in lS_i:
            print("S_i", S_i.shape)

        x = self.apply_mlp(dense_x, self.bot_l)
        print("x", x.shape)
        # debug prints
        # print("intermediate")
        # print(x.detach().cpu().numpy())

        # process sparse features(using embeddings), resulting in a list of row vectors
        ly = self.apply_emb(lS_o, lS_i, self.emb_l, self.v_W_l)
        # for y in ly:
        #     print(y.detach().cpu().numpy())

        
        for y in ly:
            print("y", y.shape)


        # interact features (dense and sparse)
        z = self.interact_features(x, ly)
        print("z", z.shape)
        
        # print(z.detach().cpu().numpy())
        
        print("top_l", self.top_l)

        # obtain probability of a click (using top mlp)
        p = self.apply_mlp(z, self.top_l)

        print("p", p.shape)

        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z = torch.clamp(p, min=self.loss_threshold,
                            max=(1.0 - self.loss_threshold))
        else:
            z = p

        return z

    def parallel_forward(self, dense_x, lS_o, lS_i):
        ### prepare model (overwrite) ###
        # WARNING: # of devices must be >= batch size in parallel_forward call
        batch_size = dense_x.size()[0]
        ndevices = min(self.ndevices, batch_size, len(self.emb_l))
        device_ids = range(ndevices)
        # WARNING: must redistribute the model if mini-batch size changes(this is common
        # for last mini-batch, when # of elements in the dataset/batch size is not even
        if self.parallel_model_batch_size != batch_size:
            self.parallel_model_is_not_prepared = True

        if self.parallel_model_is_not_prepared or self.sync_dense_params:
            # replicate mlp (data parallelism)
            self.bot_l_replicas = replicate(self.bot_l, device_ids)
            self.top_l_replicas = replicate(self.top_l, device_ids)
            self.parallel_model_batch_size = batch_size

        if self.parallel_model_is_not_prepared:
            # distribute embeddings (model parallelism)
            t_list = []
            w_list = []
            for k, emb in enumerate(self.emb_l):
                d = torch.device("cuda:" + str(k % ndevices))
                t_list.append(emb.to(d))
                if self.weighted_pooling == "learned":
                    w_list.append(Parameter(self.v_W_l[k].to(d)))
                elif self.weighted_pooling == "fixed":
                    w_list.append(self.v_W_l[k].to(d))
                else:
                    w_list.append(None)
            self.emb_l = nn.ModuleList(t_list)
            if self.weighted_pooling == "learned":
                self.v_W_l = nn.ParameterList(w_list)
            else:
                self.v_W_l = w_list
            self.parallel_model_is_not_prepared = False

        ### prepare input (overwrite) ###
        # scatter dense features (data parallelism)
        # print(dense_x.device)
        dense_x = scatter(dense_x, device_ids, dim=0)
        # distribute sparse features (model parallelism)
        if (len(self.emb_l) != len(lS_o)) or (len(self.emb_l) != len(lS_i)):
            sys.exit(
                "ERROR: corrupted model input detected in parallel_forward call")

        t_list = []
        i_list = []
        for k, _ in enumerate(self.emb_l):
            d = torch.device("cuda:" + str(k % ndevices))
            t_list.append(lS_o[k].to(d))
            i_list.append(lS_i[k].to(d))
        lS_o = t_list
        lS_i = i_list

        ### compute results in parallel ###
        # bottom mlp
        # WARNING: Note that the self.bot_l is a list of bottom mlp modules
        # that have been replicated across devices, while dense_x is a tuple of dense
        # inputs that has been scattered across devices on the first (batch) dimension.
        # The output is a list of tensors scattered across devices according to the
        # distribution of dense_x.
        x = parallel_apply(self.bot_l_replicas, dense_x, None, device_ids)
        # debug prints
        # print(x)

        # embeddings
        ly = self.apply_emb(lS_o, lS_i, self.emb_l, self.v_W_l)
        # debug prints
        # print(ly)

        # butterfly shuffle (implemented inefficiently for now)
        # WARNING: Note that at this point we have the result of the embedding lookup
        # for the entire batch on each device. We would like to obtain partial results
        # corresponding to all embedding lookups, but part of the batch on each device.
        # Therefore, matching the distribution of output of bottom mlp, so that both
        # could be used for subsequent interactions on each device.
        if len(self.emb_l) != len(ly):
            sys.exit(
                "ERROR: corrupted intermediate result in parallel_forward call")

        t_list = []
        for k, _ in enumerate(self.emb_l):
            d = torch.device("cuda:" + str(k % ndevices))
            y = scatter(ly[k], device_ids, dim=0)
            t_list.append(y)
        # adjust the list to be ordered per device
        ly = list(map(lambda y: list(y), zip(*t_list)))
        # debug prints
        # print(ly)

        # interactions
        z = []
        for k in range(ndevices):
            zk = self.interact_features(x[k], ly[k])
            z.append(zk)
        # debug prints
        # print(z)

        # top mlp
        # WARNING: Note that the self.top_l is a list of top mlp modules that
        # have been replicated across devices, while z is a list of interaction results
        # that by construction are scattered across devices on the first (batch) dim.
        # The output is a list of tensors scattered across devices according to the
        # distribution of z.
        p = parallel_apply(self.top_l_replicas, z, None, device_ids)

        ### gather the distributed results ###
        p0 = gather(p, self.output_d, dim=0)

        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z0 = torch.clamp(
                p0, min=self.loss_threshold, max=(1.0 - self.loss_threshold)
            )
        else:
            z0 = p0

        return z0


def dash_separated_ints(value):
    vals = value.split("-")
    for val in vals:
        try:
            int(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of ints" % value
            )

    return value


def dash_separated_floats(value):
    vals = value.split("-")
    for val in vals:
        try:
            float(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of floats" % value
            )

    return value


def run():
    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="Train Deep Learning Recommendation Model (DLRM)"
    )
    # model related parameters
    parser.add_argument("--arch-sparse-feature-size", type=int, default=2)
    parser.add_argument(
        "--arch-embedding-size", type=dash_separated_ints, default="4-3-2"
    )
    # j will be replaced with the table number
    parser.add_argument(
        "--arch-mlp-bot", type=dash_separated_ints, default="4-3-2")
    parser.add_argument(
        "--arch-mlp-top", type=dash_separated_ints, default="4-2-1")
    parser.add_argument(
        "--arch-interaction-op", type=str, choices=["dot", "cat"], default="dot"
    )
    parser.add_argument("--arch-interaction-itself",
                        action="store_true", default=False)
    parser.add_argument("--weighted-pooling", type=str, default=None)
    # embedding table options
    parser.add_argument("--md-flag", action="store_true", default=False)
    parser.add_argument("--md-threshold", type=int, default=200)
    parser.add_argument("--md-temperature", type=float, default=0.3)
    parser.add_argument("--md-round-dims", action="store_true", default=False)
    parser.add_argument("--qr-flag", action="store_true", default=False)
    parser.add_argument("--qr-threshold", type=int, default=200)
    parser.add_argument("--qr-operation", type=str, default="mult")
    parser.add_argument("--qr-collisions", type=int, default=4)
    # activations and loss
    parser.add_argument("--activation-function", type=str, default="relu")
    parser.add_argument("--loss-function", type=str,
                        default="mse")  # or bce or wbce
    parser.add_argument(
        "--loss-weights", type=dash_separated_floats, default="1.0-1.0"
    )  # for wbce
    parser.add_argument("--loss-threshold", type=float, default=0.0)  # 1.0e-7
    parser.add_argument("--round-targets", type=bool, default=False)
    # data
    parser.add_argument("--data-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=0)
    parser.add_argument(
        "--data-generation", type=str, choices=["random", "dataset", "internal"], default="random"
    )  # synthetic, dataset or internal
    parser.add_argument(
        "--rand-data-dist", type=str, default="uniform"
    )  # uniform or gaussian
    parser.add_argument("--rand-data-min", type=float, default=0)
    parser.add_argument("--rand-data-max", type=float, default=1)
    parser.add_argument("--rand-data-mu", type=float, default=-1)
    parser.add_argument("--rand-data-sigma", type=float, default=1)
    parser.add_argument("--data-trace-file", type=str,
                        default="./input/dist_emb_j.log")
    parser.add_argument("--data-set", type=str,
                        default="kaggle")  # or terabyte
    parser.add_argument("--raw-data-file", type=str, default="")
    parser.add_argument("--processed-data-file", type=str, default="")
    parser.add_argument("--data-randomize", type=str,
                        default="total")  # or day or none
    parser.add_argument("--data-trace-enable-padding",
                        type=bool, default=False)
    parser.add_argument("--max-ind-range", type=int, default=-1)
    parser.add_argument("--data-sub-sample-rate",
                        type=float, default=0.0)  # in [0, 1]
    parser.add_argument("--num-indices-per-lookup", type=int, default=10)
    parser.add_argument("--num-indices-per-lookup-fixed",
                        type=bool, default=False)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--memory-map", action="store_true", default=False)
    # training
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--nepochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--print-precision", type=int, default=5)
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    parser.add_argument("--sync-dense-params", type=bool, default=True)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument(
        "--dataset-multiprocessing",
        action="store_true",
        default=False,
        help="The Kaggle dataset can be multiprocessed in an environment \
                        with more than 7 CPU cores and more than 20 GB of memory. \n \
                        The Terabyte dataset can be multiprocessed in an environment \
                        with more than 24 CPU cores and at least 1 TB of memory.",
    )
    # inference
    parser.add_argument("--inference-only", action="store_true", default=False)
    # quantize
    parser.add_argument("--quantize-mlp-with-bit", type=int, default=32)
    parser.add_argument("--quantize-emb-with-bit", type=int, default=32)
    # onnx
    parser.add_argument("--save-onnx", action="store_true", default=False)
    # gpu
    parser.add_argument("--use-gpu", action="store_true", default=True)
    # distributed
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--dist-backend", type=str, default="")
    # debugging and profiling
    parser.add_argument("--print-freq", type=int, default=1)
    parser.add_argument("--test-freq", type=int, default=-1)
    parser.add_argument("--test-mini-batch-size", type=int, default=-1)
    parser.add_argument("--test-num-workers", type=int, default=-1)
    parser.add_argument("--print-time", action="store_true", default=False)
    parser.add_argument("--print-wall-time",
                        action="store_true", default=False)
    parser.add_argument("--debug-mode", action="store_true", default=False)
    parser.add_argument("--enable-profiling",
                        action="store_true", default=False)
    parser.add_argument("--plot-compute-graph",
                        action="store_true", default=False)
    parser.add_argument("--tensor-board-filename",
                        type=str, default="run_kaggle_pt")
    # store/load model
    parser.add_argument("--save-model", type=str, default="")
    parser.add_argument("--load-model", type=str, default="")
    # mlperf logging (disables other output and stops early)
    parser.add_argument("--mlperf-logging", action="store_true", default=False)
    # stop at target accuracy Kaggle 0.789, Terabyte (sub-sampled=0.875) 0.8107
    parser.add_argument("--mlperf-acc-threshold", type=float, default=0.0)
    # stop at target AUC Terabyte (no subsampling) 0.8025
    parser.add_argument("--mlperf-auc-threshold", type=float, default=0.0)
    parser.add_argument("--mlperf-bin-loader",
                        action="store_true", default=False)
    parser.add_argument("--mlperf-bin-shuffle",
                        action="store_true", default=False)
    # mlperf gradient accumulation iterations
    parser.add_argument("--mlperf-grad-accum-iter", type=int, default=1)
    # LR policy
    parser.add_argument("--lr-num-warmup-steps", type=int, default=0)
    parser.add_argument("--lr-decay-start-step", type=int, default=0)
    parser.add_argument("--lr-num-decay-steps", type=int, default=0)

    global args
    global nbatches
    global nbatches_test
    global writer
    args = parser.parse_args()

    if args.dataset_multiprocessing:
        assert float(sys.version[:3]) > 3.7, (
            "The dataset_multiprocessing "
            + "flag is susceptible to a bug in Python 3.7 and under. "
            + "https://github.com/facebookresearch/dlrm/issues/172"
        )

    if args.weighted_pooling is not None:
        if args.qr_flag:
            sys.exit(
                "ERROR: quotient remainder with weighted pooling is not supported")
        if args.md_flag:
            sys.exit(
                "ERROR: mixed dimensions with weighted pooling is not supported")

    ### some basic setup ###
    np.random.seed(args.numpy_rand_seed)
    np.set_printoptions(precision=args.print_precision)
    torch.set_printoptions(precision=args.print_precision)
    torch.manual_seed(args.numpy_rand_seed)

    if args.test_mini_batch_size < 0:
        # if the parameter is not set, use the training batch size
        args.test_mini_batch_size = args.mini_batch_size
    if args.test_num_workers < 0:
        # if the parameter is not set, use the same parameter for training
        args.test_num_workers = args.num_workers

    use_gpu = args.use_gpu and torch.cuda.is_available()

    if not args.debug_mode:
        ext_dist.init_distributed(
            local_rank=args.local_rank, use_gpu=use_gpu, backend=args.dist_backend
        )

    if use_gpu:
        torch.cuda.manual_seed_all(args.numpy_rand_seed)
        torch.backends.cudnn.deterministic = True
        if ext_dist.my_size > 1:
            ngpus = 1
            device = torch.device("cuda", ext_dist.my_local_rank)
        else:
            ngpus = torch.cuda.device_count()
            device = torch.device("cuda", 0)
        print("Using {} GPU(s)...".format(ngpus))
    else:
        device = torch.device("cpu")
        print("Using CPU...")

    ### prepare training data ###
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    # input data

    if args.data_generation == "dataset":
        train_data, train_ld, test_data, test_ld = dp.make_criteo_data_and_loaders(
            args)
        table_feature_map = {idx: idx for idx in range(len(train_data.counts))}
        nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)
        nbatches_test = len(test_ld)

        ln_emb = train_data.counts
        # enforce maximum limit on number of vectors per embedding
        if args.max_ind_range > 0:
            ln_emb = np.array(
                list(
                    map(
                        lambda x: x if x < args.max_ind_range else args.max_ind_range,
                        ln_emb,
                    )
                )
            )
        else:
            ln_emb = np.array(ln_emb)
        m_den = train_data.m_den
        ln_bot[0] = m_den
    elif args.data_generation == "internal":
        if not has_internal_libs:
            raise Exception("Internal libraries are not available.")
        NUM_BATCHES = 5000
        nbatches = args.num_batches if args.num_batches > 0 else NUM_BATCHES
        train_ld, feature_to_num_embeddings = fbDataLoader(
            args.data_size, nbatches)
        ln_emb = np.array(list(feature_to_num_embeddings.values()))
        m_den = ln_bot[0]
    else:
        # input and target at random
        ln_emb = np.fromstring(args.arch_embedding_size, dtype=int, sep="-")
        m_den = ln_bot[0]
        train_data, train_ld, test_data, test_ld = dp.make_random_data_and_loader(
            args, ln_emb, m_den
        )
        nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)
        nbatches_test = len(test_ld)

    args.ln_emb = ln_emb.tolist()
    if args.mlperf_logging:
        print("command line args: ", json.dumps(vars(args)))

    ### parse command line arguments ###
    m_spa = args.arch_sparse_feature_size
    ln_emb = np.asarray(ln_emb)
    num_fea = ln_emb.size + 1  # num sparse + num dense features

    m_den_out = ln_bot[ln_bot.size - 1]
    if args.arch_interaction_op == "dot":
        # approach 1: all
        # num_int = num_fea * num_fea + m_den_out
        # approach 2: unique
        if args.arch_interaction_itself:
            num_int = (num_fea * (num_fea + 1)) // 2 + m_den_out
        else:
            num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
    elif args.arch_interaction_op == "cat":
        num_int = num_fea * m_den_out
    else:
        sys.exit(
            "ERROR: --arch-interaction-op="
            + args.arch_interaction_op
            + " is not supported"
        )
    arch_mlp_top_adjusted = str(num_int) + "-" + args.arch_mlp_top
    ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")

    # sanity check: feature sizes and mlp dimensions must match
    if m_den != ln_bot[0]:
        sys.exit(
            "ERROR: arch-dense-feature-size "
            + str(m_den)
            + " does not match first dim of bottom mlp "
            + str(ln_bot[0])
        )
    if args.qr_flag:
        if args.qr_operation == "concat" and 2 * m_spa != m_den_out:
            sys.exit(
                "ERROR: 2 arch-sparse-feature-size "
                + str(2 * m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
                + " (note that the last dim of bottom mlp must be 2x the embedding dim)"
            )
        if args.qr_operation != "concat" and m_spa != m_den_out:
            sys.exit(
                "ERROR: arch-sparse-feature-size "
                + str(m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
            )
    else:
        if m_spa != m_den_out:
            sys.exit(
                "ERROR: arch-sparse-feature-size "
                + str(m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
            )
    if num_int != ln_top[0]:
        sys.exit(
            "ERROR: # of feature interactions "
            + str(num_int)
            + " does not match first dimension of top mlp "
            + str(ln_top[0])
        )

    # test prints (model arch)
    if args.debug_mode:
        print("model arch:")
        print(
            "mlp top arch "
            + str(ln_top.size - 1)
            + " layers, with input to output dimensions:"
        )
        print(ln_top)
        print("# of interactions")
        print(num_int)
        print(
            "mlp bot arch "
            + str(ln_bot.size - 1)
            + " layers, with input to output dimensions:"
        )
        print(ln_bot)
        print("# of features (sparse and dense)")
        print(num_fea)
        print("dense feature size")
        print(m_den)
        print("sparse feature size")
        print(m_spa)
        print(
            "# of embeddings (= # of sparse features) "
            + str(ln_emb.size)
            + ", with dimensions "
            + str(m_spa)
            + "x:"
        )
        print(ln_emb)

        print("data (inputs and targets):")
        for j, inputBatch in enumerate(train_ld):
            X, lS_o, lS_i, T, W, CBPP = unpack_batch(inputBatch)

            torch.set_printoptions(precision=4)
            # early exit if nbatches was set by the user and has been exceeded
            if nbatches > 0 and j >= nbatches:
                break
            print("mini-batch: %d" % j)
            print(X.detach().cpu())
            # transform offsets to lengths when printing
            print(
                torch.IntTensor(
                    [
                        np.diff(
                            S_o.detach().cpu().tolist() + list(lS_i[i].shape)
                        ).tolist()
                        for i, S_o in enumerate(lS_o)
                    ]
                )
            )
            print([S_i.detach().cpu() for S_i in lS_i])
            print(T.detach().cpu())

    global ndevices
    ndevices = min(ngpus, args.mini_batch_size, num_fea - 1) if use_gpu else -1

    ### construct the neural network specified above ###
    # WARNING: to obtain exactly the same initialization for
    # the weights we need to start from the same random seed.
    # np.random.seed(args.numpy_rand_seed)
    global dlrm
    dlrm = DLRM_Net(
        m_spa,
        ln_emb,
        ln_bot,
        ln_top,
        arch_interaction_op=args.arch_interaction_op,
        arch_interaction_itself=args.arch_interaction_itself,
        sigmoid_bot=-1,
        sigmoid_top=ln_top.size - 2,
        sync_dense_params=args.sync_dense_params,
        loss_threshold=args.loss_threshold,
        ndevices=ndevices,
        qr_flag=args.qr_flag,
        qr_operation=args.qr_operation,
        qr_collisions=args.qr_collisions,
        qr_threshold=args.qr_threshold,
        md_flag=args.md_flag,
        md_threshold=args.md_threshold,
        weighted_pooling=args.weighted_pooling,
        loss_function=args.loss_function,
    )

    # test prints
    if args.debug_mode:
        print("initial parameters (weights and bias):")
        for param in dlrm.parameters():
            print(param.detach().cpu().numpy())
        # print(dlrm)

    if use_gpu:
        # Custom Model-Data Parallel
        # the mlps are replicated and use data parallelism, while
        # the embeddings are distributed and use model parallelism
        dlrm = dlrm.to(device)  # .cuda()
        if dlrm.ndevices > 1:
            dlrm.emb_l, dlrm.v_W_l = dlrm.create_emb(
                m_spa, ln_emb, args.weighted_pooling
            )
        else:
            if dlrm.weighted_pooling == "fixed":
                for k, w in enumerate(dlrm.v_W_l):
                    dlrm.v_W_l[k] = w.cuda()

    # distribute data parallel mlps
    if ext_dist.my_size > 1:
        if use_gpu:
            device_ids = [ext_dist.my_local_rank]
            dlrm.bot_l = ext_dist.DDP(dlrm.bot_l, device_ids=device_ids)
            dlrm.top_l = ext_dist.DDP(dlrm.top_l, device_ids=device_ids)
        else:
            dlrm.bot_l = ext_dist.DDP(dlrm.bot_l)
            dlrm.top_l = ext_dist.DDP(dlrm.top_l)

    if not args.inference_only:
        if use_gpu and args.optimizer in ["adagrad"]:
            sys.exit("GPU version of Adagrad is not supported by PyTorch.")
        # specify the optimizer algorithm
        opts = {
            "sgd": torch.optim.SGD,
            "adagrad": torch.optim.Adagrad,
        }

        parameters = (
            dlrm.parameters()
            if ext_dist.my_size == 1
            else [
                {
                    "params": [p for emb in dlrm.emb_l for p in emb.parameters()],
                    "lr": args.learning_rate,
                },
                # TODO check this lr setup
                # bottom mlp has no data parallelism
                # need to check how do we deal with top mlp
                {
                    "params": dlrm.bot_l.parameters(),
                    "lr": args.learning_rate,
                },
                {
                    "params": dlrm.top_l.parameters(),
                    "lr": args.learning_rate,
                },
            ]
        )
        optimizer = opts[args.optimizer](parameters, lr=args.learning_rate)
        # lr_scheduler = LRPolicyScheduler(
        #     optimizer,
        #     args.lr_num_warmup_steps,
        #     args.lr_decay_start_step,
        #     args.lr_num_decay_steps,
        # )

    ### main loop ###

    # training or inference
    best_acc_test = 0
    best_auc_test = 0
    skip_upto_epoch = 0
    skip_upto_batch = 0
    total_time = 0
    total_loss = 0
    total_iter = 0
    total_samp = 0

    print("time/loss/accuracy (if enabled):")

    tb_file = "./" + args.tensor_board_filename
    writer = SummaryWriter(tb_file)

    ext_dist.barrier()
    with torch.autograd.profiler.profile(
        args.enable_profiling, use_cuda=use_gpu, record_shapes=True
    ) as prof:
        if not args.inference_only:
            k = 0
            total_time_begin = 0
            while k < args.nepochs:
                if k < skip_upto_epoch:
                    continue

                if args.mlperf_logging:
                    previous_iteration_time = None

                for j, inputBatch in enumerate(train_ld):
                    if j == 0 and args.save_onnx:
                        X_onnx, lS_o_onnx, lS_i_onnx, _, _, _ = unpack_batch(
                            inputBatch)

                    if j < skip_upto_batch:
                        continue

                    X, lS_o, lS_i, T, W, CBPP = unpack_batch(inputBatch)

                    if args.mlperf_logging:
                        current_time = time_wrap(use_gpu)
                        if previous_iteration_time:
                            iteration_time = current_time - previous_iteration_time
                        else:
                            iteration_time = 0
                        previous_iteration_time = current_time
                    else:
                        t1 = time_wrap(use_gpu)

                    # early exit if nbatches was set by the user and has been exceeded
                    if nbatches > 0 and j >= nbatches:
                        break

                    # Skip the batch if batch size not multiple of total ranks
                    if ext_dist.my_size > 1 and X.size(0) % ext_dist.my_size != 0:
                        print(
                            "Warning: Skiping the batch %d with size %d"
                            % (j, X.size(0))
                        )
                        continue

                    # = args.mini_batch_size except maybe for last
                    mbs = T.shape[0]

                    # forward pass
                    Z = dlrm_wrap(
                        X,
                        lS_o,
                        lS_i,
                        use_gpu,
                        device,
                        ndevices=ndevices,
                    )

                    if ext_dist.my_size > 1:
                        T = T[ext_dist.get_my_slice(mbs)]
                        W = W[ext_dist.get_my_slice(mbs)]

                    # loss
                    E = loss_fn_wrap(Z, T, use_gpu, device)

                    # compute loss and accuracy
                    L = E.detach().cpu().numpy()  # numpy array
                    # training accuracy is not disabled
                    # S = Z.detach().cpu().numpy()  # numpy array
                    # T = T.detach().cpu().numpy()  # numpy array

                    # # print("res: ", S)

                    # # print("j, train: BCE ", j, L)

                    # mbs = T.shape[0]  # = args.mini_batch_size except maybe for last
                    # A = np.sum((np.round(S, 0) == T).astype(np.uint8))

                    with record_function("DLRM backward"):
                        # scaled error gradient propagation
                        # (where we do not accumulate gradients across mini-batches)
                        if (
                            args.mlperf_logging
                            and (j + 1) % args.mlperf_grad_accum_iter == 0
                        ) or not args.mlperf_logging:
                            optimizer.zero_grad()
                        # backward pass
                        E.backward()

                        # optimizer
                        if (
                            args.mlperf_logging
                            and (j + 1) % args.mlperf_grad_accum_iter == 0
                        ) or not args.mlperf_logging:
                            optimizer.step()
                            # lr_scheduler.step()

                    if args.mlperf_logging:
                        total_time += iteration_time
                    else:
                        t2 = time_wrap(use_gpu)
                        total_time += t2 - t1

                    total_loss += L * mbs
                    total_iter += 1
                    total_samp += mbs

                    should_print = ((j + 1) % args.print_freq == 0) or (
                        j + 1 == nbatches
                    )
                    should_test = (
                        (args.test_freq > 0)
                        and (args.data_generation in ["dataset", "random"])
                        and (((j + 1) % args.test_freq == 0) or (j + 1 == nbatches))
                    )

                    # print time, loss and accuracy
                    if should_print or should_test:
                        gT = 1000.0 * total_time / total_iter if args.print_time else -1
                        total_time = 0

                        train_loss = total_loss / total_samp
                        total_loss = 0

                        str_run_type = (
                            "inference" if args.inference_only else "training"
                        )

                        wall_time = ""
                        if args.print_wall_time:
                            wall_time = " ({})".format(time.strftime("%H:%M"))

                        print(
                            "Finished {} it {}/{} of epoch {}, {:.2f} ms/it,".format(
                                str_run_type, j + 1, nbatches, k, gT
                            )
                            + " loss {:.6f}".format(train_loss)
                            + wall_time,
                            flush=True,
                        )

                        log_iter = nbatches * k + j + 1
                        writer.add_scalar("Train/Loss", train_loss, log_iter)

                        total_iter = 0
                        total_samp = 0

                    # testing
                    if should_test:
                        epoch_num_float = (j + 1) / len(train_ld) + k + 1
                        # don't measure training iter time in a test iteration
                        if args.mlperf_logging:
                            previous_iteration_time = None
                        print(
                            "Testing at - {}/{} of epoch {},".format(
                                j + 1, nbatches, k)
                        )
                        model_metrics_dict, is_best = inference(
                            args,
                            dlrm,
                            best_acc_test,
                            best_auc_test,
                            test_ld,
                            device,
                            use_gpu,
                            log_iter,
                        )

                        if (
                            is_best
                            and not (args.save_model == "")
                            and not args.inference_only
                        ):
                            model_metrics_dict["epoch"] = k
                            model_metrics_dict["iter"] = j + 1
                            model_metrics_dict["train_loss"] = train_loss
                            model_metrics_dict["total_loss"] = total_loss
                            model_metrics_dict[
                                "opt_state_dict"
                            ] = optimizer.state_dict()
                            print("Saving model to {}".format(args.save_model))
                            torch.save(model_metrics_dict, args.save_model)

                        # Uncomment the line below to print out the total time with overhead
                        # print("Total test time for this group: {}" \
                        # .format(time_wrap(use_gpu) - accum_test_time_begin))

                        if (
                            args.mlperf_logging
                            and (args.mlperf_acc_threshold > 0)
                            and (best_acc_test > args.mlperf_acc_threshold)
                        ):
                            print(
                                "MLPerf testing accuracy threshold "
                                + str(args.mlperf_acc_threshold)
                                + " reached, stop training"
                            )
                            break

                        if (
                            args.mlperf_logging
                            and (args.mlperf_auc_threshold > 0)
                            and (best_auc_test > args.mlperf_auc_threshold)
                        ):
                            print(
                                "MLPerf testing auc threshold "
                                + str(args.mlperf_auc_threshold)
                                + " reached, stop training"
                            )
                            break

                k += 1  # nepochs

        else:
            print("Testing for inference only")
            inference(
                args,
                dlrm,
                best_acc_test,
                best_auc_test,
                test_ld,
                device,
                use_gpu,
            )

    # profiling
    if args.enable_profiling:
        time_stamp = str(datetime.datetime.now()).replace(" ", "_")
        with open("dlrm_s_pytorch" + time_stamp + "_shape.prof", "w") as prof_f:
            prof_f.write(
                prof.key_averages(group_by_input_shape=True).table(
                    sort_by="self_cpu_time_total"
                )
            )
        with open("dlrm_s_pytorch" + time_stamp + "_total.prof", "w") as prof_f:
            prof_f.write(prof.key_averages().table(
                sort_by="self_cpu_time_total"))
        prof.export_chrome_trace("dlrm_s_pytorch" + time_stamp + ".json")
        # print(prof.key_averages().table(sort_by="cpu_time_total"))

    # plot compute graph
    if args.plot_compute_graph:
        sys.exit(
            "ERROR: Please install pytorchviz package in order to use the"
            + " visualization. Then, uncomment its import above as well as"
            + " three lines below and run the code again."
        )
        # V = Z.mean() if args.inference_only else E
        # dot = make_dot(V, params=dict(dlrm.named_parameters()))
        # dot.render('dlrm_s_pytorch_graph') # write .pdf file

    # test prints
    if not args.inference_only and args.debug_mode:
        print("updated parameters (weights and bias):")
        for param in dlrm.parameters():
            print(param.detach().cpu().numpy())

    total_time_end = time_wrap(use_gpu)


if __name__ == "__main__":
    run()
