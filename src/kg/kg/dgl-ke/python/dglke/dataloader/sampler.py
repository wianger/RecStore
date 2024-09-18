# -*- coding: utf-8 -*-
#
# sampler.py
#
# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from collections import Counter
import logging
import math
import numpy as np
import scipy as sp
import dgl.backend as F
import dgl
import os
import sys
import pickle
import time
import torch as th
from dgl.base import NID, EID
import datetime

import sys
sys.path.append("/home/xieminhui/RecStore/src/python")  # nopep8
# sys.path.append("/home/xieminhui/RecStore/src/framework_adapters/torch")  # nopep8
import recstore
# from python.test_utils import diff_objs, diff_presampling


def SoftRelationPartition(edges, n, has_importance=False, threshold=0.05):
    """This partitions a list of edges to n partitions according to their
    relation types. For any relation with number of edges larger than the
    threshold, its edges will be evenly distributed into all partitions.
    For any relation with number of edges smaller than the threshold, its
    edges will be put into one single partition.

    Algo:
    For r in relations:
        if r.size() > threshold
            Evenly divide edges of r into n parts and put into each relation.
        else
            Find partition with fewest edges, and put edges of r into
            this partition.

    Parameters
    ----------
    edges : (heads, rels, tails) triple
        Edge list to partition
    n : int
        Number of partitions
    threshold : float
        The threshold of whether a relation is LARGE or SMALL
        Default: 5%

    Returns
    -------
    List of np.array
        Edges of each partition
    List of np.array
        Edge types of each partition
    bool
        Whether there exists some relations belongs to multiple partitions
    """
    if has_importance:
        heads, rels, tails, e_impts = edges
    else:
        heads, rels, tails = edges
    print('relation partition {} edges into {} parts'.format(len(heads), n))
    uniq, cnts = np.unique(rels, return_counts=True)
    idx = np.flip(np.argsort(cnts))
    cnts = cnts[idx]
    uniq = uniq[idx]
    assert cnts[0] > cnts[-1]
    edge_cnts = np.zeros(shape=(n,), dtype=np.int64)
    rel_cnts = np.zeros(shape=(n,), dtype=np.int64)
    rel_dict = {}
    rel_parts = []
    cross_rel_part = []
    for _ in range(n):
        rel_parts.append([])

    large_threshold = int(len(rels) * threshold)
    capacity_per_partition = int(len(rels) / n)
    # ensure any relation larger than the partition capacity will be split
    large_threshold = capacity_per_partition if capacity_per_partition < large_threshold \
        else large_threshold
    num_cross_part = 0
    for i in range(len(cnts)):
        cnt = cnts[i]
        r = uniq[i]
        r_parts = []
        if cnt > large_threshold:
            avg_part_cnt = (cnt // n) + 1
            num_cross_part += 1
            for j in range(n):
                part_cnt = avg_part_cnt if cnt > avg_part_cnt else cnt
                r_parts.append([j, part_cnt])
                rel_parts[j].append(r)
                edge_cnts[j] += part_cnt
                rel_cnts[j] += 1
                cnt -= part_cnt
            cross_rel_part.append(r)
        else:
            idx = np.argmin(edge_cnts)
            r_parts.append([idx, cnt])
            rel_parts[idx].append(r)
            edge_cnts[idx] += cnt
            rel_cnts[idx] += 1
        rel_dict[r] = r_parts

    for i, edge_cnt in enumerate(edge_cnts):
        print('part {} has {} edges and {} relations'.format(
            i, edge_cnt, rel_cnts[i]))
    print('{}/{} duplicated relation across partitions'.format(num_cross_part, len(cnts)))

    parts = []
    for i in range(n):
        parts.append([])
        rel_parts[i] = np.array(rel_parts[i])

    for i, r in enumerate(rels):
        r_part = rel_dict[r][0]
        part_idx = r_part[0]
        cnt = r_part[1]
        parts[part_idx].append(i)
        cnt -= 1
        if cnt == 0:
            rel_dict[r].pop(0)
        else:
            rel_dict[r][0][1] = cnt

    for i, part in enumerate(parts):
        parts[i] = np.array(part, dtype=np.int64)
    shuffle_idx = np.concatenate(parts)
    heads[:] = heads[shuffle_idx]
    rels[:] = rels[shuffle_idx]
    tails[:] = tails[shuffle_idx]
    if has_importance:
        e_impts[:] = e_impts[shuffle_idx]

    off = 0
    for i, part in enumerate(parts):
        parts[i] = np.arange(off, off + len(part))
        off += len(part)
    cross_rel_part = np.array(cross_rel_part)

    return parts, rel_parts, num_cross_part > 0, cross_rel_part


def BalancedRelationPartition(edges, n, has_importance=False):
    """This partitions a list of edges based on relations to make sure
    each partition has roughly the same number of edges and relations.
    Algo:
    For r in relations:
      Find partition with fewest edges
      if r.size() > num_of empty_slot
         put edges of r into this partition to fill the partition,
         find next partition with fewest edges to put r in.
      else
         put edges of r into this partition.

    Parameters
    ----------
    edges : (heads, rels, tails) triple
        Edge list to partition
    n : int
        number of partitions

    Returns
    -------
    List of np.array
        Edges of each partition
    List of np.array
        Edge types of each partition
    bool
        Whether there exists some relations belongs to multiple partitions
    """
    if has_importance:
        heads, rels, tails, e_impts = edges
    else:
        heads, rels, tails = edges
    print('relation partition {} edges into {} parts'.format(len(heads), n))
    uniq, cnts = np.unique(rels, return_counts=True)
    idx = np.flip(np.argsort(cnts))
    cnts = cnts[idx]
    uniq = uniq[idx]
    assert cnts[0] > cnts[-1]
    edge_cnts = np.zeros(shape=(n,), dtype=np.int64)
    rel_cnts = np.zeros(shape=(n,), dtype=np.int64)
    rel_dict = {}
    rel_parts = []
    for _ in range(n):
        rel_parts.append([])

    max_edges = (len(rels) // n) + 1
    num_cross_part = 0
    for i in range(len(cnts)):
        cnt = cnts[i]
        r = uniq[i]
        r_parts = []

        while cnt > 0:
            idx = np.argmin(edge_cnts)
            if edge_cnts[idx] + cnt <= max_edges:
                r_parts.append([idx, cnt])
                rel_parts[idx].append(r)
                edge_cnts[idx] += cnt
                rel_cnts[idx] += 1
                cnt = 0
            else:
                cur_cnt = max_edges - edge_cnts[idx]
                r_parts.append([idx, cur_cnt])
                rel_parts[idx].append(r)
                edge_cnts[idx] += cur_cnt
                rel_cnts[idx] += 1
                num_cross_part += 1
                cnt -= cur_cnt
        rel_dict[r] = r_parts

    for i, edge_cnt in enumerate(edge_cnts):
        print('part {} has {} edges and {} relations'.format(
            i, edge_cnt, rel_cnts[i]))
    print('{}/{} duplicated relation across partitions'.format(num_cross_part, len(cnts)))

    parts = []
    for i in range(n):
        parts.append([])
        rel_parts[i] = np.array(rel_parts[i])

    for i, r in enumerate(rels):
        r_part = rel_dict[r][0]
        part_idx = r_part[0]
        cnt = r_part[1]
        parts[part_idx].append(i)
        cnt -= 1
        if cnt == 0:
            rel_dict[r].pop(0)
        else:
            rel_dict[r][0][1] = cnt

    for i, part in enumerate(parts):
        parts[i] = np.array(part, dtype=np.int64)
    shuffle_idx = np.concatenate(parts)
    heads[:] = heads[shuffle_idx]
    rels[:] = rels[shuffle_idx]
    tails[:] = tails[shuffle_idx]
    if has_importance:
        e_impts[:] = e_impts[shuffle_idx]

    off = 0
    for i, part in enumerate(parts):
        parts[i] = np.arange(off, off + len(part))
        off += len(part)

    return parts, rel_parts, num_cross_part > 0


def RandomPartition(edges, n, has_importance=False):
    """This partitions a list of edges randomly across n partitions

    Parameters
    ----------
    edges : (heads, rels, tails) triple
        Edge list to partition
    n : int
        number of partitions

    Returns
    -------
    List of np.array
        Edges of each partition
    """
    if has_importance:
        heads, rels, tails, e_impts = edges
    else:
        heads, rels, tails = edges
    print('random partition {} edges into {} parts'.format(len(heads), n))
    idx = np.random.permutation(len(heads))
    heads[:] = heads[idx]
    rels[:] = rels[idx]
    tails[:] = tails[idx]
    if has_importance:
        e_impts[:] = e_impts[idx]

    part_size = int(math.ceil(len(idx) / n))
    parts = []
    for i in range(n):
        start = part_size * i
        end = min(part_size * (i + 1), len(idx))
        parts.append(idx[start:end])
        print('RandomPartition: part {} has {} edges'.format(
            i, len(parts[-1])))
    return parts


def ConstructGraph(dataset, args):
    """Construct Graph for training

    Parameters
    ----------
    dataset :
        the dataset
    args :
        Global configs.
    """
    src = [dataset.train[0]]
    etype_id = [dataset.train[1]]
    dst = [dataset.train[2]]
    num_train = len(dataset.train[0])

    if args.dataset == "wikikg90M":
        valid_dict = dataset.valid
        num_valid = len(valid_dict['h,r->t']['hr'])
    elif hasattr(dataset, 'valid') and dataset.valid is not None:
        src.append(dataset.valid[0])
        etype_id.append(dataset.valid[1])
        dst.append(dataset.valid[2])
        num_valid = len(dataset.valid[0])
    else:
        num_valid = 0
    if args.dataset == "wikikg90M":
        test_dict = dataset.test
        num_test = len(test_dict['h,r->t']['hr'])
    elif hasattr(dataset, 'test') and dataset.test is not None:
        src.append(dataset.test[0])
        etype_id.append(dataset.test[1])
        dst.append(dataset.test[2])
        num_test = len(dataset.test[0])
    else:
        num_test = 0

    if args.dataset == "wikikg90M":
        print('|valid|:', num_valid)
        print('|test|:', num_test)
        return

    src = np.concatenate(src)
    etype_id = np.concatenate(etype_id)
    dst = np.concatenate(dst)
    n_entities = dataset.n_entities

    coo = sp.sparse.coo_matrix((np.ones(len(src)), (src, dst)), shape=[
                               n_entities, n_entities])

    g = dgl._deprecate.graph.DGLGraph(
        coo, readonly=True, multigraph=True, sort_csr=True)
    # g = dgl.DGLGraph(coo, readonly=True, multigraph=True, sort_csr=True)

    g.edata['tid'] = F.tensor(etype_id, F.int64)
    if args.has_edge_importance:
        e_impts = F.tensor(dataset.train[3], F.float32)
        e_impts_vt = F.zeros((num_valid + num_test,),
                             dtype=F.float32, ctx=F.cpu())
        g.edata['impts'] = F.cat([e_impts, e_impts_vt], dim=0)
    return g


class TensorCounter():
    def __init__(self, ):
        self.counter = Counter()

    def Update(self, id_tensor):
        self.counter.update(id_tensor.tolist())

    def most_common(self, k):
        return self.counter.most_common(k)

    def __eq__(self, __value: object) -> bool:
        return self.counter == __value.counter


class TrainDataset(object):
    """Dataset for training

    Parameters
    ----------
    dataset : KGDataset
        Original dataset.
    args :
        Global configs.
    ranks:
        Number of partitions.
    """

    # output: self.part_dict
    def PartitionGraph(self, g, args, ranks):
        new_version_g = dgl.DGLGraph(g.edges())
        new_version_g.edata['tid'] = g.edata['tid']

        print("original graph: ", g)
        pickle_path = f"/tmp/metis_{args.dataset}_{ranks}"
        if os.path.exists(pickle_path):
            print("Loading cached metis", flush=True)
            with open(pickle_path, 'rb') as f:
                self.part_dict = pickle.load(f)
                print(self.part_dict)
        else:
            print("Partitoning with metis", flush=True)
            self.part_dict = dgl.transforms.metis_partition(
                new_version_g, ranks, 1)
            with open(pickle_path, 'wb') as f:
                pickle.dump(self.part_dict, f)

        for i, each_part in self.part_dict.items():
            print(
                f'MertisPartition: part {i} has {each_part.num_nodes()} N {each_part.num_edges()} E')
            each_part.edata['tid'] = new_version_g.edata['tid'][each_part.edata[EID]]

    def __init__(self, g, dataset, args, ranks=64, has_importance=False):
        triples = dataset.train
        num_train = len(triples[0])
        print('|Train|:', num_train)

        self.nr_world = ranks
        if ranks > 1 and args.rel_part:
            self.edge_parts, self.rel_parts, self.cross_part, self.cross_rels = \
                SoftRelationPartition(
                    triples, ranks, has_importance=has_importance)
        elif ranks > 1:
            # self.edge_parts = RandomPartition(
            #     triples, ranks, has_importance=has_importance)
            # self.cross_part = True

            self.PartitionGraph(g, args, ranks)
            self.dataset_name = dataset.name
            self.cross_part = True

        else:
            self.edge_parts = [np.arange(num_train)]
            self.rel_parts = [np.arange(dataset.n_relations)]
            self.cross_part = False

        self.g = g

    def PreSampling(self, batch_size, cached_ratio,
                    neg_sample_size=2,
                    neg_chunk_size=None,
                    num_workers=1,
                    shuffle=True,
                    exclude_positive=False,
                    has_edge_importance=False
                    ):
        dgl.random.xmh_seed(0)

        cache_sizes_all_rank = []
        keys_ordered_all_rank = []
        n_entities = self.g.number_of_nodes()
        cache_size_per_rank = int(cached_ratio * n_entities // self.nr_world)
        # cache_size_per_rank = int(cached_ratio * n_entities)

        # PreSampling
        for rank in range(self.nr_world):
            head = self.create_sampler(
                batch_size, neg_sample_size, neg_chunk_size, "head", num_workers, shuffle, exclude_positive, rank)
            tail = self.create_sampler(
                batch_size, neg_sample_size, neg_chunk_size, "tail", num_workers, shuffle, exclude_positive, rank)
            sampler = NewBidirectionalOneShotIterator(head, tail, neg_sample_size, neg_sample_size,
                                                      True, n_entities,
                                                      has_edge_importance, None)
            ids_counter_per_rank = TensorCounter()

            step = 0
            # while True:
            while step <= 1000:
                try:
                    pos_g, neg_g = next(sampler)
                    step += 1
                except StopIteration:
                    break

                nids = pos_g.ndata['id']
                ids_counter_per_rank.Update(nids)

                if neg_g.neg_head:
                    neg_head_ids = neg_g.ndata['id'][neg_g.head_nid]
                    ids_counter_per_rank.Update(neg_head_ids)
                else:
                    neg_tail_ids = neg_g.ndata['id'][neg_g.tail_nid]
                    ids_counter_per_rank.Update(neg_tail_ids)

                # diff_presampling(pos_g, neg_g, rank, step)
            temp = ids_counter_per_rank.most_common(
                3 * cache_size_per_rank)

            # diff_objs(ids_counter_per_rank, f"ids_counter_per_rank{rank}")
            two_times_popular_keys = [each[0] for each in temp]
            keys_ordered_all_rank.append(two_times_popular_keys)

        def in_any_set(key, sets):
            for each_set in sets:
                if key in each_set:
                    return True
            return False
        logging.warn(f"Stop PreSampling")

        logging.warn(f"Start finding hottest IDs")
        all_rank_keys_sets = [set(keys_ordered_all_rank[i])
                              for i in range(self.nr_world)]
        all_rank_hotsets = []
        for rank in range(self.nr_world):
            ids_counter_per_rank = keys_ordered_all_rank[rank]
            ids_counter_other_rank = [all_rank_keys_sets[i]
                                      for i in range(self.nr_world) if i != rank]
            cached_keys = th.zeros((cache_size_per_rank,), dtype=th.int64)
            j = 0
            for each_key in ids_counter_per_rank:
                if in_any_set(each_key, ids_counter_other_rank):
                    continue
                cached_keys[j] = each_key
                j += 1
                if j == cache_size_per_rank:
                    break
            cached_keys = cached_keys.narrow(0, 0, j)
            # assert j == per_rank_cache_size
            print(f"Rank{rank}: cached key size {j}")
            cache_sizes_all_rank.append(j)
            all_rank_hotsets.append(cached_keys)
            
        logging.warn(f"Stop finding hottest IDs")






        # a map, <original ID> -> <new ID>
        # with hottest IDs first, then the rest IDs
        logging.warn(f"Start construct renumbering_dict")
        renumbering_dict = th.full((n_entities,), -1, dtype=th.int64)
        # start_id = 0
        # for rank in range(self.nr_world):
        #     for each in all_rank_hotsets[rank]:
        #         assert -1 == renumbering_dict[each]
        #         renumbering_dict[each] = start_id
        #         start_id += 1

        # for id in range(n_entities):
        #     if -1 == renumbering_dict[id]:
        #         renumbering_dict[id] = start_id
        #         start_id += 1
        # logging.warn("Construct renumbering_dict done")
        # assert start_id == n_entities

        recstore.construct_renumbering_dict_op(
            renumbering_dict,
            self.nr_world,
            all_rank_hotsets,
        )
        logging.warn(f"Stop construct renumbering_dict")

        return renumbering_dict, cache_sizes_all_rank

    def RenumberingGraph(self, renumbering_dict):
        del self.g
        print("Before renumbering graph: ", self.part_dict[0].ndata)
        for i, each_part in self.part_dict.items():
            each_part.ndata[NID] = renumbering_dict[each_part.ndata[NID]]
        print("After renumbering graph: ", self.part_dict[0].ndata)

    def create_sampler(self, batch_size, neg_sample_size=2, neg_chunk_size=None, mode='head', num_workers=32,
                       shuffle=True, exclude_positive=False, rank=0, real_train=False):
        """Create sampler for training

        Parameters
        ----------
        batch_size : int
            Batch size of each mini batch.
        neg_sample_size : int
            How many negative edges sampled for each node.
        neg_chunk_size : int
            How many edges in one chunk. We split one batch into chunks.
        mode : str
            Sampling mode.
        number_workers: int
            Number of workers used in parallel for this sampler
        shuffle : bool
            If True, shuffle the seed edges.
            If False, do not shuffle the seed edges.
            Default: False
        exclude_positive : bool
            If True, exlucde true positive edges in sampled negative edges
            If False, return all sampled negative edges even there are positive edges
            Default: False
        rank : int
            Which partition to sample.

        Returns
        -------
        dgl.contrib.sampling.EdgeSampler
            Edge sampler
        """

        EdgeSampler = getattr(dgl.contrib.sampling, 'EdgeSampler')
        # assert batch_size % neg_sample_size == 0, 'batch_size should be divisible by B'
        # return EdgeSampler(self.g,
        #                    seed_edges=F.tensor(self.edge_parts[rank]),
        #                    batch_size=batch_size,
        #                    neg_sample_size=int(neg_sample_size/neg_chunk_size),
        #                    chunk_size=neg_chunk_size,
        #                    negative_mode=mode,
        #                    num_workers=num_workers,
        #                    shuffle=shuffle,
        #                    exclude_positive=exclude_positive,
        #                    return_false_neg=False)

        # convert new version DGLGraph to deprecated DGLGraph
        part_g = dgl._deprecate.graph.DGLGraph(self.part_dict[rank].edges(),
                                               readonly=True,
                                               multigraph=True,
                                               sort_csr=True)
        part_g.edata['tid'] = self.part_dict[rank].edata['tid']
        part_g.ndata[NID] = self.part_dict[rank].ndata[NID]

        if rank == 0:
            if real_train:
                print("part_g:", part_g)

        return EdgeSampler(part_g,
                           # seed_edges=F.tensor(self.part_dict[rank]),
                           seed_edges=F.tensor(
                               np.arange(part_g.number_of_edges())),
                           batch_size=batch_size,
                           neg_sample_size=int(neg_sample_size/neg_chunk_size),
                           chunk_size=neg_chunk_size,
                           negative_mode=mode,
                           num_workers=num_workers,
                           shuffle=shuffle,
                           exclude_positive=exclude_positive,
                           return_false_neg=False)

# class ChunkNegEdgeSubgraph(dgl.DGLGraph):


class ChunkNegEdgeSubgraph(dgl._deprecate.graph.DGLGraph):
    """Wrapper for negative graph

        Parameters
        ----------
        neg_g : DGLGraph
            Graph holding negative edges.
        num_chunks : int
            Number of chunks in sampled graph.
        chunk_size : int
            Info of chunk_size.
        neg_sample_size : int
            Info of neg_sample_size.
        neg_head : bool
            If True, negative_mode is 'head'
            If False, negative_mode is 'tail'
    """

    def __init__(self, subg, num_chunks, chunk_size,
                 neg_sample_size, neg_head):
        super(ChunkNegEdgeSubgraph, self).__init__(graph_data=subg.sgi.graph,
                                                   readonly=True,
                                                   parent=subg._parent)
        self.ndata[NID] = subg.sgi.induced_nodes.tousertensor()
        self.edata[EID] = subg.sgi.induced_edges.tousertensor()
        self.subg = subg
        self.num_chunks = num_chunks
        self.chunk_size = chunk_size
        self.neg_sample_size = neg_sample_size
        self.neg_head = neg_head

    @property
    def head_nid(self):
        return self.subg.head_nid

    @property
    def tail_nid(self):
        return self.subg.tail_nid


def create_neg_subgraph(pos_g, neg_g, chunk_size, neg_sample_size, is_chunked,
                        neg_head, num_nodes):
    """KG models need to know the number of chunks, the chunk size and negative sample size
    of a negative subgraph to perform the computation more efficiently.
    This function tries to infer all of these information of the negative subgraph
    and create a wrapper class that contains all of the information.

    Parameters
    ----------
    pos_g : DGLGraph
        Graph holding positive edges.
    neg_g : DGLGraph
        Graph holding negative edges.
    chunk_size : int
        Chunk size of negative subgrap.
    neg_sample_size : int
        Negative sample size of negative subgrap.
    is_chunked : bool
        If True, the sampled batch is chunked.
    neg_head : bool
        If True, negative_mode is 'head'
        If False, negative_mode is 'tail'
    num_nodes: int
        Total number of nodes in the whole graph.

    Returns
    -------
    ChunkNegEdgeSubgraph
        Negative graph wrapper
    """
    assert neg_g.number_of_edges() % pos_g.number_of_edges() == 0
    # We use all nodes to create negative edges. Regardless of the sampling algorithm,
    # we can always view the subgraph with one chunk.
    if (neg_head and len(neg_g.head_nid) == num_nodes) \
            or (not neg_head and len(neg_g.tail_nid) == num_nodes):
        num_chunks = 1
        chunk_size = pos_g.number_of_edges()
    elif is_chunked:
        # This is probably for evaluation.
        if pos_g.number_of_edges() < chunk_size \
                and neg_g.number_of_edges() % neg_sample_size == 0:
            num_chunks = 1
            chunk_size = pos_g.number_of_edges()
        # This is probably the last batch in the training. Let's ignore it.
        elif pos_g.number_of_edges() % chunk_size > 0:
            return None
        else:
            num_chunks = int(pos_g.number_of_edges() / chunk_size)
        assert num_chunks * chunk_size == pos_g.number_of_edges()
    else:
        num_chunks = pos_g.number_of_edges()
        chunk_size = 1
    return ChunkNegEdgeSubgraph(neg_g, num_chunks, chunk_size,
                                neg_sample_size, neg_head)


class EvalSampler(object):
    """Sampler for validation and testing

    Parameters
    ----------
    g : DGLGraph
        Graph containing KG graph
    edges : tensor
        Seed edges
    batch_size : int
        Batch size of each mini batch.
    neg_sample_size : int
        How many negative edges sampled for each node.
    neg_chunk_size : int
        How many edges in one chunk. We split one batch into chunks.
    mode : str
        Sampling mode.
    number_workers: int
        Number of workers used in parallel for this sampler
    filter_false_neg : bool
        If True, exlucde true positive edges in sampled negative edges
        If False, return all sampled negative edges even there are positive edges
        Default: True
    """

    def __init__(self, g, edges, batch_size, neg_sample_size, neg_chunk_size, mode, num_workers=32,
                 filter_false_neg=True):
        EdgeSampler = getattr(dgl.contrib.sampling, 'EdgeSampler')
        self.sampler = EdgeSampler(g,
                                   batch_size=batch_size,
                                   seed_edges=edges,
                                   neg_sample_size=neg_sample_size,
                                   chunk_size=neg_chunk_size,
                                   negative_mode=mode,
                                   num_workers=num_workers,
                                   shuffle=False,
                                   exclude_positive=False,
                                   relations=g.edata['tid'],
                                   return_false_neg=filter_false_neg)
        self.sampler_iter = iter(self.sampler)
        self.mode = mode
        self.neg_head = 'head' in mode
        self.g = g
        self.filter_false_neg = filter_false_neg
        self.neg_chunk_size = neg_chunk_size
        self.neg_sample_size = neg_sample_size

    def __iter__(self):
        return self

    def __next__(self):
        """Get next batch

        Returns
        -------
        DGLGraph
            Sampled positive graph
        ChunkNegEdgeSubgraph
            Negative graph wrapper
        """
        while True:
            pos_g, neg_g = next(self.sampler_iter)
            if self.filter_false_neg:
                neg_positive = neg_g.edata['false_neg']
            neg_g = create_neg_subgraph(pos_g, neg_g,
                                        self.neg_chunk_size,
                                        self.neg_sample_size,
                                        'chunk' in self.mode,
                                        self.neg_head,
                                        self.g.number_of_nodes())
            if neg_g is not None:
                break

        pos_g.ndata['id'] = pos_g.parent_nid
        neg_g.ndata['id'] = neg_g.parent_nid
        pos_g.edata['id'] = pos_g._parent.edata['tid'][pos_g.parent_eid]
        if self.filter_false_neg:
            neg_g.edata['bias'] = F.astype(-neg_positive, F.float32)
        return pos_g, neg_g

    def reset(self):
        """Reset the sampler
        """
        self.sampler_iter = iter(self.sampler)
        return self


class WikiEvalSampler(object):
    """Sampler for validation and testing for wikikg90M dataset

    Parameters
    ----------
    edges : tensor
        sampled test data
    batch_size : int
        Batch size of each mini batch.
    mode : str
        Sampling mode.
    """

    def __init__(self, edges, batch_size, mode):
        self.edges = edges
        self.batch_size = batch_size
        self.mode = mode
        self.neg_head = 'head' in mode
        self.cnt = 0
        self.mode = 'h,r->t'
        self.num_edges = len(self.edges['h,r->t']['hr'])

    def __iter__(self):
        return self

    def __next__(self):
        """Get next batch

        Returns
        -------
        tensor of size (batch_size, 2)
            sampled head and relation pair
        tensor of size (batchsize, 1)
            the index of the true tail entity
        tensor of size (bath_size, 1001)
            candidates for the tail entities (1001 candidates in total, out of which one is a positive entity)
        """
        if self.cnt == self.num_edges:
            raise StopIteration
        beg = self.cnt
        if self.cnt + self.batch_size > self.num_edges:
            self.cnt = self.num_edges
        else:
            self.cnt += self.batch_size
        return F.tensor(self.edges['h,r->t']['hr'][beg:self.cnt], F.int64), F.tensor(self.edges['h,r->t']['t_correct_index'][beg:self.cnt], F.int64), F.tensor(self.edges['h,r->t']['t_candidate'][beg:self.cnt], F.int64)

    def reset(self):
        """Reset the sampler
        """
        self.cnt = 0
        return self


class EvalDataset(object):
    """Dataset for validation or testing

    Parameters
    ----------
    dataset : KGDataset
        Original dataset.
    args :
        Global configs.
    """

    def __init__(self, g, dataset, args):
        self.num_train = len(dataset.train[0])
        if args.dataset == "wikikg90M":
            self.valid_dict = dataset.valid
            self.num_valid = len(self.valid_dict['h,r->t']['hr'])
        elif dataset.valid is not None:
            self.num_valid = len(dataset.valid[0])
        else:
            self.num_valid = 0
        if args.dataset == "wikikg90M":
            self.test_dict = dataset.test
            self.num_test = len(self.test_dict['h,r->t']['hr'])
        elif dataset.test is not None:
            self.num_test = len(dataset.test[0])
        else:
            self.num_test = 0

        if args.dataset == "wikikg90M":
            print('|valid|:', self.num_valid)
            print('|test|:', self.num_test)
            return

        assert self.num_valid + \
            self.num_test > 1, "we need to have at least validation set or test set."
        self.g = g

        if args.eval_percent < 1:
            self.valid = np.random.randint(0, self.num_valid,
                                           size=(int(self.num_valid * args.eval_percent),)) + self.num_train
        else:
            self.valid = np.arange(
                self.num_train, self.num_train + self.num_valid)
        print('|valid|:', len(self.valid))

        if args.eval_percent < 1:
            self.test = np.random.randint(0, self.num_test,
                                          size=(int(self.num_test * args.eval_percent,)))
            self.test += self.num_train + self.num_valid
        else:
            self.test = np.arange(
                self.num_train + self.num_valid, self.g.number_of_edges())
        print('|test|:', len(self.test))

    def get_edges(self, eval_type):
        """ Get all edges in this dataset

        Parameters
        ----------
        eval_type : str
            Sampling type, 'valid' for validation and 'test' for testing

        Returns
        -------
        np.array
            Edges
        """
        if eval_type == 'valid':
            return self.valid
        elif eval_type == 'test':
            return self.test
        else:
            raise Exception('get invalid type: ' + eval_type)

    def get_dicts(self, eval_type):
        """ Get all edges dict in this dataset

        Parameters
        ----------
        eval_type : str
            Sampling type, 'valid' for validation and 'test' for testing

        Returns
        -------
        dict
            all edges
        """
        if eval_type == 'valid':
            return self.valid_dict
        elif eval_type == 'test':
            return self.test_dict
        else:
            raise Exception('get invalid type: ' + eval_type)

    def create_sampler(self, eval_type, batch_size, neg_sample_size, neg_chunk_size,
                       filter_false_neg, mode='head', num_workers=32, rank=0, ranks=1):
        """Create sampler for validation or testing

        Parameters
        ----------
        eval_type : str
            Sampling type, 'valid' for validation and 'test' for testing
        batch_size : int
            Batch size of each mini batch.
        neg_sample_size : int
            How many negative edges sampled for each node.
        neg_chunk_size : int
            How many edges in one chunk. We split one batch into chunks.
        filter_false_neg : bool
            If True, exlucde true positive edges in sampled negative edges
            If False, return all sampled negative edges even there are positive edges
        mode : str
            Sampling mode.
        number_workers: int
            Number of workers used in parallel for this sampler
        rank : int
            Which partition to sample.
        ranks : int
            Total number of partitions.

        Returns
        -------
        dgl.contrib.sampling.EdgeSampler
            Edge sampler
        """
        edges = self.get_edges(eval_type)
        beg = edges.shape[0] * rank // ranks
        end = min(edges.shape[0] * (rank + 1) // ranks, edges.shape[0])
        edges = edges[beg: end]
        return EvalSampler(self.g, edges, batch_size, neg_sample_size, neg_chunk_size,
                           mode, num_workers, filter_false_neg)

    def create_sampler_wikikg90M(self, eval_type, batch_size, mode='head', rank=0, ranks=1):
        """Create sampler for validation and testing of wikikg90M dataset.

        Parameters
        ----------
        eval_type : str
            Sampling type, 'valid' for validation and 'test' for testing
        batch_size : int
            Batch size of each mini batch.
        mode : str
            Sampling mode.
        rank : int
            Which partition to sample.
        ranks : int
            Total number of partitions.

        Returns
        -------
        dgl.contrib.sampling.EdgeSampler
            Edge sampler
        """
        edges = self.get_dicts(eval_type)
        new_edges = {}

        assert 'tail' in mode

        """
        This function will split the edges into total number of partitions parts. And then calculate the
        corresponding begin and end index for each part to create evaluate sampler.
        """
        beg = edges['h,r->t']['hr'].shape[0] * rank // ranks
        end = min(edges['h,r->t']['hr'].shape[0] * (rank + 1) //
                  ranks, edges['h,r->t']['hr'].shape[0])
        new_edges['h,r->t'] = {'hr': edges['h,r->t']['hr'][beg:end],
                               't_candidate': edges['h,r->t']['t_candidate'][beg:end],
                               #    't_correct_index': edges['h,r->t']['t_correct_index'][beg:end]
                               }
        if 't_correct_index' in edges['h,r->t']:
            new_edges['h,r->t']['t_correct_index'] = edges['h,r->t']['t_correct_index'][beg:end]
        else:
            new_edges['h,r->t']['t_correct_index'] = np.zeros(
                end-beg, dtype=np.short)

        return WikiEvalSampler(new_edges, batch_size, mode)


class NewBidirectionalOneShotIterator:
    """Grouped sampler iterator

    Parameters
    ----------
    dataloader_head : dgl.contrib.sampling.EdgeSampler
        EdgeSampler in head mode
    dataloader_tail : dgl.contrib.sampling.EdgeSampler
        EdgeSampler in tail mode
    neg_chunk_size : int
        How many edges in one chunk. We split one batch into chunks.
    neg_sample_size : int
        How many negative edges sampled for each node.
    is_chunked : bool
        If True, the sampled batch is chunked.
    num_nodes : int
        Total number of nodes in the whole graph.
    """

    def __init__(self, dataloader_head, dataloader_tail, neg_chunk_size, neg_sample_size,
                 is_chunked, num_nodes, has_edge_importance=False, renumbering_dict=None, real_train=False):
        self.sampler_head = dataloader_head
        self.sampler_tail = dataloader_tail
        self.iterator_head = self.one_shot_iterator(dataloader_head, neg_chunk_size,
                                                    neg_sample_size, is_chunked,
                                                    True, num_nodes, has_edge_importance,
                                                    renumbering_dict, real_train)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail, neg_chunk_size,
                                                    neg_sample_size, is_chunked,
                                                    False, num_nodes, has_edge_importance,
                                                    renumbering_dict, real_train)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            pos_g, neg_g = next(self.iterator_head)
        else:
            pos_g, neg_g = next(self.iterator_tail)
        return pos_g, neg_g

    @staticmethod
    def one_shot_iterator(dataloader, neg_chunk_size, neg_sample_size, is_chunked,
                          neg_head, num_nodes, has_edge_importance=False, renumbering_dict=None, real_train=False):
        while True:
            for pos_g, neg_g in dataloader:
                neg_g = create_neg_subgraph(pos_g, neg_g, neg_chunk_size, neg_sample_size,
                                            is_chunked, neg_head, num_nodes)
                if neg_g is None:
                    continue

                # OLD ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
                # pos_g.ndata['id'] = pos_g.parent_nid
                # neg_g.ndata['id'] = neg_g.parent_nid
                #######################

                # NEW ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
                pos_g.ndata['id'] = pos_g._parent.ndata[NID][pos_g.parent_nid]
                neg_g.ndata['id'] = neg_g._parent.ndata[NID][neg_g.parent_nid]

                if renumbering_dict is not None:
                    assert False, "dont touch"
                    pos_g.ndata['id'] = renumbering_dict[pos_g.ndata['id']]
                    neg_g.ndata['id'] = renumbering_dict[neg_g.ndata['id']]
                #######################

                pos_g.edata['id'] = pos_g._parent.edata['tid'][pos_g.parent_eid]
                if has_edge_importance:
                    pos_g.edata['impts'] = pos_g._parent.edata['impts'][pos_g.parent_eid]
                yield pos_g, neg_g

            if not real_train:
                break
