import os
import torch as th


def diff_objs(new_tensor, name):
    return
    file_name = f"/tmp/cached_tensor_{name}.pkl"
    if os.path.exists(file_name):
        old_tensor = th.load(file_name)
        if type(old_tensor) is th.Tensor:
            if not (old_tensor == new_tensor).all():
                raise Exception(f"tensor diff, {name}")
        else:
            if not (old_tensor == new_tensor):
                raise Exception(f"obj diff, {name}")
    else:
        th.save(new_tensor, file_name)


def diff_presampling(pos_g, neg_g, rank, step):
    nids = pos_g.ndata['id']
    diff_objs(nids, f"xxxnids-{rank}-{step}")

    if neg_g.neg_head:
        neg_head_ids = neg_g.ndata['id'][neg_g.head_nid]
        diff_objs(neg_head_ids, f"xxxxneg_head_ids-{rank}-{step}")
    else:
        neg_tail_ids = neg_g.ndata['id'][neg_g.tail_nid]
        diff_objs(neg_tail_ids, f"xxxxneg_tail_ids-{rank}-{step}")


def diff_forward(pos_g, neg_g, rank, step):
    def diff_tensor(new_tensor, name):
        file_name = f"/tmp/cached_tensor_{name}_{rank}_{step}.pkl"
        if os.path.exists(file_name):
            old_tensor = th.load(file_name)
            assert (old_tensor == new_tensor).all()
        else:
            th.save(new_tensor, file_name)

    nids = pos_g.ndata['id']
    # model.log_tensor(nids)
    # diff_tensor(nids, "nids")

    if neg_g.neg_head:
        neg_head_ids = neg_g.ndata['id'][neg_g.head_nid]
        # model.log_tensor(neg_head_ids)
        # diff_tensor(neg_head_ids, "neg_head_ids")
    else:
        neg_tail_ids = neg_g.ndata['id'][neg_g.tail_nid]
        # model.log_tensor(neg_tail_ids)
        # diff_tensor(neg_tail_ids, "neg_tail_ids")
