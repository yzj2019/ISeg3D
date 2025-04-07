"""
Instance Segmentation / Interactive Segmentation dataset utils

Author: Zijian Yu (https://github.com/yzj2019)
Please cite our work if the code is helpful to you.
"""

import random
import torch
from collections.abc import Mapping
from ..utils import collate_fn


def collate_fn_iseg(batch, instance_ignore_label=-1):
    """instance reid for minibatch"""
    assert isinstance(
        batch[0], Mapping
    )  # currently, only support input_dict, rather than input_list
    batch = collate_fn(batch)
    # reid, 避免 batch 中 instance 的id重复
    if "instance" in batch.keys():
        b = batch["offset"].shape[0]
        r, id_max = 0, 0
        for i in range(b):
            l = r
            r = batch["offset"][i]
            ignore_mask = batch["instance"][l:r] != instance_ignore_label
            batch["instance"][l:r][ignore_mask] += id_max
            id_max = batch["instance"][l:r].max() + 1
    return batch
