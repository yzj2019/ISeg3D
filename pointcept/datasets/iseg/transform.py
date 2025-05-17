"""
Instance Segmentation / Interactive Segmentation dataset augmentation

Author: Zijian Yu (https://github.com/yzj2019)
Please cite our work if the code is helpful to you.
"""

import random
import torch
import numpy as np
from collections.abc import Mapping
from ..transform import TRANSFORMS, index_operator


"""
省得改写transform了, 为了满足预采样的需要
- 直接复用 sampled_index 来避免丢弃采样点
    - RandomDropout、GridSample(train)
- 将idx转为mask, 来适应需要做 index_operator 的transform
    - index_operator 是需要大幅度换顺序的
        - 要求处理的数据, 在 dim0 上位置一致, 直接 indexing
    - 增加 index_valid_keys, 让 index_operator 直接处理
    - SphereCrop、ShufflePoint、CropBoundary
- 其它 transform 只需要改变值, 不会改变点排序, 所以不需要转换
- 保留 mask, 方便 concat mini batch, 不需要 reindex
"""


@TRANSFORMS.register_module()
class SampledIndex2Mask(object):
    def __call__(self, data_dict):
        if "sampled_index" in data_dict:
            mask = np.zeros_like(data_dict["segment"]).astype(bool)
            mask[data_dict["sampled_index"]] = True
            data_dict["sampled_mask"] = mask
            data_dict["index_valid_keys"].append("sampled_mask")
        return data_dict


@TRANSFORMS.register_module()
class InstanceCrop(object):
    """随机将一部分 instance 整体 crop 掉, 改变 instance 的出现频率"""

    def __init__(
        self,
        sample_rate=None,
        mode="random",
        semantic_ignore=-1,
        instance_ignore=-1,
        semantic_background=(0, 1),
    ):
        self.sample_rate = sample_rate
        assert mode in ["random", "background"]
        self.mode = mode
        self.semantic_ignore = semantic_ignore
        self.instance_ignore = instance_ignore
        self.semantic_background = semantic_background

    def __call__(self, data_dict):
        # TODO
        pass
