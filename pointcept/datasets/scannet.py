"""
ScanNet20 / ScanNet200 / ScanNet Data Efficient Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import glob
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence

from pointcept.utils.logger import get_root_logger
from pointcept.utils.cache import shared_dict
from .builder import DATASETS
from .defaults import DefaultDataset
from .transform import Compose, TRANSFORMS
from .preprocessing.scannet.meta_data.scannet200_constants import (
    VALID_CLASS_IDS_20,
    VALID_CLASS_IDS_200,
)


@DATASETS.register_module()
class ScanNetDataset(DefaultDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "segment20",
        "instance",
    ]
    class2id = np.array(VALID_CLASS_IDS_20)

    def __init__(
        self,
        lr_file=None,
        la_file=None,
        **kwargs,
    ):
        self.lr = np.loadtxt(lr_file, dtype=str) if lr_file is not None else None
        self.la = torch.load(la_file) if la_file is not None else None
        super().__init__(**kwargs)

    def get_data_list(self):
        if self.lr is None:
            data_list = super().get_data_list()
        else:
            data_list = [
                os.path.join(self.data_root, "train", name) for name in self.lr
            ]
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        if self.cache:
            cache_name = f"pointcept-{name}"
            return shared_dict(cache_name)

        data_dict = {}
        assets = os.listdir(data_path)
        for asset in assets:
            if not asset.endswith(".npy"):
                continue
            if asset[:-4] not in self.VALID_ASSETS:
                continue
            data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))
        data_dict["name"] = name
        data_dict["coord"] = data_dict["coord"].astype(np.float32)
        data_dict["color"] = data_dict["color"].astype(np.float32)
        data_dict["normal"] = data_dict["normal"].astype(np.float32)

        if "segment20" in data_dict.keys():
            data_dict["segment"] = (
                data_dict.pop("segment20").reshape([-1]).astype(np.int32)
            )
        elif "segment200" in data_dict.keys():
            data_dict["segment"] = (
                data_dict.pop("segment200").reshape([-1]).astype(np.int32)
            )
        else:
            data_dict["segment"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )

        if "instance" in data_dict.keys():
            data_dict["instance"] = (
                data_dict.pop("instance").reshape([-1]).astype(np.int32)
            )
        else:
            data_dict["instance"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )
        if self.la:
            sampled_index = self.la[self.get_data_name(idx)]
            mask = np.ones_like(data_dict["segment"], dtype=bool)
            mask[sampled_index] = False
            data_dict["segment"][mask] = self.ignore_index
            data_dict["sampled_index"] = sampled_index
        return data_dict

    # modified for ins seg
    def get_idx_by_name(self, scene_name):
        '''scene_name like 'scene0011_00'''
        return [self.data_list.index(s) for s in self.data_list if scene_name in s][0]

    def get_data_by_name(self, scene_name):
        '''scene_name like 'scene0011_00'''
        idx = self.get_idx_by_name(scene_name)
        return self.get_data(idx)

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)]).split(".")[0]

    def prepare_train_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        segment = data_dict.pop("segment")
        data_dict = self.transform(data_dict)
        # modified for ins seg
        coord = data_dict["coord"]
        if "instance" in data_dict.keys():
            instance = data_dict.pop("instance")
        else:
            instance = None
        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))

        input_dict_list = []
        for data in data_dict_list:
            data_part_list = self.test_voxelize(data)
            for data_part in data_part_list:
                if self.test_crop:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                input_dict_list += data_part

        for i in range(len(input_dict_list)):
            input_dict_list[i] = self.post_transform(input_dict_list[i])
        data_dict = dict(
            fragment_list=input_dict_list, segment=segment, instance=instance, name=self.get_data_name(idx), coord=coord
        )
        return data_dict

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop


@DATASETS.register_module()
class ScanNet200Dataset(ScanNetDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "segment200",
        "instance",
    ]
    class2id = np.array(VALID_CLASS_IDS_200)
