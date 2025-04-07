"""
Instance Segmentation / Interactive Segmentation Engine

Author: Zijian Yu (https://github.com/yzj2019)
Please cite our work if the code is helpful to you.
"""

import torch
import torch.utils.data
from functools import partial

from .train import Trainer, TRAINERS
from .test import TesterBase, TESTERS
from .defaults import worker_init_fn
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, collate_fn_iseg


@TRAINERS.register_module(name="InsSegTrainer")
class InsSegTrainer(Trainer):
    """instance segmentation trainer, collate_fn 换成 instance reid 的"""

    def __init__(self, cfg):
        super(InsSegTrainer, self).__init__(cfg)

    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)
        # TODO 修改 VALID_ASSETS index_valid_keys(可选), 添加 sampled_index
        train_data.VALID_ASSETS = train_data.VALID_ASSETS + [
            "sampled_idx_fps",
            "sampled_idx_rand",
        ]
        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        # 因为在用DistributedSampler实例化sampler的时候就已经shuffle过数据了
        # 所以在定义dataloader的时候不需要再将shuffle设置为True
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=partial(
                collate_fn_iseg, instance_ignore_label=self.cfg.instance_ignore_label
            ),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=True,
            persistent_workers=True,
        )
        return train_loader

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset(self.cfg.data.val)
            # TODO 修改 VALID_ASSETS index_valid_keys(可选), 添加 sampled_index
            val_data.VALID_ASSETS = val_data.VALID_ASSETS + [
                "sampled_idx_fps",
                "sampled_idx_rand",
            ]
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=self.cfg.num_worker_per_gpu,
                pin_memory=True,
                sampler=val_sampler,
                collate_fn=partial(
                    collate_fn_iseg,
                    instance_ignore_label=self.cfg.instance_ignore_label,
                ),
            )
        return val_loader


@TESTERS.register_module()
class InsSegTester(TesterBase):
    """instance segmentation tester"""

    def __init__(self, cfg):
        super(InsSegTester, self).__init__(cfg)

    def test(self):
        pass

    def collate_fn(self, batch):
        return batch
