"""
Interactive Segmentation Hook

Author: Zijian Yu (yzj18@mail.ustc.edu.cn)
Please cite our work if the code is helpful to you.
"""

import numpy as np
import random
import torch
import torch.distributed as dist
import pointops
from uuid import uuid4

import pointcept.utils.comm as comm
from pointcept.utils.misc import intersection_and_union_gpu

from .default import HookBase
from .builder import HOOKS
from pointcept.utils.clicker import CLICKER



@HOOKS.register_module()
class ClicksMaker(HookBase):
    def __init__(self, clicker_cfg, pcd_only=True):
        self.clicker_cfg = clicker_cfg
        self.pcd_only = pcd_only

    def before_step(self):
        '''创建clicker, 获取image, 转所有需要的数据到gpu(因为Trainer.run_step()里的转gpu过程不支持)'''
        # clicker
        clicker = CLICKER.build(self.clicker_cfg)
        # 考虑clicker参数的调度
        clicker.max_iou = 0.3 + 0.5 * self.trainer.epoch / self.trainer.cfg.eval_epoch      # 训练时调整取next click的阈值, 为了那些iou不好的类别
        clicker.jitter_prob = 0.1 + ((self.trainer.epoch / self.trainer.cfg.eval_epoch)) * 0.9
        # clicker.choose_prob = (4 * self.trainer.epoch / self.trainer.cfg.eval_epoch) % 1      # 15终止了, 16~3, 18~4, 效果还行, 但也没想象的那么好, 
        # clicker.choose_prob = abs(np.cos(np.pi*(-1/2 + 3 * self.trainer.epoch / self.trainer.cfg.eval_epoch)))    # 17,效果不好
        # 考虑在同一个epoch中不同iter里混用, mask3d-19, 训练不太充分, 比16~18好, 不如20
        # iter=self.trainer.comm_info["iter"] + 1, max_iter=len(self.trainer.train_loader)
        # clicker.choose_prob = (self.trainer.epoch / self.trainer.cfg.eval_epoch + \
        #                        4 * (self.trainer.comm_info["iter"] + 1)/len(self.trainer.train_loader)) % 1
        # 随机取, 比如abs高斯分布整除1, 保证取0概率大一点
        # clicker.choose_prob = abs(random.gauss(0, 0.5)) % 1         # 20, 
        clicker.cut = self.trainer.epoch % 3
        # 取dict
        pcd_dict = self.trainer.comm_info["input_dict"]["pcd_dict"]
        if self.pcd_only:
            img_dict, fusion = None, None
            dict_list = [pcd_dict]
        else:
            # 同样可以通过offset来表明属于哪个click
            img_dict, fusion = self.trainer.train_loader.dataset.get_img_fusion(pcd_dict)
            dict_list = [pcd_dict, img_dict, fusion]
        # 转tensor, gpu: 这里由于在input_dict上加了一层pcd_dict, 所以原先写在trainer的转gpu的代码失效了
        for d in dict_list:
            for key in d.keys():
                # collate_fn过了, pcd_dict下面一层就是tensor, 而不是各个batch; 由于不同batch的N_points不一定一样, 所以不能简单加一个dim
                if isinstance(d[key], torch.Tensor):
                    d[key] = d[key].cuda(non_blocking=True)
        # 要确保 img_dict 和 fusion 之下需要的数据为 tensor
        self.trainer.comm_info["input_dict"]["img_dict"] = img_dict
        self.trainer.comm_info["input_dict"]["fusion"] = fusion
        self.trainer.comm_info["input_dict"]["clicker"] = clicker


@HOOKS.register_module()
class InteractiveSemSegEvaluator(HookBase):
    def __init__(self, clicker_cfg, pcd_only=True):
        self.clicker_cfg = clicker_cfg
        self.pcd_only = pcd_only

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        for i, input_dict in enumerate(self.trainer.val_loader):
            # clicker
            clicker = CLICKER.build(self.clicker_cfg)
            # 取dict
            pcd_dict = input_dict["pcd_dict"]
            if self.pcd_only:
                img_dict, fusion = None, None
                dict_list = [pcd_dict]
            else:
                img_dict, fusion = self.trainer.train_loader.dataset.get_img_fusion(pcd_dict)
                dict_list = [pcd_dict, img_dict, fusion]
            # 转gpu
            for d in dict_list:
                for key in d.keys():
                    if isinstance(d[key], torch.Tensor):
                        d[key] = d[key].cuda(non_blocking=True)

            with torch.no_grad():
                input_dict["img_dict"] = img_dict
                input_dict["fusion"] = fusion
                input_dict["clicker"] = clicker
                output_dict = self.trainer.model(input_dict)
            output_last = output_dict["seg_logits_last"]        # one click refine前的logits
            output = output_dict["seg_logits"]                  # refine后的
            loss = output_dict["loss"]
            pred = output.max(1)[1]
            pcd_dict = input_dict["pcd_dict"]
            segment = pcd_dict["segment"]

            if "origin_coord" in pcd_dict.keys():
                idx, _ = pointops.knn_query(1, pcd_dict["coord"].float(), pcd_dict["offset"].int(),
                                            pcd_dict["origin_coord"].float(), pcd_dict["origin_offset"].int())
                pred = pred[idx.flatten().long()]
                segment = pcd_dict["origin_segment"]
            intersection, union, target = \
                intersection_and_union_gpu(
                    pred, segment, self.trainer.cfg.data.num_classes, self.trainer.cfg.data.ignore_index)
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            # Here there is no need to sync since sync happened in dist.all_reduce
            self.trainer.storage.put_scalar("val_intersection", intersection)
            self.trainer.storage.put_scalar("val_union", union)
            self.trainer.storage.put_scalar("val_target", target)
            self.trainer.storage.put_scalar("val_loss", loss.item())
            info = "Val: [{iter}/{max_iter}] ".format(iter=i + 1, max_iter=len(self.trainer.val_loader))
            if "origin_coord" in pcd_dict.keys():
                info = "Interp. " + info
            self.trainer.logger.info(info + "Loss {loss:.4f} ".format(iter=i + 1,
                                                                      max_iter=len(self.trainer.val_loader),
                                                                      loss=loss.item()))
        loss_avg = self.trainer.storage.history("val_loss").avg
        intersection = self.trainer.storage.history("val_intersection").total
        union = self.trainer.storage.history("val_union").total
        target = self.trainer.storage.history("val_target").total
        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)
        m_iou = np.mean(iou_class)
        m_acc = np.mean(acc_class)
        all_acc = sum(intersection) / (sum(target) + 1e-10)
        self.trainer.logger.info("Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
            m_iou, m_acc, all_acc))
        for i in range(self.trainer.cfg.data.num_classes):
            self.trainer.logger.info("Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                idx=i, name=self.trainer.cfg.data.names[i], iou=iou_class[i], accuracy=acc_class[i]))
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mIoU", m_iou, current_epoch)
            self.trainer.writer.add_scalar("val/mAcc", m_acc, current_epoch)
            self.trainer.writer.add_scalar("val/allAcc", all_acc, current_epoch)
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = m_iou  # save for saver
        self.trainer.comm_info["current_metric_name"] = "mIoU"  # save for saver

    def after_train(self):
        self.trainer.logger.info("Best {}: {:.4f}".format(
            "mIoU", self.trainer.best_metric_value))
