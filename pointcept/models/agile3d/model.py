"""
懒得再写一版纯点云的了, 就这样了
"""

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np

from ..builder import MODELS, build_model

from ..losses import build_criteria, LOSSES

from pointcept.utils.misc import intersection_and_union_gpu


@MODELS.register_module()
class Agile3dSegmentor(nn.Module):
    def __init__(
        self,
        pcd_backbone=None,
        mask_decoder=None,
        loss=None,
        semantic=True,
        clicks_from_instance=False,
        max_train_iter=3,
        mask_threshold=0.0,
    ):
        """
        interactive的segmentor, 返回没做softmax的结果
        - 直接由backbone预测seg的结果, 训练模式则self.training设为True, 其它时候返回backbone的输出和criteria的losses
        - pcd_backbone和img_backbone默认直接在build_model构建类的时候, __init__里加载预训练参数
        - pcd_backbone和img_backbone默认在forward过程中更新embedding
        - pcd_backbone和img_backbone需要有get_embedding方法, 获取点云和图像的embedding
        - semantic: 是否为语义分割任务
        - clicks_from_instance: 是否利用instance gt去构建click
        """
        super().__init__()
        self.pcd_backbone = build_model(pcd_backbone)
        self.mask_decoder = build_model(mask_decoder)
        self.loss = LOSSES.build(loss)
        self.num_classes = mask_decoder.num_classes
        self.max_train_iter = max_train_iter
        self.mask_threshold = mask_threshold
        self.semantic = semantic
        self.clicks_from_instance = clicks_from_instance

    def refine(self, pred, pcd_dict, clicks):
        """
        场景点云的预测结果, 用该场景的RGBD去refine
        - pred: 上一次的预测logits, tensor, N_points x num_classes
        - pcd_dict: fragment_list 中的一个, 做了voxelize, ['coord', 'discrete_coord', 'index', 'offset', 'feat']
        - clicks

        Returns:
        - masks: 归一化后的
        - cls_logits: 每个click做分类
        - new_logits:

        TODO:
        - 如何利用pred?
        """
        if len(clicks) == 0:
            return pred
        masks_heatmap, cls_logits = self.mask_decoder(
            pcd_dict, clicks, pred
        )  # N_points x N_clicks, N_clicks x num_classes

        if not self.training:
            masks_heatmap[masks_heatmap <= self.mask_threshold] = 0  # 非训练时对mask做截断
        new_logits = (
            masks_heatmap @ cls_logits
        )  # N_points x num_classes, 相当于point wise地叠加了多个mask

        # new_logits = cls_logits[masks,:]          # N_points x num_classes
        return masks_heatmap, cls_logits, new_logits  # N_points x num_classes

    def forward_train(self, pcd_dict, clicker, img_dict, fusion):
        # 训练策略写在这里了
        # with torch.no_grad():
        # 前 self.max_train_iter-1 次, 都不用梯度回传, 只有最后一次会回传梯度
        backbone_res_dict = self.pcd_backbone(pcd_dict)
        pcd_pred = backbone_res_dict["seg_logits"]  # (n, k)
        pcd_dict["feature_maps"] = backbone_res_dict[
            "feature_maps"
        ]  # pcd embedding list, [(N_points, embedding_dims[i])]

        clicks = clicker.make_init_clicks()  # 获取 clicks simulation
        masks_pred, cls_logits, pred_refine = self.refine(
            pcd_pred, pcd_dict, clicks
        )  # 优化pred
        clicks_dict = dict(
            masks_pred=masks_pred,
            cls_logits=cls_logits,
            pred_last=pcd_pred,
            pred_refine=pred_refine,
            clicks=clicks,
            clicks_from_instance=self.clicks_from_instance,
        )
        loss, info_dict = self.loss(clicks_dict, pcd_dict)

        # 后面需要用 masks_pred 梯度, 所以这里建个新的, 截断用来算 per click mask iou
        masks = masks_pred.clone().detach()
        masks[masks <= self.mask_threshold] = 0
        masks[masks > 0] = 1
        masks_target = info_dict["masks_target"]
        cls_target = info_dict["cls_target"]
        cls_pred = F.softmax(cls_logits, -1).max(1)[1]
        intersection, union, target = intersection_and_union_gpu(
            cls_pred, cls_target, 20, -1
        )
        clicks_cls_iou = intersection / (union + 1e-10)
        clicks_cls_iou = clicks_cls_iou[union != 0].mean()
        clicks_masks_intersection = (masks * masks_target).sum(0)
        clicks_masks_union = ((masks + masks_target) > 0).float().sum(0)
        clicks_masks_iou = clicks_masks_intersection / clicks_masks_union
        clicks_masks_ap50 = torch.Tensor(
            [
                (clicks_masks_iou[clicks_masks_iou > 0.5]).shape[0]
                / clicks_masks_iou.shape[0]
            ]
        )

        # iter for max_train_iter
        # for i in range(self.max_train_iter-2):
        #     clicker.update_pred(pred_refine.max(1)[1].data)
        #     clicks = clicker.make_next_click()
        #     pred_refine = self.refine(pred_refine, pcd_dict, clicks)
        #     loss += self.loss(pred_refine, pcd_dict, clicks)
        # clicker.update_pred(pred_refine.max(1)[1].data)
        # clicks = clicker.make_next_click()
        # pred_refine = self.refine(pred_refine, pcd_dict, clicks)
        # seg_logits = pred_refine
        # loss += self.loss(seg_logits, pcd_dict, clicks)

        return dict(
            loss=loss,
            clicks_mask_loss=info_dict["clicks_mask_loss"],
            clicks_cls_loss=info_dict["clicks_cls_loss"],
            mask_loss=info_dict["mask_loss"],
            clicks_masks_iou=clicks_masks_iou.mean(),
            clicks_masks_ap50=clicks_masks_ap50,
            clicks_cls_iou=clicks_cls_iou,
        )

    def forward_eval(self, pcd_dict, clicker, img_dict, fusion):
        backbone_res_dict = self.pcd_backbone(pcd_dict)
        pcd_pred = backbone_res_dict["seg_logits"]  # (n, k)
        pcd_dict["feature_maps"] = backbone_res_dict[
            "feature_maps"
        ]  # pcd embedding list, [(N_points, embedding_dims[i])]

        clicks = clicker.make_init_clicks()  # 获取 clicks simulation
        masks_pred, cls_logits, pred_refine = self.refine(
            pcd_pred, pcd_dict, clicks
        )  # 优化pred
        clicks_dict = dict(
            masks_pred=masks_pred,
            cls_logits=cls_logits,
            pred_last=pcd_pred,
            pred_refine=pred_refine,
            clicks=clicks,
            clicks_from_instance=self.clicks_from_instance,
        )
        loss, info_dict = self.loss(clicks_dict, pcd_dict)
        return dict(loss=loss, seg_logits=pred_refine, seg_logits_last=pcd_pred)

    def forward_test(self, pcd_dict, clicker, img_dict, fusion, pcd_pred):
        """TODO 测试阶段, 没有真值, 随机取点并聚合mask"""
        raise NotImplementedError

    def forward(self, input_dict):
        """分开写成forward_train"""
        pcd_dict = input_dict["pcd_dict"]
        clicker = input_dict["clicker"]
        img_dict = input_dict["img_dict"]
        fusion = input_dict["fusion"]

        res = self.pcd_backbone(pcd_dict)
        pcd_dict["pcd_pred"] = res["seg_logits"]
        pcd_dict["feature_maps"] = res["feature_maps"]

        # train
        if self.training:
            clicker.set_state(
                "train",
                [pcd_dict],  # 这里加一层, 因为需要的是 fragment_list
                img_dict=img_dict,
                fusion=fusion,
            )
            return self.forward_train(pcd_dict, clicker, img_dict, fusion)
        # val
        elif "segment" in pcd_dict.keys():
            clicker.set_state("val", [pcd_dict], img_dict=img_dict, fusion=fusion)
            return self.forward_eval(pcd_dict, clicker, img_dict, fusion)
        # test
        else:
            clicker.set_state(
                "test",
                [pcd_dict],
                semantic=input_dict["semantic"],
                instance=input_dict["instance"],
                img_dict=img_dict,
                fusion=fusion,
            )
            return self.forward_test(pcd_dict, clicker, img_dict, fusion)
