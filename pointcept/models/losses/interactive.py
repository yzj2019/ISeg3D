"""
Interactive Losses

Please cite our work if the code is helpful to you.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .builder import LOSSES, build_criteria
from ..utils import offset2batch


@LOSSES.register_module()
class Agile3DLoss(nn.Module):
    def __init__(self,
                 clicks_mask_loss_cfg = None,
                 clicks_cls_loss_cfg = None,
                 mask_loss_cfg = None,
                 ):
        super(Agile3DLoss, self).__init__()
        self.clicks_mask_loss = build_criteria(clicks_mask_loss_cfg)
        self.clicks_cls_loss = build_criteria(clicks_cls_loss_cfg)
        self.mask_loss = build_criteria(mask_loss_cfg)


    def get_clicks_target(self, clicks, target, instance, pred_masks, batch):
        '''
        构建以click为中心的监督信号
        - clicks: N_clicks, 详见 clicker
        - target: tensor on gpu, shape as (N_points,)
        - instance: tensor on gpu, shape as (N_points,), masks target
        - pred_masks: N_points x N_clicks, False or True
        - batch: which batch the point belongs to , shape as (N_points,)

        Return:
        - masks_target: N_points x N_clicks, 一个点不一定只对应一个click
        - cls_target: N_clicks
        - class_tag_to_mask: batch_id, class_tag -> 1d mask
        '''
        class_tag_to_mask = {}
        masks_target = []
        index = []
        for i in range(len(clicks)):
            for j in clicks[i].keys():
                click = clicks[i][j]['init']
                index.append(click.index)
                class_tag = click.class_tag
                batch_id = click.batch_id
                mask = torch.zeros((target.shape[0],)).float()
                mask[((instance == class_tag) * (batch == batch_id)).bool()] = 1.0
                masks_target.append(mask)
                if batch_id not in class_tag_to_mask.keys():
                    class_tag_to_mask[batch_id] = {}
                class_tag_to_mask[batch_id][class_tag] = pred_masks[:,i]

        masks_target = torch.stack(masks_target, dim=1).to(target.device)
        index = torch.Tensor(index).long()
        cls_target = target[index]

        return masks_target, cls_target, class_tag_to_mask

    def forward(self, clicks_dict, pcd_dict):
        '''
        Arguments
        - masks_heatmap: 场景中点属于哪个click对应的mask, 归一化后的, N_points x N_clicks
        - cls_logits: click点对应的类别, N_clicks x num_classes
        - pred_last: 优化前的prediction, N_points x num_classes
        - pred_refine: 优化后的prediction, N_points x num_classes
        - clicks: 点击list, N_clicks
        - clicks_from_instance: 是否通过instance来构建click
        - pcd_dict:
        '''
        clicks = clicks_dict["clicks"]
        pred_last = clicks_dict["pred_last"]
        pred_refine = clicks_dict["pred_refine"]
        mask_threshold = clicks_dict["mask_threshold"]
        last_masks_heatmap, last_cls_logits = pred_last['masks_heatmap'], pred_last['cls_logits']
        masks_heatmap, cls_logits = pred_refine['masks_heatmap'], pred_refine['cls_logits']
        pred_refine = masks_heatmap @ cls_logits
        clicks_from_instance = clicks_dict["clicks_from_instance"]

        target = pcd_dict["segment"]
        grid_coord = pcd_dict["grid_coord"]
        batch = offset2batch(pcd_dict["offset"])

        masks_target, cls_target, class_tag_to_mask = self.get_clicks_target(clicks, target, 
                                                          pcd_dict["instance"] if clicks_from_instance else target, 
                                                          masks_heatmap > mask_threshold,
                                                          batch)         # 可改这里成instance监督

        clicks_mask_loss = self.clicks_mask_loss(masks_heatmap, masks_target)        # binary ce, 用focal loss 来算 bce
        clicks_cls_loss = self.clicks_cls_loss(cls_logits, cls_target)            # reduction="mean", click之间归一化
        mask_loss = self.mask_loss(pred_refine, target)
        loss = clicks_mask_loss + clicks_cls_loss + mask_loss
        # distance = torch.ones((grid_coord.shape[0],)).cuda().to(grid_coord.device)         # 保证在同一个device上
        # distance = distance / grid_coord.shape[0]
        # # TODO 批量处理
        # for click in clicks:
        #     # 需要按照batch内部处理
        #     # 保证mask与grid_coord.shape[0]一致
        #     batch_mask = batch == click.batch_id
        #     mask = batch_mask
        #     # mask = target == click.class_tag
        #     # mask = torch.logical_and(mask, batch_mask).bool()
        #     distance_to_click = ((grid_coord[mask] - click.coords.to(grid_coord.device)) ** 2).sum(1)
        #     # distance[mask] += F.softmax(distance_to_click.float(), -1)       # 归一化, 和是1, 代替取均值了
        #     distance[mask] += distance_to_click
        # distance = F.softmax(distance.float(), -1).view(1,-1)       # 归一化, 和是1, 代替取均值了
        # # distance = distance.float().view(1,-1)
        # # loss = self.ce_loss(pred_refine, target) + self.dice_loss(pred_refine, target)
        # loss = self.ce_loss(pred_refine, target).view(-1,1)
        # loss = distance @ loss
        # loss = self.ce_loss(pred_refine, target)

        info_dict = dict(
            clicks_mask_loss=clicks_mask_loss, 
            clicks_cls_loss=clicks_cls_loss, 
            mask_loss=mask_loss,
            masks_target=masks_target,
            cls_target=cls_target,
            class_tag_to_mask=class_tag_to_mask)
        return loss, info_dict
