"""
懒得再写一版纯点云的了, 就这样了
"""

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np
import itertools
from addict import Dict

from .positional_embedding import PositionEmbeddingCoordsSine, PositionalEncoding3D
from ..builder import MODELS, build_model
from ..losses import build_criteria, LOSSES

from pointcept.models.utils.structure import Point  # 使用 Point 类, DefaultSegmentorv2
from pointcept.utils_iseg.structure import Scene, Query
from pointcept.utils_iseg.matcher import HungarianMatcher
from pointcept.utils_iseg.ins_seg import (
    mask_to_id,
    id_to_mask,
    unique_id,
    get_pred,
    get_target,
    softgroup_post_process,
)


@MODELS.register_module("Mask3dSegmentor-learnable")
class Mask3dSegmentor(nn.Module):
    def __init__(
        self,
        pcd_backbone=None,
        mask_decoder=None,
        loss=None,
        matcher_cfg=None,
        mask_threshold=0.0,
        num_query=200,
        topk_per_scene=200,
        semantic_ignore=-1,
        instance_ignore=-1,
        semantic_background=(0, 1),
    ):
        """with learnable latent query
        - pcd_backbone: 点云backbone
        - mask_decoder: mask decoder
        - loss: 损失函数
        - mask_threshold: 测试时 mask 截断 threshold
        - num_query: 查询点数
        - semantic_ignore: 忽略的 semantic id
        - instance_ignore: 忽略的 instance id
        """
        super().__init__()
        self.pcd_backbone = build_model(pcd_backbone)
        self.mask_decoder = build_model(mask_decoder)
        self.num_classes = self.mask_decoder.num_classes
        self.embedding_dim = self.mask_decoder.embedding_dim
        self.loss = LOSSES.build(loss)
        self.matcher = HungarianMatcher(**matcher_cfg)
        self.mask_threshold = mask_threshold
        self.semantic_ignore = semantic_ignore
        self.instance_ignore = instance_ignore
        self.semantic_background = semantic_background
        self.topk_per_scene = topk_per_scene
        # latent query
        self.num_query = num_query
        self.latent_query = nn.Embedding(
            num_query, self.mask_decoder.query_embedding_dim
        )
        # positional encoding
        self.pos_enc = PositionEmbeddingCoordsSine(
            pos_type="fourier",
            d_pos=self.embedding_dim,
            gauss_scale=1.0,
            normalize=True,
        )

    def get_batched_pe(self, coords, batch):
        """
        获取 positional encoding
        - coords: (N_point, 3)
        - batch: (N_point,)
        """
        bs = batch.max().item() + 1
        pe_list = []
        for b in range(bs):
            mask = batch == b
            mins = coords[mask].min(dim=0)[0]
            maxs = coords[mask].max(dim=0)[0]
            pe = self.pos_enc(coords[mask].float(), input_range=[mins, maxs])
            pe_list.append(pe)
        return torch.cat(pe_list)

    def find_nearest_point(self, source, query, source_batch, query_batch):
        """
        获取 points 在 scene 中的最近点, 返回 index (offset 对应的 batch 内的, 不是全局index)
        - source: (N_point, 3)
        - query: (N_query, 3)
        """
        assert (
            source_batch.max().item() == query_batch.max().item()
        ), "source_batch and query_batch must have the same shape"
        bs = source_batch.max().item() + 1
        dist = torch.cdist(source, query)
        matched_idx = torch.zeros(query.shape[0], dtype=torch.long)
        for i in range(bs):
            mask_s = source_batch == i
            mask_q = query_batch == i
            matched_idx[mask_q] = dist[mask_s, mask_q].argmin(dim=0)
        return matched_idx

    def construct_scene(self, pcd_dict):
        """
        构建 scene
        - scene: scene data dict
            - feature_pyramid: features collection layer
                - x_list(): return x_list, x in ['feat', 'coord', 'batch']
                - downsample(): downsample feat
                - features sorted from coarse to fine
            - feat_list: list, feat_list[j].shape == (N_point[j], features_dims[j])
            - coord_list: list, coord_list[j].shape == (N_point[j], 3)
            - batch_list: list, batch_list[j].shape == (N_point[j],)
            - pe_list: list, pe_list[j].shape == (N_point[j], embedding_dim)
        """
        scene = Scene(pcd_dict)
        scene.feature_pyramid = self.pcd_backbone.feature_pyramid
        scene.feat_list = scene.feature_pyramid.feat_list()
        scene.coord_list = scene.feature_pyramid.coord_list()
        scene.batch_list = scene.feature_pyramid.batch_list()
        # for i in range(len(scene.coord_list)):
        #     print("feat_list[i].shape:", scene.feat_list[i].shape)
        #     print("coord_list[i].shape:", scene.coord_list[i].shape)
        #     print("batch_list[i].shape:", scene.batch_list[i].shape)
        scene.pe_list = [
            self.get_batched_pe(scene.coord_list[i], scene.batch_list[i])
            for i in range(len(scene.coord_list))
        ]
        return scene

    def construct_query(self, scene: Scene):
        """
        构建 query
        - scene
        - query: query data dict
            - feat: (N_query, query_embedding_dim)
            - pe: positional encoding, (N_query, embedding_dim)
            - batch: (N_query,)
        """
        query = Query()
        bs = scene.batch_list[-1].max().item() + 1
        # 将 self.latent_query 复制 b 次
        query.feat = self.latent_query.weight.repeat(bs, 1).to(
            scene.feat_list[-1].device
        )
        query.pe = torch.zeros_like(query.feat)
        query.batch = torch.arange(bs, device=query.feat.device).repeat_interleave(
            self.num_query
        )
        # TODO gt_id_to_idx
        return query

    def add_new_query(self, new_query):
        """
        添加新的 query
        """
        pass

    def refine(self, pred_last, in_proj=True):
        """
        细化场景点云的预测结果
        - pred_last: 上一次的预测
            - 'masks_logits': (N_query_last, N_point[-1])
            - 'cls_logits': (N_query_last, num_classes)\n
        Returns:
        - pred_refine: 更新后的预测\n
        Tips:
        - 如何利用pred? 使用 float type attention mask
        """
        # mask_heatmap 在新增 query 时, 对新 query 的初始化 zero padding
        # TODO padding 的值, 考虑随次数调整, 或者移入 mask_decoder, 使用 head 的结果做padding
        last_masks_logits = pred_last["masks_logits"]
        one_padding = torch.ones(
            (
                self.query.feat.shape[0] - last_masks_logits.shape[0],
                last_masks_logits.shape[1],
            ),
            device=last_masks_logits.device,
            dtype=last_masks_logits.dtype,
        )
        pred_last["masks_logits"] = torch.cat(
            [last_masks_logits, one_padding], dim=0
        ).contiguous()

        masks_logits, cls_logits = self.mask_decoder(
            self.scene, self.query, pred_last, in_proj=in_proj
        )
        pred_refine = {"masks_logits": masks_logits, "cls_logits": cls_logits}
        return pred_refine

    def match_pred_target(self, pred, target):
        """构建与匹配的监督信号, 用于计算 loss;\n
        - pred, target is the result of get_pred() and get_target()
        - add matched_idx to pred and target, shape: (N_matched,)"""
        pred["matched_idx"], target["matched_idx"] = self.matcher(pred, target)
        return pred, target

    def post_process(self, pred):
        """
        后处理, 将 pred dict 转换为 互相排斥的 panoptic 结果, 同时做 soft grouping (arXiv:2203.01509)
        注意shape尚未被 get_pred() 改变
        - pred: dict, 预测结果
            - 'masks_logits': (N_query, N_point[-1])
            - 'cls_logits': (N_query, num_classes)
            - other attributes with (N_query, ...)

        Returns: pred dict
        - 'masks': (N_pred_instance, N_point[-1]), int tensor binary mask, for eval & test
        - 'masks_logits': (N_pred_instance, N_point[-1]), as logits, for loss func
        - 'cls_logits': (N_pred_instance, num_classes)
        - 'scores': (N_pred_instance,), 每个 instance 的得分
        - other attributes with (N_pred_instance, ...)
        """
        # 不能直接截断, 会出现某些point的heatmap截断为全0, argmax默认属于 query 0
        # 解决: 加一个 ignore filter mask
        masks_logits = pred["masks_logits"]
        masks = torch.cat(
            [
                torch.ones_like(masks_logits[:1, :]) * self.mask_threshold,
                masks_logits.sigmoid(),
            ],
            dim=0,
        ).contiguous()
        # TODO 把背景类也作为instance纳入训练, 测试时通过 semantic proto 做 query 来去除, 不关心的点为 -1
        pred_instance = (
            mask_to_id(masks) - 1
        )  # 预测的instance id, 互相排斥的 panoptic 结果
        pred_instance[pred_instance == -1] = self.instance_ignore
        unique_pred_id = unique_id(
            pred_instance, self.instance_ignore
        )  # filter 后, 有效的 query idx
        masks = id_to_mask(pred_instance, unique_pred_id)

        for key in pred.keys():
            pred[key] = pred[key][unique_pred_id]
        pred["masks"] = masks
        pred["masks_logits"] = (
            pred["masks_logits"] * masks
        )  # 相当于用 mask_threshold 阈值截断 masks_logits

        pred = softgroup_post_process(pred, self.topk_per_scene)
        return pred

    def forward(self, pcd_dict):
        """one time forward"""
        seg_logits = self.pcd_backbone(pcd_dict)  # (n, k)
        pcd_dict["pcd_pred"] = seg_logits
        self.scene = self.construct_scene(pcd_dict)
        self.query = self.construct_query(self.scene)

        pred_last = {
            "masks_logits": torch.ones(
                (0, pcd_dict["coord"].shape[0]),
                device=pcd_dict["coord"].device,
                dtype=pcd_dict["coord"].dtype,
            ),
            "cls_logits": torch.zeros(
                (0, self.num_classes),
                device=pcd_dict["coord"].device,
                dtype=pcd_dict["coord"].dtype,
            ),
        }
        pred_refine = self.refine(pred_last)  # 优化pred

        # 改变成合适测试的shape, 方便计算 metric
        pred = get_pred(pred_refine, self.query.batch)
        target = get_target(
            pcd_dict["segment"],
            pcd_dict["instance"],
            self.scene.batch_list[-1],
            self.instance_ignore,
        )

        # eval & test, do post process
        if not self.training:
            pred = self.post_process(pred)
        # test dataset, target is empty
        if target["is_testset"]:
            return Dict(pred=pred, target=target)
        # train & eval dataset, 返回的信息打印 log(engines/hooks/misc.py)
        pred, target = self.match_pred_target(pred, target)
        loss, info_dict = self.loss(pred, target)

        # 确保所有参数都参与梯度计算
        if self.training:
            # 添加一个小的正则化损失，确保所有参数都被使用
            reg_loss = 0
            for name, param in self.named_parameters():
                if param.requires_grad:
                    reg_loss = reg_loss + 0.0 * param.sum()
            loss = loss + reg_loss

            return Dict(loss=loss, **info_dict)
        else:
            # return pred target for evaluation
            return Dict(pred=pred, target=target, loss=loss, **info_dict)
