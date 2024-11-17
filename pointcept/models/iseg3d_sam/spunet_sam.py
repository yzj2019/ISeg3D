from collections import OrderedDict
from torch import Tensor
import torch.nn as nn
import os
import torch

from pointcept.models.losses import build_criteria
from ..builder import MODELS, build_model
from pointcept.datasets.scannet_fusion import ScanNetImageDataset


@MODELS.register_module()
class InterSeg_spunet_sam(nn.Module):
    def __init__(self, pcd_backbone=None, img_backbone=None, criteria=None):
        '''
        默认的interactive的segmentor，返回没做softmax的结果
        - 直接由backbone预测seg的结果，训练模式则self.training设为True，其它时候返回backbone的输出和criteria的losses
        - pcd_backbone和img_backbone默认直接在build_model构建类的时候，__init__里加载预训练参数
        - pcd_backbone: DefaultSegmentorPretrained
        - img_backbone: SegmentAnything
        '''
        super().__init__()
        self.pcd_backbone = build_model(pcd_backbone)
        self.img_backbone = build_model(img_backbone)
        self.criteria = build_criteria(criteria)

    def forward_pcd(self, pcd_dict):
        '''仅点云推理'''
        return self.pcd_backbone(pcd_dict)
    
    def forward_img(self, img_dict):
        '''仅图片推理'''
        return self.img_backbone(img_dict)
    
    def refine(self, pred, img_dict, fusion):
        '''
        场景点云的预测结果，用该场景的RGBD去refine
        - fusion: ['sampled_points', 'sampled_colors', 'sampled_index', 'image_projected']
        '''
        img_dict = self.img_backbone(img_dict)
        # 默认给的是cls_agonistic_seg，用来做refinement
        for i in range(len(img_dict['img_list'])):
            cls_agonistic_seg = img_dict["cls_agonistic_seg"][i]
        return pred



@MODELS.register_module()
class InterSeg_spunet_samhq(nn.Module):
    def __init__(self, pcd_backbone=None, img_backbone=None, criteria=None):
        '''
        默认的interactive的segmentor，返回没做softmax的结果
        - 直接由backbone预测seg的结果，训练模式则self.training设为True，其它时候返回backbone的输出和criteria的losses
        - pcd_backbone和img_backbone默认直接在build_model构建类的时候，__init__里加载预训练参数
        - pcd_backbone: DefaultSegmentorPretrained
        - img_backbone: SegmentAnythingHQ
        '''
        super().__init__()
        self.pcd_backbone = build_model(pcd_backbone)
        self.img_backbone = build_model(img_backbone)
        self.criteria = build_criteria(criteria)

    def forward_pcd(self, pcd_dict):
        '''仅点云推理'''
        return self.pcd_backbone(pcd_dict)
    
    def forward_img(self, img_dict):
        '''仅图片推理'''
        return self.img_backbone(img_dict)
    
    def refine(self, pred, img_dict, fusion):
        '''
        场景点云的预测结果，用该场景的RGBD去refine
        - fusion: ['sampled_points', 'sampled_colors', 'sampled_index', 'image_projected']
        '''
        img_dict = self.img_backbone(img_dict)
        # 默认给的是cls_agonistic_seg，用来做refinement
        for i in range(len(img_dict['img_list'])):
            cls_agonistic_seg = img_dict["cls_agonistic_seg"][i]
        return pred