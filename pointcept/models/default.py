import torch.nn as nn
import os
import torch
from collections import OrderedDict

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from .builder import MODELS, build_model


@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        seg_logits = self.seg_head(point.feat)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_dict):
        feat = self.backbone(input_dict)
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)


@MODELS.register_module()
class DefaultSegmentorPretrained(nn.Module):
    def __init__(self, backbone=None, ckpt_path=None, requires_grad=False, criteria=None):
        '''
        默认的segmentor的预训练版本，返回没做softmax的结果
        - 直接由backbone预测seg的结果，训练模式则self.training设为True，其它时候返回backbone的输出和criteria的losses
        - 在__init__()中加载ckpt
        '''
        super().__init__()
        self.backbone = build_model(backbone)           # 需要build，最上层的model会在训练/测试时build
        self.criteria = build_criteria(criteria)
        # 加载ckpt的state_dict
        if os.path.isfile(ckpt_path):
            checkpoint = torch.load(ckpt_path)
        else:
            raise RuntimeError("=> No checkpoint found at '{}'".format(ckpt_path))
        state_dict = checkpoint['state_dict']
        new_state_dict = OrderedDict()
        for name, value in state_dict.items():
            if name.startswith("module."):
                name = name[7:]  # module.xxx.xxx -> xxx.xxx
            new_state_dict[name] = value
        self.load_state_dict(new_state_dict)
        # 冻住所有参数
        if not requires_grad:
            for p in self.parameters():
                p.requires_grad = False
    
    def get_embedding(self):
        '''编码, 更新input_dict['feat']成网络编码后的point-wise的features, 在forward一次后调用, 会返回forward对应的scene的embedding'''
        return self.backbone.embedding

    def forward(self, input_dict):
        seg_logits = self.backbone(input_dict)
        return dict(seg_logits=seg_logits, feature_maps=[self.get_embedding()])
    

@MODELS.register_module()
class DefaultSegmentorFPN(nn.Module):
    def __init__(self, backbone=None, ckpt_path=None, requires_grad=True, criteria=None):
        '''
        默认的segmentor, 仅返回 feature maps
        '''
        super().__init__()
        self.backbone = build_model(backbone)           # 需要build，最上层的model会在训练/测试时build
        self.criteria = build_criteria(criteria)

        if ckpt_path != None:
            if os.path.isfile(ckpt_path):
                checkpoint = torch.load(ckpt_path)
                state_dict = checkpoint['state_dict']
                new_state_dict = OrderedDict()
                for name, value in state_dict.items():
                    if name.startswith("module."):
                        name = name[7:]  # module.xxx.xxx -> xxx.xxx
                    new_state_dict[name] = value
                self.load_state_dict(new_state_dict, strict=False)
            else:
                print(f"No ckpt found at {ckpt_path}")
        
        # 冻住所有参数
        if not requires_grad:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, input_dict):
        seg_logits, feature_maps = self.backbone(input_dict)
        return dict(seg_logits=seg_logits, feature_maps=feature_maps)


@MODELS.register_module()
class DefaultInteractiveSegmentor(nn.Module):
    def __init__(self, pcd_backbone, pcd_only=True, img_backbone=None, mask_decoder=None, criteria=None):
        '''
        默认的interactive的segmentor，返回没做softmax的结果
        - 直接由backbone预测seg的结果，训练模式则self.training设为True，其它时候返回backbone的输出和criteria的losses
        - pcd_backbone和img_backbone默认直接在build_model构建类的时候，__init__里加载预训练参数
        - pcd_backbone和img_backbone需要有encode方法，做点云和图像的encode
        '''
        super().__init__()
        self.pcd_backbone = build_model(pcd_backbone)
        self.img_backbone = build_model(img_backbone)
        self.mask_decoder = build_model(mask_decoder)
        self.criteria = build_criteria(criteria)
        
    # def load_state_dict(self, state_dict, strict=True):
    #     '''重写父类方法，做重命名'''
    #     new_state_dict = OrderedDict()
    #     for name, value in state_dict.items():
    #         if name.startswith("backbone."):
    #             name = "pcd_backbone." + name  # backbone.xxx -> pcd_backbone.backbone.xxx
    #         new_state_dict[name] = value
    #     return super().load_state_dict(new_state_dict, strict)

    def forward_pcd(self, pcd_dict):
        '''仅点云推理'''
        return self.pcd_backbone(pcd_dict)
    
    def forward_img(self, img_dict):
        '''仅图片推理'''
        return self.img_backbone(img_dict)

    def refine(self, pred, pcd_dict, clicks, img_dict=None, fusion=None):
        '''
        场景点云的预测结果, 用该场景的RGBD去refine
        - pred: 纯pcd的预测结果
        - pcd_dict: fragment_list 中的一个, 做了voxelize, ['coord', 'discrete_coord', 'index', 'offset', 'feat']
        - fusion: ['sampled_points', 'sampled_colors', 'sampled_index', 'image_projected'], 是直接对未voxelize的point做fusion
        '''
        raise NotImplementedError
        img_dict = self.img_backbone(img_dict)                  # 附上2d的预测结果
        # 默认给的是cls_agonistic_seg，用来做refinement
        for i in range(len(img_dict['img_list'])):
            cls_agonistic_seg = img_dict["cls_agonistic_seg"][i]
        img_embedding = self.img_backbone.encode(img_dict)      # img embedding
        scene_embedding = self.pcd_backbone.encode(pcd_dict)    # pcd embedding
        index_part = pcd_dict["index"]      # voxelize产生的，voxel在原数据的point的index
    
    def forward(self, input_dict):
        raise NotImplementedError
        pcd_dict = input_dict["pcd_dict"]
        img_dict = input_dict["img_dict"]
        fusion = input_dict["fusion"]
        clicks = input_dict["clicks"]
        pcd_pred = self.forward_pcd(input_dict)["seg_logits"]  # (n, k)
        pred_refine_part = self.refine(pcd_pred, pcd_dict, clicks, 
                                       img_dict=img_dict, fusion=fusion)    # 优化pred
        seg_logits = pred_refine_part
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)