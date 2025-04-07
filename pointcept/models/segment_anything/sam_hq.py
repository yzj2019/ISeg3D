import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

from ..builder import MODELS

from pointcept.models.losses import build_criteria

from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from .predictor import MySamPredictor
from .utils import ResizeLongestSide


@MODELS.register_module("SAM-HQ")
class SegmentAnythingHQ(nn.Module):
    """
    segment anything in high quality
    - 模型自行load ckpt，并且冻住所有参数
    """

    def __init__(self, model_type="vit_h", ckpt_path=None, criteria=None):
        super().__init__()
        assert ckpt_path != None
        # device = "cuda:3"
        self.sam = sam_model_registry[model_type](checkpoint=ckpt_path).cuda()
        # self.sam = sam_model_registry[model_type](checkpoint=ckpt_path).to(device)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        self.predictor = MySamPredictor(self.sam)
        self.criteria = build_criteria(criteria)
        self.transform = ResizeLongestSide(self.sam.image_encoder.img_size)
        # 冻住所有参数
        for p in self.parameters():
            p.requires_grad = False

    @staticmethod
    def get_sam(image, mask_generator):
        """
        获取所有class无关的mask, 构成一张 label map
        """
        masks = mask_generator.generate(image)
        group_ids = np.full((image.shape[0], image.shape[1]), -1, dtype=int)
        num_masks = len(masks)
        group_counter = 0
        for i in reversed(range(num_masks)):
            # print(masks[i]["predicted_iou"])
            group_ids[masks[i]["segmentation"]] = group_counter
            group_counter += 1
        return group_ids

    def encode(self, image: np.ndarray, image_format: str = "RGB"):
        """输入一个 RGB image (H,W,C), 返回用SAM encode 的编码
        - SAM这里的feature是per-grid的

        input:
        - image (np.ndarray): The image for calculating masks. Expects an
            image in HWC uint8 format, with pixel values in [0, 255].
        - image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.sam.image_format:
            image = image[..., ::-1]
        # Transform the image to the form expected by the model
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=self.sam.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[
            None, :, :, :
        ]
        # 此时已经转换成了 1,3,H_new,W_new 形状的
        self.original_size = image.shape[:2]
        return self.__encode_torch_image(input_image_torch)

    def __encode_torch_image(self, transformed_image: torch.Tensor):
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method. Expects the input
        image to be already transformed to the format expected by the model.

        Arguments:
        - transformed_image (torch.Tensor): The input image, with shape
            1x3xHxW, which has been transformed with ResizeLongestSide.
        """
        assert (
            len(transformed_image.shape) == 4
            and transformed_image.shape[1] == 3
            and max(*transformed_image.shape[2:]) == self.sam.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {self.sam.image_encoder.img_size}."

        input_image = self.sam.preprocess(transformed_image)
        features, _ = self.sam.image_encoder(input_image)
        # TODO 将decoder里做的upscaling写进来, 方便按pixel选embedding
        return features

    def transform_coords(self, coords: np.ndarray):
        """将(N,2)形状的pixel坐标, 转化为reshape后的坐标, 对应feature map的位置"""
        return self.transform.apply_coords(coords, self.original_size)

    def forward(self, input_dict):
        """输入一整个scene的image_dict, 输出附带上预测的结果"""
        input_dict["cls_agonistic_seg"] = []
        for i in range(len(input_dict["img_list"])):
            input_dict["cls_agonistic_seg"].append(
                self.get_sam(input_dict["img_list"][i])
            )
        return input_dict


# @MODELS.register_module("SAM-HQ-")
# class SegmentAnythingHQ-(nn.Module):
#     '''
#     segment anything in high quality
#     - 模型自行load ckpt，并且冻住所有参数
#     '''
