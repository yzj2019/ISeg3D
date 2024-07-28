import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List
import cv2

from ..builder import MODELS

from pointcept.models.losses import build_criteria

from segment_anything import sam_model_registry_baseline, SamPredictor, SamAutomaticMaskGenerator
from .predictor import MySamPredictor
from .utils import ResizeLongestSide


@MODELS.register_module("SAM")
class SegmentAnything(nn.Module):
    '''
    segment anything
    - 模型自行load ckpt，并且冻住所有参数
    '''
    def __init__(self, model_type="vit_h", ckpt_path=None, criteria=None):
        super().__init__()
        assert ckpt_path != None
        # device = "cuda:3"
        self.sam = sam_model_registry_baseline[model_type](checkpoint=ckpt_path).cuda()
        # self.sam = sam_model_registry_baseline[model_type](checkpoint=ckpt_path).to(device)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        self.predictor = MySamPredictor(self.sam)
        self.criteria = build_criteria(criteria)
        self.transform = ResizeLongestSide(self.sam.image_encoder.img_size)
        # 冻住所有参数
        for p in self.parameters():
            p.requires_grad = False

    @staticmethod
    def get_sam(image, mask_generator):
        '''
        获取所有class无关的mask, 构成一张 label map
        '''
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
        '''输入一个 RGB image (H,W,C), 返回用SAM encode 的编码
        - SAM这里的feature是per-grid的
        
        input:
        - image (np.ndarray): The image for calculating masks. Expects an
            image in HWC uint8 format, with pixel values in [0, 255].
        - image_format (str): The color format of the image, in ['RGB', 'BGR'].
        '''
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.sam.image_format:
            image = image[..., ::-1]
        # Transform the image to the form expected by the model
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=self.sam.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        # 此时已经转换成了 1,3,H_new,W_new 形状的
        self.original_size = image.shape[:2]
        return self.__encode_torch_image(input_image_torch)
    
    def __encode_torch_image(self, transformed_image: torch.Tensor):
        """
        Calculates the image embeddings for the provided image. Expects the input
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
        return features


    def encode_prompt(self, 
                      image: np.ndarray, 
                      image_format: str = "RGB",
                      point_coords: Optional[np.ndarray] = None,
                      point_labels: Optional[np.ndarray] = None,
                      box: Optional[np.ndarray] = None,
                      mask_input: Optional[np.ndarray] = None,
                      multimask_output: bool = True
                      ):
        '''
        输入一个 RGB image (H,W,C), 以及对应的prompt, 返回用SAM encode+decode的prompt编码
        
        input:
        - image (np.ndarray): The image for calculating masks. Expects an
            image in HWC uint8 format, with pixel values in [0, 255].
        - image_format (str): The color format of the image, in ['RGB', 'BGR'].
        - point_coords (np.ndarray or None): A Nx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
        - point_labels (np.ndarray or None): A length N array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
        - box (np.ndarray or None): A length 4 array given a box prompt to the
            model, in XYXY format.
        - mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form 1xHxW, where
            for SAM, H=W=256.
        - multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
        '''
        # 1. image预处理
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.sam.image_format:
            image = image[..., ::-1]
        # Transform the image to the form expected by the model
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=self.sam.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        # 此时已经转换成了 1,3,H_new,W_new 形状的
        self.original_size = image.shape[:2]

        # 2. prompt预处理
        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = self.transform.apply_coords(point_coords, self.original_size)        # resize
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.sam.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.sam.device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        if box is not None:
            box = self.transform.apply_boxes(box, self.original_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.sam.device)
            box_torch = box_torch[None, :]
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.sam.device)
            mask_input_torch = mask_input_torch[None, :, :, :]

        # 3. encode
        mask_embedding, prompt_embedding = self.__encode_torch_prompt(
            input_image_torch,
            coords_torch,
            labels_torch,
            box_torch,
            mask_input_torch,
            multimask_output
        )
        
        return prompt_embedding


    def __encode_torch_prompt(self,
                              transformed_image: torch.Tensor,
                              point_coords: Optional[torch.Tensor],
                              point_labels: Optional[torch.Tensor],
                              boxes: Optional[torch.Tensor] = None,
                              mask_input: Optional[torch.Tensor] = None,
                              multimask_output: bool = True):
        """
        输入一个 RGB image (H,W,C), 以及对应的prompt, 返回用SAM encode+decode的prompt编码

        Arguments:
        - transformed_image (torch.Tensor): The input image, with shape
            1x3xHxW, which has been transformed with ResizeLongestSide.
        """
        assert (
            len(transformed_image.shape) == 4
            and transformed_image.shape[1] == 3
            and max(*transformed_image.shape[2:]) == self.sam.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {self.sam.image_encoder.img_size}."
        # 1. 图像编码
        input_image = self.sam.preprocess(transformed_image)
        features, _ = self.sam.image_encoder(input_image)

        # 将sam decoder里做的upscaling写进来, 方便按pixel选embedding
        # 2. 参数, 参考 sam.py 的 forward 流程
        # image_embeddings = features.unsqueeze(0)
        image_embeddings = features             # batched input 才需要再加一层
        image_pe = self.sam.prompt_encoder.get_dense_pe()
        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None
        # sparse: (1, 3, 256); dense: (1, 256, 64, 64)
        sparse_prompt_embeddings, dense_prompt_embeddings = self.sam.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )

        # 3. decode 的流程, 参考 sam 的 mask_decoder.py
        # 构造 output_tokens 并与 prompt embeddings 拼接
        output_tokens = torch.cat([self.sam.mask_decoder.iou_token.weight, self.sam.mask_decoder.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape          # (1, 256, 64, 64)

        # Run the transformer
        hs, src = self.sam.mask_decoder.transformer(src, pos_src, tokens)   # (1, 8, 256), (1, 4096, 256)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.sam.mask_decoder.num_mask_tokens), :]     # (1, 4, 256)

        # Upscale mask embeddings, 做 ConvTranspose
        src = src.transpose(1, 2).view(b, c, h, w)      # (1, 256, 64, 64)
        upscaled_embedding = self.sam.mask_decoder.output_upscaling(src)    # (1, 32, 256, 256)
        # 经过每类不同尺度的mask各自的三层MLP, 维度256/8=32了
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.sam.mask_decoder.num_mask_tokens):
            hyper_in_list.append(self.sam.mask_decoder.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)

        # 4. Select the correct prompt embedding
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        # 如果multimask, 则dim1为3; 否则dim1为1
        mask_embedding = upscaled_embedding[:, mask_slice, :, :]
        prompt_embedding = hyper_in[:, mask_slice]                  # (1, 3, 32)

        return mask_embedding, prompt_embedding
    

    def transform_coords(self, coords: np.ndarray):
        '''将(N,2)形状的pixel坐标, 转化为reshape后的坐标, 对应feature map的位置'''
        return self.transform.apply_coords(coords, self.original_size)

    def forward(self, input_dict):
        '''输入一整个scene的image_dict, 输出附带上预测的结果'''
        input_dict["cls_agonistic_seg"] = []
        for i in range(len(input_dict["img_list"])):
            input_dict["cls_agonistic_seg"].append(self.get_sam(input_dict["img_list"][i]))
        return input_dict


