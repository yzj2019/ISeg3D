# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from segment_anything import SamPredictor

from typing import Optional, Tuple, List


class MySamPredictor(SamPredictor):
    """重写SamPredictor, 增加获取decode后的prompt token的方法"""

    def set_image_list(
        self,
        image_list: List[np.ndarray],
        image_format: str = "RGB",
    ) -> None:
        """
        Calculates the image embeddings for the provided image list, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
        - image_list (List[np.ndarray]): The image for calculating masks. Expects an
            image in HWC uint8 format, with pixel values in [0, 255].
        - image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        images = np.array(image_list)
        # import pdb;pdb.set_trace()
        if image_format != self.model.image_format:
            images = images[..., ::-1]

        # Transform the image to the form expected by the model
        # import pdb;pdb.set_trace()
        input_images = []
        for i in range(images.shape[0]):
            input_images.append(self.transform.apply_image(images[i]))
        input_images_torch = torch.as_tensor(np.array(input_images), device=self.device)
        input_images_torch = input_images_torch.permute(0, 3, 1, 2).contiguous()

        self.set_torch_image(input_images_torch, images.shape[1:3])

    def get_token(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        hq_token_only: bool = False,
    ):
        """
        获取prompt对应的token embedding

        """
        # Transform input prompts
        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = self.transform.apply_coords(point_coords, self.original_size)
            coords_torch = torch.as_tensor(
                point_coords, dtype=torch.float, device=self.device
            )
            labels_torch = torch.as_tensor(
                point_labels, dtype=torch.int, device=self.device
            )
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        if box is not None:
            box = self.transform.apply_boxes(box, self.original_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
            box_torch = box_torch[None, :]
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(
                mask_input, dtype=torch.float, device=self.device
            )
            mask_input_torch = mask_input_torch[None, :, :, :]
