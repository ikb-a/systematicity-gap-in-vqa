"""
 coding=utf-8
 Copyright 2018, Antonio Mendoza Hao Tan, Mohit Bansal
 Adapted From Facebook Inc, Detectron2
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
     http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.import copy
 """
import sys
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn
from typing import List, Union, Tuple

from vislang.vislang.models.lxmert.utils import img_tensorize, Config


class ResizeShortestEdge:
    def __init__(self, short_edge_length: List[int], max_size: int = sys.maxsize):
        """
        Args:
            short_edge_length (list[min, max])
            max_size (int): maximum allowed longest edge length.
        """
        self.interp_method = "bilinear"
        self.max_size = max_size
        self.short_edge_length = short_edge_length

    def __call__(self, imgs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Given a list of image tensors, randomly choose an integer between
        self.sort_edge_length[0] and self.sort_edge_length[1] (inclusive), and
        scale the shorter edge of the image up to this size. Next, if
        the long side exceeds self.max_size, then scale the image down until
        the longer side is exactly self.max_size.

        NOTE: If the img datatype is NOT np.uint8 then the dimensions of each
        image should be HxWxC, and each output image created is CxHxW.
        If the img datatype is np.uint8 (which is untested), then I think we
        end up with HxWxC instead? Weird behaviour.

        :param imgs: List of images, each a (HxWxC) tensor.
        :return:
        """
        img_augs = []
        for img in imgs:
            h, w = img.shape[:2]
            # later: provide list and randomly choose index for resize
            size = np.random.randint(
                self.short_edge_length[0], self.short_edge_length[1] + 1
            )
            if size == 0:
                return img
            scale = size * 1.0 / min(h, w)
            if h < w:
                newh, neww = size, scale * w
            else:
                newh, neww = scale * h, size
            if max(newh, neww) > self.max_size:
                scale = self.max_size * 1.0 / max(newh, neww)
                newh = newh * scale
                neww = neww * scale
            neww = int(neww + 0.5)
            newh = int(newh + 0.5)

            if img.dtype == np.uint8:
                raise Exception(
                    "I could be wrong, but I think this block is "
                    "wrong (I believe) it will create HxWxC tensors "
                    "as opposed to the 1xCxHxW tensors that the other"
                    "case creates -- please check before proceeding!"
                )
                pil_image = Image.fromarray(img)
                pil_image = pil_image.resize((neww, newh), Image.BILINEAR)
                img = np.asarray(pil_image)
            else:
                img = img.permute(2, 0, 1).unsqueeze(0)  # 3, 0, 1)  # hw(c) -> nchw
                img = nn.functional.interpolate(
                    img, (newh, neww), mode=self.interp_method, align_corners=False
                ).squeeze(0)
            img_augs.append(img)

        return img_augs


class Preprocess:
    def __init__(self, cfg: Config):
        """
        cfg.INPUT.MAX_SIZE_TEST: Max height/width length that the image can be
        scaled up to (int)
        cfg.INPUT.MIN_SIZE_TEST: Shortest side of the image is scaled up to
        this size, unless doing so violates MAX_SIZE_TEST, in which case we
        scale as much as possible (int)
        cfg.MODEL.PIXEL_MEAN: list of length 3; mean for each colour channel
        cfg.MODEL.PIXEL_STD: list of length 3; std deviation for each colour
        channel (set to all ones to disable normalization by std deviation)
        cfg.INPUT.FORMAT: Colour channel order will be BGR, UNLESS! cfg.INPUT.FORMAT == "RGB" *exactly* (in that case, its RGB).

        :param cfg:
        """
        self.aug = ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        self.input_format = cfg.INPUT.FORMAT
        self.size_divisibility = cfg.SIZE_DIVISIBILITY
        self.pad_value = cfg.PAD_VALUE
        self.max_image_size = cfg.INPUT.MAX_SIZE_TEST
        self.device = cfg.MODEL.DEVICE
        self.pixel_std = (
            torch.tensor(cfg.MODEL.PIXEL_STD)
            .to(self.device)
            .view(len(cfg.MODEL.PIXEL_STD), 1, 1)
        )
        self.pixel_mean = (
            torch.tensor(cfg.MODEL.PIXEL_MEAN)
            .to(self.device)
            .view(len(cfg.MODEL.PIXEL_STD), 1, 1)
        )
        self.normalizer = lambda x: (x - self.pixel_mean) / self.pixel_std

    def pad(self, images: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param images: List of CxHxW image tensors
        :returns: All images right/bottom padded with self.pad_value to the smallest possible shared size.
        """
        max_size = tuple(
            max(s) for s in zip(*[img.shape for img in images])
        )  # get max dimension length for each dim over all images.
        image_sizes = [im.shape[-2:] for im in images]
        images = [
            nn.functional.pad(
                im,
                [0, max_size[-1] - size[1], 0, max_size[-2] - size[0]],
                value=self.pad_value,
            )
            for size, im in zip(image_sizes, images)
        ]

        return torch.stack(images), torch.tensor(image_sizes)

    def __call__(
        self, images: List[Union[str, torch.Tensor]], single_image: bool = False
    ):
        """
        Performs:
        0) Convert any images that are URLs or paths into torch.Tensor, keep any
           torch.Tensor unchanged. Any provided image tensors should be in
           HxWxC format and have the colour
           channels in the same order as cfg.INPUT.FORMAT  (i.e., BGR, unless
           cfg.INPUT.FORMAT == "RGB" *exactly*)
        1) Rescale images (longest side at most cfg.INPUT.MAX_SIZE_TEST, and
           shortest side as close to cfg.INPUT.MIN_SIZE_TEST as possible under
           the prior constraint)
        2) Channel-wise z-normalize images; specifically, subtract cfg.MODEL.PIXEL_MEAN
        and divide by cfg.MODEL.PIXEL_STD (both these should be lists of 3 elements, in the
        same channel order specified by cfg.INPUT.FORMAT)
        3) Right/bottom pad all images with cfg.PAD_VALUE to the same size (i.e., max length in the batch for each dimension)
        4) Return the processed images (NxCxHxW tensor), their scaled
           Heights & Widths *without* the padding (Nx2 tensor), and the scaling
           factor on each dimension (Nx2 tensor).


        :param images: List of CxHxW tensors, or URLs/paths to images
        :param single_image: If true, just return the values directly instead of in lists of length 1.
        :return:
        """
        with torch.no_grad():
            if not isinstance(images, list):
                images = [images]
            if single_image:
                assert len(images) == 1
            for i in range(len(images)):
                if isinstance(images[i], torch.Tensor):
                    images.insert(i, images.pop(i).to(self.device).float())
                elif not isinstance(images[i], torch.Tensor):
                    images.insert(
                        i,
                        torch.as_tensor(
                            img_tensorize(images.pop(i), input_format=self.input_format)
                        )
                        .to(self.device)
                        .float(),
                    )
            # resize smallest edge
            raw_sizes = torch.tensor([im.shape[:2] for im in images]).to(
                self.device
            )  # get list of (height, width) of images
            images = self.aug(images)
            # transpose images and convert to torch tensors
            # images = [torch.as_tensor(i.astype("float32")).permute(2, 0, 1).to(self.device) for i in images]
            # now normalize before pad to avoid useless arithmetic
            images = [self.normalizer(x) for x in images]
            # now pad them to do the following operations
            images, sizes = self.pad(images)
            sizes = sizes.to(self.device)
            # Normalize

            if self.size_divisibility > 0:
                raise NotImplementedError()
            # pad
            scales_yx = torch.true_divide(raw_sizes, sizes)
            if single_image:
                return images[0], sizes[0], scales_yx[0]
            else:
                return images, sizes, scales_yx


def _scale_box(boxes, scale_yx):
    boxes[:, 0::2] *= scale_yx[:, 1]
    boxes[:, 1::2] *= scale_yx[:, 0]
    return boxes


def _clip_box(tensor, box_size: Tuple[int, int]):
    assert torch.isfinite(tensor).all(), "Box tensor contains infinite or NaN!"
    h, w = box_size
    tensor[:, 0].clamp_(min=0, max=w)
    tensor[:, 1].clamp_(min=0, max=h)
    tensor[:, 2].clamp_(min=0, max=w)
    tensor[:, 3].clamp_(min=0, max=h)
