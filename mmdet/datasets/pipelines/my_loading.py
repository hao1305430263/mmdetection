# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
import pycocotools.mask as maskUtils

from mmdet.core import BitmapMasks, PolygonMasks
from ..builder import PIPELINES

try:
    from panopticapi.utils import rgb2id
except ImportError:
    rgb2id = None


@PIPELINES.register_module()
class my_LoadImageFromFile:
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(
        self,
        to_float32=False,
        color_type="color",
        channel_order="bgr",
        file_client_args=dict(backend="disk"),
    ):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results["img_prefix"] is not None:
            filename0 = osp.join(
                results["img_prefix"], results["img_info"][0]["filename"]
            )
            filename1 = osp.join(
                results["img_prefix"], results["img_info"][1]["filename"]
            )
        else:
            filename0 = results["img_info"][0]["filename"]
            filename1 = results["img_info"][1]["filename"]

        img_bytes1 = self.file_client.get(filename0)
        img_bytes2 = self.file_client.get(filename1)
        img1 = mmcv.imfrombytes(
            img_bytes1, flag=self.color_type, channel_order=self.channel_order
        )
        img2 = mmcv.imfrombytes(
            img_bytes2, flag=self.color_type, channel_order=self.channel_order
        )
        if self.to_float32:
            img1 = img1.astype(np.float32)
        if self.to_float32:
            img2 = img1.astype(np.float32)
        img = np.concatenate((img1, img2), axis=2)

        results["img_info"] = results["img_info"][0]
        results["filename"] = filename0
        results["ori_filename"] = results["img_info"]["filename"]
        results["img"] = img
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        results["img_fields"] = ["img"]
        return results

    def __repr__(self):
        repr_str = (
            f"{self.__class__.__name__}("
            f"to_float32={self.to_float32}, "
            f"color_type='{self.color_type}', "
            f"channel_order='{self.channel_order}', "
            f"file_client_args={self.file_client_args})"
        )
        return repr_str
