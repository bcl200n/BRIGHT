"""Mask R-CNN model with 4-channel fusion input (PaddlePaddle version).

Uses paddle.vision.models.detection.maskrcnn_resnet50_fpn as base,
then replaces conv1 for 4-channel (RGB+SAR) input and the box/mask
predictors for the custom number of classes.
"""

import numpy as np
import paddle
import paddle.nn as nn
from paddle.vision.models.detection import MaskRCNN


def build_model(
    num_classes: int = 4,
    pretrained: bool = True,
    pixel_mean: list = None,
    pixel_std: list = None,
    box_detections_per_img: int = 1500,
    rpn_pre_nms_top_n_test: int = 1500,
    rpn_post_nms_top_n_test: int = 1500,
) -> nn.Layer:
    """Build a Mask R-CNN model with 4-channel input for RGB+SAR fusion.

    The first three channels correspond to pre-event RGB and the fourth to
    post-event SAR.  Conv1 is replaced with a 4-channel variant: the first
    three channels copy pretrained ImageNet weights and the fourth is
    initialised with the mean of the RGB kernel weights.
    """
    image_mean = list(pixel_mean) if pixel_mean is not None else [0.0] * 4
    image_std = list(pixel_std) if pixel_std is not None else [1.0] * 4

    backbone = paddle.vision.models.resnet50(pretrained=pretrained)

    model = MaskRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
        rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
        box_detections_per_img=box_detections_per_img,
    )

    old_conv1 = model.backbone.conv1
    old_weight = old_conv1.weight.numpy()

    new_conv1 = nn.Conv2D(
        in_channels=4,
        out_channels=old_conv1._out_channels,
        kernel_size=old_conv1._kernel_size,
        stride=old_conv1._stride,
        padding=old_conv1._padding,
        dilation=old_conv1._dilation,
        groups=old_conv1._groups,
        bias_attr=False if old_conv1.bias is None else True,
    )

    new_weight = np.zeros(
        [old_conv1._out_channels, 4, old_conv1._kernel_size[0], old_conv1._kernel_size[1]],
        dtype=np.float32,
    )
    new_weight[:, :3, :, :] = old_weight
    new_weight[:, 3:, :, :] = old_weight.mean(axis=1, keepdims=True)
    new_conv1.weight.set_value(new_weight)

    if old_conv1.bias is not None:
        new_conv1.bias.set_value(old_conv1.bias.numpy())

    model.backbone.conv1 = new_conv1

    model.image_mean = image_mean
    model.image_std = image_std

    return model
