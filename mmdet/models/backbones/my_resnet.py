import torch
import torch.nn as nn

from ..builder import BACKBONES
from .resnet import ResNet


@BACKBONES.register_module()
class my_ResNet(nn.Module):
    def __init__(
        self,
        depth,
        in_channels=3,
        stem_channels=None,
        base_channels=64,
        num_stages=4,
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        out_indices=(0, 1, 2, 3),
        style="pytorch",
        deep_stem=False,
        avg_down=False,
        frozen_stages=-1,
        conv_cfg=None,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=True,
        dcn=None,
        stage_with_dcn=(False, False, False, False),
        plugins=None,
        with_cp=False,
        zero_init_residual=True,
        pretrained=None,
        init_cfg0=None,
        init_cfg1=None,
    ):
        super(my_ResNet, self).__init__()

        self.Res1 = ResNet(
            depth=depth,
            in_channels=in_channels,
            stem_channels=stem_channels,
            base_channels=base_channels,
            num_stages=num_stages,
            strides=strides,
            dilations=dilations,
            out_indices=out_indices,
            style=style,
            deep_stem=deep_stem,
            avg_down=avg_down,
            frozen_stages=frozen_stages,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            norm_eval=norm_eval,
            dcn=dcn,
            stage_with_dcn=stage_with_dcn,
            plugins=plugins,
            with_cp=with_cp,
            zero_init_residual=zero_init_residual,
            pretrained=pretrained,
            init_cfg=init_cfg0,
        )
        self.Res2 = ResNet(
            depth=depth,
            in_channels=in_channels,
            stem_channels=stem_channels,
            base_channels=base_channels,
            num_stages=num_stages,
            strides=strides,
            dilations=dilations,
            out_indices=out_indices,
            style=style,
            deep_stem=deep_stem,
            avg_down=avg_down,
            frozen_stages=frozen_stages,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            norm_eval=norm_eval,
            dcn=dcn,
            stage_with_dcn=stage_with_dcn,
            plugins=plugins,
            with_cp=with_cp,
            zero_init_residual=zero_init_residual,
            pretrained=pretrained,
            init_cfg=init_cfg1,
        )

    def forward(self, x):  # should return a tuple
        x1 = x[:, :3, :, :]
        x2 = x[:, 3:, :, :]
        y1 = self.Res1(x1)
        y2 = self.Res2(x2)
        out = []
        for y1_, y2_ in zip(y1, y2):
            temp = torch.cat((y1_, y2_), 1)
            out.append(temp)
        return out
