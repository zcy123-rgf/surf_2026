import torch.nn as nn


class ConvModule(nn.Module):
    """Minimal subset of mmcv.cnn.ConvModule used by CLRNet."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=dict(type="ReLU"),
        inplace=True,
        **kwargs,
    ):
        super().__init__()
        if conv_cfg is not None:
            raise NotImplementedError("The local ConvModule shim only supports Conv2d.")

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias if norm_cfg is None else False,
        )

        if self.with_norm:
            norm_type = norm_cfg.get("type", "BN")
            if norm_type not in ("BN", "BatchNorm2d"):
                raise NotImplementedError(f"Unsupported norm type: {norm_type}")
            self.bn = nn.BatchNorm2d(out_channels)

        if self.with_activation:
            act_type = act_cfg.get("type", "ReLU") if isinstance(act_cfg, dict) else "ReLU"
            if act_type != "ReLU":
                raise NotImplementedError(f"Unsupported activation type: {act_type}")
            self.activate = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.with_norm:
            x = self.bn(x)
        if self.with_activation:
            x = self.activate(x)
        return x

