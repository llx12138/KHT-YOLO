# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Convolution modules."""

import math

import numpy as np
import torch
import torch.nn as nn

__all__ = (
    "LocalContextShare",
    "GCSCA_Light",
    "GCSCA",
    "Conv",
    "Conv2",
    "LightConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "RepConv",
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes Ghost Convolution module with primary and cheap operations for efficient feature learning."""
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)



# 单尺度卷积：将空间注意力的多尺度卷积简化为3x3卷积，降低计算复杂度。
# 恒等残差连接：用恒等映射保留输入信息，减少额外计算。
# 平均池化：在全局上下文中采用较大核的平均池化，进一步减小计算量。
# class GCSCA(nn.Module):
#     def __init__(self, in_channels, reduction=16):
#         super(GCSCA, self).__init__()
#
#         # 通道注意力模块
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
#         self.sigmoid_c = nn.Sigmoid()
#
#         # 空间注意力模块
#         self.conv_spatial = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
#         self.sigmoid_s = nn.Sigmoid()
#
#         # 全局上下文模块
#         self.global_pool = nn.AdaptiveAvgPool2d(1)
#
#         # 残差连接
#         self.residual = nn.Identity()
#
#     def forward(self, x):
#         # 通道注意力部分
#         avg_out = self.avg_pool(x)
#         channel_att = self.fc2(self.relu(self.fc1(avg_out)))
#         channel_att = self.sigmoid_c(channel_att)
#         x_c = x * channel_att
#
#         # 空间注意力部分
#         max_pool = torch.max(x_c, dim=1, keepdim=True)[0]
#         avg_pool = torch.mean(x_c, dim=1, keepdim=True)
#         spatial_att = self.sigmoid_s(self.conv_spatial(torch.cat([max_pool, avg_pool], dim=1)))
#         x_s = x_c * spatial_att
#
#         # 全局上下文
#         import torch.nn.functional as F
#         global_context = self.global_pool(x_s)
#         global_context = F.interpolate(global_context, size=x_s.shape[2:], mode='nearest')  # 上采样到x_s的形状
#         x_g = x_s + global_context
#
#         # 特征融合和残差连接
#         x_out = x_g + self.residual(x)
#
#         return x_out

# 确保 LocalContextShare 之后的层可以处理通道数
# 如果在 LocalContextShare 之后你有卷积层，确保这些层的 in_channels 参数正确设置

# 确保后续层的 in_channels 设置正确
import torch.nn.functional as F

class LocalContextShare(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(LocalContextShare, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)  # 全局池化
        self.pool2 = nn.AdaptiveAvgPool2d(2)  # 中间池化
        self.fc = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)  # 使用 reduction
        self.sigmoid = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.ones(1))  # 可学习参数

    def forward(self, x):
        context1 = self.pool1(x)  # (batch_size, in_channels, 1, 1)
        context2 = self.pool2(x)  # (batch_size, in_channels, 2, 2)

        # 计算上下文特征并进行通道缩减
        shared_feature = self.sigmoid(self.fc(context1) + self.fc(context2))

        # 上采样 shared_feature 以匹配输入 x 的尺寸
        shared_feature = F.interpolate(shared_feature, size=x.shape[2:], mode='bilinear', align_corners=False)

        # 将通道数恢复到与输入 x 相同
        shared_feature = shared_feature.expand(-1, x.size(1), -1, -1)  # 扩展到与 x 通道数相同

        return x * shared_feature * self.alpha  # 返回经过上下文加权的 x




class GCSCA(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(GCSCA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        self.sigmoid_c = nn.Sigmoid()

        # 空间注意力模块（多尺度卷积 + 稀疏）
        self.conv_spatial3 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.conv_spatial5 = nn.Conv2d(2, 2, kernel_size=5, padding=2, bias=False, groups=2)
        self.conv_spatial7 = nn.Conv2d(2, 2, kernel_size=7, padding=3, bias=False, groups=2)
        self.conv_spatial9 = nn.Conv2d(2, 1, kernel_size=9, padding=4, bias=False)  # 新增
        self.conv_fuse = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.sigmoid_s = nn.Sigmoid()

        # 全局上下文模块
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.final_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
        self.residual = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # 可学习权重
        self.weight_c = nn.Parameter(torch.ones(1))
        self.weight_s = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # 通道注意力
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        channel_att = self.fc2(self.relu(self.fc1(avg_out + max_out)))
        channel_att = self.sigmoid_c(channel_att)

        x_c = x * (self.weight_c * channel_att)  # 应用可学习权重

        # 空间注意力
        max_pool = torch.max(x_c, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x_c, dim=1, keepdim=True)
        spatial_input = torch.cat([max_pool, avg_pool], dim=1)

        # 多尺度卷积
        spatial_att3 = self.conv_spatial3(spatial_input)
        spatial_att5 = self.conv_spatial5(spatial_input)
        spatial_att7 = self.conv_spatial7(spatial_input)
        spatial_att9 = self.conv_spatial9(spatial_input)  # 新增

        # 融合空间注意力
        spatial_att = self.sigmoid_s(self.conv_fuse(spatial_att3 + spatial_att5 + spatial_att7 + spatial_att9))
        x_s = x_c * (self.weight_s * spatial_att)  # 应用可学习权重

        # 全局上下文
        global_context = self.global_pool(x_s)
        x_g = x_s + global_context

        # 特征融合和残差连接
        x_out = torch.cat([x_s, x_g], dim=1)
        x_out = self.final_conv(x_out) + self.residual(x)

        return x_out



# 轻量版GCSCA，用于neck部分
class GCSCA_Light(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super(GCSCA_Light, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.sigmoid_c = nn.Sigmoid()

        # 简化空间注意力模块
        self.conv_spatial5 = nn.Conv2d(2, 1, kernel_size=5, padding=2, bias=False)
        self.sigmoid_s = nn.Sigmoid()

    def forward(self, x):
        # 通道注意力部分
        avg_out = self.avg_pool(x)
        channel_att = self.fc(avg_out)
        channel_att = self.sigmoid_c(channel_att)
        x_c = x * channel_att

        # 空间注意力部分
        max_pool = torch.max(x_c, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x_c, dim=1, keepdim=True)
        spatial_att = self.conv_spatial5(torch.cat([max_pool, avg_pool], dim=1))
        spatial_att = self.sigmoid_s(spatial_att)
        x_s = x_c * spatial_att

        return x_s


# class EnhancedGCSCA(nn.Module):提升0.003
# 通道注意力：新增最大池化，与平均池化结果相加以增强特征。
# 空间注意力：引入多尺度卷积（3x3、5x5、7x7），捕捉不同尺度的空间信息。
# 残差连接：保留输入信息并优化梯度流。
# class GCSCA003(nn.Module):
#     def __init__(self, in_channels, reduction=16):
#         # super(EnhancedGCSCA, self).__init__()
#         super(GCSCA, self).__init__()
#
#         # 通道注意力模块
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)  # 新增最大池化
#         self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
#         self.sigmoid_c = nn.Sigmoid()
#
#         # 空间注意力模块（多尺度卷积）
#         self.conv_spatial3 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
#         self.conv_spatial5 = nn.Conv2d(2, 1, kernel_size=5, padding=2, bias=False)
#         self.conv_spatial7 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
#         self.sigmoid_s = nn.Sigmoid()
#
#         # 全局上下文模块
#         self.global_pool = nn.AdaptiveAvgPool2d(1)
#
#         # 特征融合
#         self.final_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
#         self.residual = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # 残差连接
#
#     def forward(self, x):
#         # 通道注意力部分
#         avg_out = self.avg_pool(x)
#         max_out = self.max_pool(x)  # 新增
#         channel_att = self.fc2(self.relu(self.fc1(avg_out + max_out)))  # 融合平均和最大池化
#         channel_att = self.sigmoid_c(channel_att)
#         x_c = x * channel_att
#
#         # 空间注意力部分（多尺度）
#         max_pool = torch.max(x_c, dim=1, keepdim=True)[0]
#         avg_pool = torch.mean(x_c, dim=1, keepdim=True)
#         spatial_att3 = self.conv_spatial3(torch.cat([max_pool, avg_pool], dim=1))
#         spatial_att5 = self.conv_spatial5(torch.cat([max_pool, avg_pool], dim=1))
#         spatial_att7 = self.conv_spatial7(torch.cat([max_pool, avg_pool], dim=1))
#         spatial_att = self.sigmoid_s(spatial_att3 + spatial_att5 + spatial_att7)  # 融合多尺度卷积
#         x_s = x_c * spatial_att
#
#         # 全局上下文
#         global_context = self.global_pool(x_s)
#         x_g = x_s + global_context
#
#         # 特征融合和残差连接
#         x_out = torch.cat([x_s, x_g], dim=1)
#         x_out = self.final_conv(x_out) + self.residual(x)
#
#         return x_out

# class GCSCA(nn.Module):
#     def __init__(self, in_channels, reduction=16):
#         super(GCSCA, self).__init__()
#
#         # 通道注意力模块
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
#         self.sigmoid_c = nn.Sigmoid()
#
#         # 空间注意力模块
#         self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
#         self.sigmoid_s = nn.Sigmoid()
#
#         # 全局上下文模块
#         self.global_pool = nn.AdaptiveAvgPool2d(1)
#
#         # 最终卷积，用于融合特征
#         self.final_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
#
#     def forward(self, x):
#         # 通道注意力部分
#         avg_out = self.avg_pool(x)
#         channel_att = self.fc2(self.relu(self.fc1(avg_out)))
#         channel_att = self.sigmoid_c(channel_att)
#         x_c = x * channel_att
#
#         # 空间注意力部分
#         max_pool = torch.max(x_c, dim=1, keepdim=True)[0]
#         avg_pool = torch.mean(x_c, dim=1, keepdim=True)
#         spatial_att = self.conv_spatial(torch.cat([max_pool, avg_pool], dim=1))
#         spatial_att = self.sigmoid_s(spatial_att)
#         x_s = x_c * spatial_att
#
#         # 全局上下文部分
#         global_context = self.global_pool(x_s)
#         x_g = x_s + global_context  # 将全局上下文广播并逐像素相加
#
#         # 特征融合
#         x_out = torch.cat([x_s, x_g], dim=1)
#         x_out = self.final_conv(x_out)
#
#         return x_out
