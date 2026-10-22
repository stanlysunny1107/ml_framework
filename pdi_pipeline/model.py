"""Original supplied BE-Net architecture, without the single-patient CLI."""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from scipy.io import loadmat, savemat


class SpectralGradientConv(nn.Module):
    """Apply a learned 5 x 5 Laplacian-weighted spatial filter and entropy-based channel attention.

    Input spectral bands are treated as channels. This operation computes spatial
    gradients; it does not explicitly differentiate along the wavelength axis."""

    def __init__(self, inch, outch, groups, ks=5, stride=1, first_layer="False"):
        super(SpectralGradientConv, self).__init__()
        self.first = first_layer
        self.padding = int(ks // 2)
        self.stride = (stride, stride)
        self.inch, self.outch, self.groups = (int(inch), int(outch), int(groups))
        self.weight = nn.Parameter(
            torch.Tensor(self.outch, self.inch // groups, ks, ks)
        )
        self.conv3x3 = nn.Conv2d(self.outch, self.outch, 3, 1, 1)
        self.bn = nn.BatchNorm2d(self.outch)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.reset_parameters()
        self.gmaxpool = torch.nn.AdaptiveMaxPool2d(1)
        self.gavgpool = torch.nn.AdaptiveAvgPool2d(1)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        epsilon = 1e-10
        # Repeat the spatial stencil for each input channel of the learned filters.
        list_x = []
        for i in range(self.weight.shape[1]):
            list_x.append(
                torch.tensor(
                    [
                        [-2, -4, -4, -4, -2],
                        [-4, 0, 8, 0, -4],
                        [-4, 8, 24, 8, -4],
                        [-4, 0, 8, 0, -4],
                        [-2, -4, -4, -4, -2],
                    ],
                    device=x.device,
                )
            )
        list_x = torch.stack(list_x, 0)
        weight_x = torch.mul(self.weight, list_x)
        input_x = F.conv2d(
            x, weight_x, stride=self.stride, padding=self.padding, groups=self.groups
        )
        conv = self.relu(self.bn(self.conv3x3(input_x)))
        gmax_group1, gavg_group1 = (self.gmaxpool(conv), self.gavgpool(conv))
        cat = torch.cat([gmax_group1.squeeze(-1), gavg_group1.squeeze(-1)], dim=2)
        B, C, N = cat.shape
        probs = torch.softmax(cat, dim=2)
        log_probs = torch.log(probs + epsilon)
        entropy = -torch.sum(probs * log_probs, dim=2)
        max_entropy = math.log(N)
        normalized_entropy = entropy / max_entropy
        weights = 1 - normalized_entropy
        out = weights.unsqueeze(-1).unsqueeze(-1) * conv + conv
        return out


class ChannelBoundaryAttention(nn.Module):
    """Fuse encoder and decoder features using grouped multiscale convolutions
    and entropy-based channel attention."""

    def __init__(self, inchannel):
        super(ChannelBoundaryAttention, self).__init__()
        self.groupchannel = inchannel // 4
        self.groupconv1 = nn.Conv2d(
            in_channels=self.groupchannel,
            out_channels=inchannel,
            kernel_size=1,
            padding=0,
        )
        self.groupconv3 = nn.Conv2d(
            in_channels=self.groupchannel,
            out_channels=inchannel,
            kernel_size=3,
            padding=1,
        )
        self.groupconv5 = nn.Conv2d(
            in_channels=self.groupchannel,
            out_channels=inchannel,
            kernel_size=3,
            dilation=2,
            padding=2,
        )
        self.groupconv7 = nn.Conv2d(
            in_channels=self.groupchannel,
            out_channels=inchannel,
            kernel_size=3,
            dilation=3,
            padding=3,
        )
        self.gmaxpool = torch.nn.AdaptiveMaxPool2d(1)
        self.gavgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.epsilon = 1e-10
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, skip, decoderx):
        skip_decoder_cat = torch.cat([skip, decoderx], dim=1)
        group1 = self.groupconv5(skip_decoder_cat[:, : self.groupchannel, :, :])
        group3 = self.groupconv7(skip_decoder_cat[:, self.groupchannel * 3 :, :, :])
        group5 = self.groupconv1(
            skip_decoder_cat[
                :, int(self.groupchannel * 2) : int(self.groupchannel * 3), :, :
            ]
        )
        group7 = self.groupconv3(
            skip_decoder_cat[
                :, int(self.groupchannel) : int(self.groupchannel * 2), :, :
            ]
        )
        groupconv = group1 + group3 + group5 + group7
        groupconv_ = self.relu(groupconv)
        gmax_group1, gavg_group1 = (
            self.gmaxpool(groupconv_),
            self.gavgpool(groupconv_),
        )
        cat = torch.cat([gmax_group1.squeeze(-1), gavg_group1.squeeze(-1)], dim=2)
        B, C, N = cat.shape
        probs = torch.softmax(cat, dim=2)
        log_probs = torch.log(probs + self.epsilon)
        entropy = -torch.sum(probs * log_probs, dim=2)
        max_entropy = math.log(N)
        normalized_entropy = entropy / max_entropy
        weights = 1 - normalized_entropy
        out = weights.unsqueeze(-1).unsqueeze(-1) * groupconv_ + groupconv_
        return out


class PyramidDecoder(nn.Module):
    """Refine decoder features using dilated convolutions, channel entropy,
    and multiscale spatial pooling."""

    def __init__(self, in_channel, out_channel):
        super(PyramidDecoder, self).__init__()
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(out_channel),
        )
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channel,
                out_channels=out_channel,
                kernel_size=3,
                dilation=2,
                padding=2,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(out_channel),
        )
        self.conv7x7 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channel,
                out_channels=out_channel,
                kernel_size=3,
                dilation=3,
                padding=3,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(out_channel),
        )
        self.maxpool3x3 = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        self.maxpool5x5 = nn.MaxPool2d(kernel_size=5, padding=2, stride=1)
        self.maxpool7x7 = nn.MaxPool2d(kernel_size=7, padding=3, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.epsilon = 1e-10

    def forward(self, x):
        x1 = self.conv3x3(x)
        x2 = self.conv5x5(x1)
        x3 = self.conv7x7(x2)
        out = x1 + x2 + x3
        B, C, H, W = out.shape
        probs = torch.softmax(out, dim=1)
        log_probs = torch.log(probs + self.epsilon)
        entropy = -torch.sum(probs * log_probs, dim=1)
        max_entropy = math.log(C)
        normalized_entropy = entropy / max_entropy
        area_ = self.sigmoid(out)
        a3x3 = self.maxpool3x3(area_)
        a5x5 = self.maxpool5x5(area_)
        a7x7 = self.maxpool7x7(area_)
        area = (a3x3 + a5x5 + a7x7) / 3 * area_
        output = out * (normalized_entropy.unsqueeze(1) + area) + out
        return output


class ConvBlock(nn.Module):
    """Apply two convolution, batch-normalization, and ReLU stages."""

    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):
    """Upsample features by two and reduce the channel dimension."""

    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class BENet(nn.Module):
    """BE-Net segmentation architecture adapted from the supplied implementation.

    Accept tensors with shape (batch, spectral_bands, height, width) and return
    unscaled class logits with shape (batch, classes, height, width).
    Use cross-entropy loss for training and argmax for class prediction."""

    def __init__(self, in_ch=3, out_ch=1):
        super(BENet, self).__init__()
        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = ConvBlock(in_ch, filters[0])
        self.Conv2 = ConvBlock(filters[0], filters[1])
        self.Conv3 = ConvBlock(filters[1], filters[2])
        self.Conv4 = ConvBlock(filters[2], filters[3])
        self.Conv5 = ConvBlock(filters[3], filters[4])
        self.Up5 = UpConv(filters[4], filters[3])
        self.Up_conv5 = PyramidDecoder(filters[4], filters[3])
        self.Up4 = UpConv(filters[3], filters[2])
        self.Up_conv4 = PyramidDecoder(filters[3], filters[2])
        self.Up3 = UpConv(filters[2], filters[1])
        self.Up_conv3 = PyramidDecoder(filters[2], filters[1])
        self.Up2 = UpConv(filters[1], filters[0])
        self.Up_conv2 = PyramidDecoder(filters[1], filters[0])
        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.spectralgradconv1 = SpectralGradientConv(
            in_ch, filters[0], 1, ks=5, stride=1, first_layer="False"
        )
        self.spectralgradconv2 = SpectralGradientConv(
            filters[0], filters[1], 1, ks=5, stride=1, first_layer="False"
        )
        self.spectralgradconv3 = SpectralGradientConv(
            filters[1], filters[2], 1, ks=5, stride=1, first_layer="False"
        )
        self.spectralgradconv4 = SpectralGradientConv(
            filters[2], filters[3], 1, ks=5, stride=1, first_layer="False"
        )
        self.spectralgradconv5 = SpectralGradientConv(
            filters[3], filters[4], 1, ks=5, stride=1, first_layer="False"
        )
        self.cba5 = ChannelBoundaryAttention(filters[4])
        self.cba4 = ChannelBoundaryAttention(filters[3])
        self.cba3 = ChannelBoundaryAttention(filters[2])
        self.cba2 = ChannelBoundaryAttention(filters[1])

    def forward(self, x):
        e1 = self.Conv1(x)
        se1 = self.spectralgradconv1(x)
        e1 = se1 + e1
        e2 = self.Maxpool1(e1)
        se2 = self.spectralgradconv2(e2)
        e2 = self.Conv2(e2)
        e2 = se2 + e2
        e3 = self.Maxpool2(e2)
        se3 = self.spectralgradconv3(e3)
        e3 = self.Conv3(e3)
        e3 = se3 + e3
        e4 = self.Maxpool3(e3)
        se4 = self.spectralgradconv4(e4)
        e4 = self.Conv4(e4)
        e4 = se4 + e4
        e5 = self.Maxpool4(e4)
        se5 = self.spectralgradconv5(e5)
        e5 = self.Conv5(e5)
        e5 = se5 + e5
        d5 = F.interpolate(self.Up5(e5), size=e4.shape[-2:], mode="nearest")
        d5 = self.cba5(e4, d5)
        d5 = self.Up_conv5(d5)
        d4 = F.interpolate(self.Up4(d5), size=e3.shape[-2:], mode="nearest")
        d4 = self.cba4(e3, d4)
        d4 = self.Up_conv4(d4)
        d3 = F.interpolate(self.Up3(d4), size=e2.shape[-2:], mode="nearest")
        d3 = self.cba3(e2, d3)
        d3 = self.Up_conv3(d3)
        d2 = F.interpolate(self.Up2(d3), size=e1.shape[-2:], mode="nearest")
        d2 = self.cba2(e1, d2)
        d2 = self.Up_conv2(d2)
        out = self.Conv(d2)
        return out
