from __future__ import absolute_import

import torch.nn as nn
import torch


__all__ = ['AlexNetV1', 'AlexNetV2', 'AlexNetV3']


class _BatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, *args, **kwargs):
        super(_BatchNorm2d, self).__init__(
            num_features, *args, eps=1e-6, momentum=0.05, **kwargs)


class _AlexNet(nn.Module):

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

class AlexNetV3(_AlexNet):
    output_stride = 8

    def __init__(self):
        super(AlexNetV3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 192, 11, 2),
            _BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(192, 512, 5, 1),
            _BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(512, 768, 3, 1),
            _BatchNorm2d(768),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(768, 768, 3, 1),
            _BatchNorm2d(768),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(768, 512, 3, 1),
            _BatchNorm2d(512))

class SiameseNetRGB(nn.Module):

    def forward(self, x1):
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x = self.conv4(x1)
        x = self.conv5(x)
        return x

    def __init__(self):
        super(SiameseNetRGB, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            _BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, groups=2))


class SiameseNetThermal(nn.Module):

    def forward(self, x2):
        x2 = self.conv1(x2)
        x2 = self.conv2(x2)
        x2 = self.conv3(x2)
        x = self.conv4(x2)
        x = self.conv5(x)
        return x

    def __init__(self):
        super(SiameseNetThermal, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 96, 11, 2),
            _BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, groups=2))


class SSMA(nn.Module):

    def __init__(self, in_channels_rgb, in_channels_ir):
        super(SSMA, self).__init__()

        #RGB BRANCH
        self.conv1_rgb = nn.Sequential(
            nn.Conv2d(in_channels_rgb, 192, 11, 2),
            _BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))

        self.conv2_rgb = nn.Sequential(
            nn.Conv2d(192, 512, 5, 1),
            _BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))

        self.conv3_rgb = nn.Sequential(
            nn.Conv2d(512, 768, 3, 1),
            _BatchNorm2d(768),
            nn.ReLU(inplace=True))

        self.conv4_rgb = nn.Sequential(
            nn.Conv2d(768, 768, 3, 1),
            _BatchNorm2d(768),
            nn.ReLU(inplace=True))


        #IR BRANCH
        self.conv1_t = nn.Sequential(
            nn.Conv2d(in_channels_ir, 192, 11, 2),
            _BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))

        self.conv2_t = nn.Sequential(
            nn.Conv2d(192, 512, 5, 1),
            _BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))

        self.conv3_t = nn.Sequential(
            nn.Conv2d(512, 768, 3, 1),
            _BatchNorm2d(768),
            nn.ReLU(inplace=True))

        self.conv4_t = nn.Sequential(
            nn.Conv2d(768, 768, 3, 1),
            _BatchNorm2d(768),
            nn.ReLU(inplace=True))

        # SSMA BLOCK
        self.reduction_rate = 16

        self.ssma_contract = nn.Conv2d(2*768, 2*768//self.reduction_rate, 3, 1, padding=1)
        self.ssma_expand = nn.Conv2d(2*768//self.reduction_rate, 2*768, 3, padding=1)

        self.out = nn.Sequential(
            nn.Conv2d(2*768, 768, 3, 1),
            _BatchNorm2d(768))

    def forward(self, x_rgb, x_ir):

        # rgb branch
        x_rgb = self.conv1_rgb(x_rgb)
        x_rgb = self.conv2_rgb(x_rgb)
        x_rgb = self.conv3_rgb(x_rgb)
        x_rgb = self.conv4_rgb(x_rgb)

        # ir branch
        x_ir = self.conv1_t(x_ir)
        x_ir = self.conv2_t(x_ir)
        x_ir = self.conv3_t(x_ir)
        x_ir = self.conv4_t(x_ir)

        x_rgbir = torch.cat((x_rgb, x_ir), 1)

        ssma_reduction = torch.relu(self.ssma_contract(x_rgbir))
        ssma_expand = torch.sigmoid(self.ssma_expand(ssma_reduction))

        mul = x_rgbir * ssma_expand

        return self.out(mul)

class Dense(nn.Module):

    def __init__(self, in_channels_rgb, in_channels_ir):
        super(Dense, self).__init__()

        #RGB BRANCH
        self.conv1_rgb = nn.Sequential(
            nn.Conv2d(in_channels_rgb, 96, 11, 2),
            _BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.MaxPool2d(3, 2))
        self.conv2_rgb = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.MaxPool2d(3, 2))
        self.conv3_rgb = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1))
        self.conv4_rgb = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1))

        #IR BRANCH
        self.conv1_ir = nn.Sequential(
            nn.Conv2d(in_channels_ir, 96, 11, 2),
            _BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2_ir = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3_ir = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv4_ir = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))


        # Shared layers
        self.conv3_shared = nn.Sequential(
            nn.Conv2d(512, 384, 5, 1, groups=2),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(1152, 512, 3, 1, groups=2),
            _BatchNorm2d(512))



    def forward(self, x_rgb, x_v):

        # rgb branch
        x_rgb = self.conv1_rgb(x_rgb)
        x_rgb = self.conv2_rgb(x_rgb)
        # ir branch
        x_v = self.conv1_ir(x_v)
        x_v = self.conv2_ir(x_v)

        x_shared = torch.cat((x_rgb, x_v), 1)
        x_shared = self.conv3_shared(x_shared)


        x_rgb = self.conv3_rgb(x_rgb)
        x_rgb = self.conv4_rgb(x_rgb)


        x_v = self.conv3_ir(x_v)
        x_v = self.conv4_ir(x_v)
        x_final = torch.cat((x_rgb, x_shared, x_v), 1)
        x_final = self.conv5(x_final)

        return x_final

class DenseSSMA(nn.Module):

    def __init__(self, in_channels_rgb, in_channels_ir):
        super(Dense, self).__init__()

        #RGB BRANCH
        self.conv1_rgb = nn.Sequential(
            nn.Conv2d(in_channels_rgb, 96, 11, 2),
            _BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.MaxPool2d(3, 2))
        self.conv2_rgb = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.MaxPool2d(3, 2))
        self.conv3_rgb = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1))
        self.conv4_rgb = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1))

        #IR BRANCH
        self.conv1_ir = nn.Sequential(
            nn.Conv2d(in_channels_ir, 96, 11, 2),
            _BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2_ir = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3_ir = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv4_ir = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))


        # Shared layers
        self.conv3_shared = nn.Sequential(
            nn.Conv2d(512, 384, 5, 1, groups=2),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(1152, 512, 3, 1, groups=2),
            _BatchNorm2d(512))

        # SSMA BLOCK
        self.reduction_rate = 16

        self.ssma_contract = nn.Conv2d(1152, 1152//self.reduction_rate, 3, 1, padding=1)
        self.ssma_expand = nn.Conv2d(1152//self.reduction_rate, 2*768, 3, padding=1)

        self.out = nn.Sequential(
            nn.Conv2d(1152, 768, 3, 1),
            _BatchNorm2d(768))

    def forward(self, x_rgb, x_v):

        # rgb branch
        x_rgb = self.conv1_rgb(x_rgb)
        x_rgb = self.conv2_rgb(x_rgb)
        # ir branch
        x_v = self.conv1_ir(x_v)
        x_v = self.conv2_ir(x_v)

        x_shared = torch.cat((x_rgb, x_v), 1)
        x_shared = self.conv3_shared(x_shared)


        x_rgb = self.conv3_rgb(x_rgb)
        x_rgb = self.conv4_rgb(x_rgb)


        x_v = self.conv3_ir(x_v)
        x_v = self.conv4_ir(x_v)
        x_rgbir = torch.cat((x_rgb, x_shared, x_v), 1)
        ssma_reduction = torch.relu(self.ssma_contract(x_rgbir))
        ssma_expand = torch.sigmoid(self.ssma_expand(ssma_reduction))

        mul = x_rgbir * ssma_expand

        return self.out(mul)
