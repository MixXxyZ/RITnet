"""
Source::
https://github.com/th2l/Eye_VR_Segmentation
"""
import torch
import torch.nn.functional as F
from torch import nn

import math

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


""" ConvBNReLU"""


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


""" ConvReLU"""


class ConvReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )


""" InvertedResidual """


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MobileNetV2_CS(nn.Module):
    def __init__(self, num_classes=4, out_shape=(640, 400), width_mult=1.0, inverted_residual_setting=None,
                 round_nearest=8, init_weights=True):
        """
        MobileNet V2 CS main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2_CS, self).__init__()
        block = InvertedResidual
        input_channel = 32
        self.out_shape = out_shape
        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 2, 1],
                [6, 24, 3, 2],
                [6, 32, 4, 2],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)

        features = [ConvBNReLU(1, input_channel, stride=2)]  # 3 for color image

        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # building last several layers
        kn_size = 3
        features.append(ConvReLU(input_channel, 64, kernel_size=kn_size))

        self.features = nn.Sequential(*features)
        
        # building segmentation layer
        c_segmentation = [64, num_classes]

        segmentation_part1 = [ConvReLU(c_segmentation[0], c_segmentation[0], kernel_size=1),
                              nn.Upsample(scale_factor=4.0, mode='bilinear',
                                          align_corners=False)]

        up_part1 = [ConvReLU(c_segmentation[0], c_segmentation[1], kernel_size=1),
                    nn.Upsample(scale_factor=4.0, mode='bilinear', align_corners=False),
                    SELayer(channel=c_segmentation[1], reduction=4)]

        self.up_part1 = nn.Sequential(*up_part1)

        conv_up = [ConvReLU(c_segmentation[0], c_segmentation[1], kernel_size=kn_size),
                   ConvReLU(c_segmentation[1], c_segmentation[1], kernel_size=kn_size),
                   ConvReLU(c_segmentation[1], c_segmentation[1], kernel_size=kn_size),
                   nn.Upsample(scale_factor=4.0, mode='bilinear', align_corners=False)]
        self.conv_up_part1 = nn.Sequential(*conv_up)

        self.segm_part1 = nn.Sequential(*segmentation_part1)

        # weight initialization (Original)
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out')
        #         if m.bias is not None:
        #             nn.init.zeros_(m.bias)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.ones_(m.weight)
        #         nn.init.zeros_(m.bias)

        if init_weights:
            self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                # m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)

        x1 = self.segm_part1(x)

        x1_seg = self.conv_up_part1(x1)

        x1_up = self.up_part1(x1)

        x = x1_seg + x1_up
        
        x_softmax = F.softmax(x, dim=1)
        sgm = torch.argmax(x_softmax, dim=1)
        
        # return x_softmax, sgm    # original
        return sgm