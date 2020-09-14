import torch
from torch import nn
import math

class _GlobalConvModule(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, squeeze_ratio=8):
        super(_GlobalConvModule, self).__init__()

        assert(kernel_size[0] % 2 != 0 and kernel_size[1] % 2 != 0)     # to prevent incompatible pad size

        pad0 = int((kernel_size[0] - 1) / 2)
        pad1 = int((kernel_size[1] - 1) / 2)
        # kernel size had better be odd number so as to avoid alignment error
        super(_GlobalConvModule, self).__init__()

        squeeze_channels = out_dim // squeeze_ratio
        self.conv_l1 = nn.Sequential(*[
            nn.Conv2d(in_dim, squeeze_channels, kernel_size=1),
            nn.BatchNorm2d(squeeze_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeeze_channels, out_dim, kernel_size=(kernel_size[0], 1), padding=(pad0, 0)),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        ])
        self.conv_l2 = nn.Sequential(*[
            nn.Conv2d(out_dim, squeeze_channels, kernel_size=1),
            nn.BatchNorm2d(squeeze_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeeze_channels, out_dim, kernel_size=(1, kernel_size[1]), padding=(0, pad1)),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        ])
        self.conv_r1 = nn.Sequential(*[
            nn.Conv2d(in_dim, squeeze_channels, kernel_size=1),
            nn.BatchNorm2d(squeeze_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeeze_channels, out_dim, kernel_size=(1, kernel_size[1]), padding=(0, pad1)),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        ])
        self.conv_r2 = nn.Sequential(*[
            nn.Conv2d(out_dim, squeeze_channels, kernel_size=1),
            nn.BatchNorm2d(squeeze_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeeze_channels, out_dim, kernel_size=(kernel_size[0], 1), padding=(pad0, 0)),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        ])

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        x = x_l + x_r
        return x

class _BoundaryRefineModule(nn.Module):
    def __init__(self, dim, squeeze_ratio=8, expand1x1_ratio=0.5, dilation_paths=1):
        super(_BoundaryRefineModule, self).__init__()
        
        # self.conv1 = nn.Sequential(*[
        #     nn.Conv2d(dim, dim, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(dim),
        #     nn.ReLU(inplace=True)
        # ])
        # self.conv2 = nn.Sequential(*[
        #     nn.Conv2d(dim, dim, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(dim),
        #     nn.ReLU(inplace=True)
        # ])

        self.conv1 = _FirePath(dim, dim, 2, squeeze_ratio=squeeze_ratio, expand1x1_ratio=expand1x1_ratio, dilation_paths=dilation_paths)

    def forward(self, x):
        # residual = self.conv1(x)
        # residual = self.conv2(residual)
        # out = x + residual
        # return out

        return self.conv1(x)

class _FireBlock(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes, dilation=1, padding=1):
        super(_FireBlock, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_batchNorm = nn.BatchNorm2d(squeeze_planes)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1_planes = expand1x1_planes
        if expand1x1_planes > 0:
            self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                    kernel_size=1)
            self.expand1x1_batchNorm = nn.BatchNorm2d(expand1x1_planes)
            self.expand1x1_activation = nn.ReLU(inplace=True)

        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, dilation=dilation, padding=padding)
        self.expand3x3_batchNorm = nn.BatchNorm2d(expand3x3_planes)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze_batchNorm(self.squeeze(x)))
        if self.expand1x1_planes > 0:
            return torch.cat([
                self.expand1x1_activation(self.expand1x1_batchNorm(self.expand1x1(x))),
                self.expand3x3_activation(self.expand3x3_batchNorm(self.expand3x3(x)))
            ], 1)
        else:
            return self.expand3x3_activation(self.expand3x3_batchNorm(self.expand3x3(x)))

class _FirePath(nn.Module):
    def __init__(self, in_channels, out_channels, num_fire_blocks, squeeze_ratio=8, expand1x1_ratio=0.5, dilation_rates=[1]):
        super(_FirePath, self).__init__()
        assert(num_fire_blocks > 0)
		
        self.in_channels = in_channels
        self.out_channels = out_channels
        squeeze_channels = out_channels // squeeze_ratio
        expand1x1_channels = int(out_channels * expand1x1_ratio)
        expand3x3_channels = out_channels - expand1x1_channels
        self.residual_block_list = nn.ModuleList()
        self.residual_block_list.append(_FireBlock(in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels, dilation=dilation_rates[0], padding=dilation_rates[0]))
        for i in range(1, num_fire_blocks):
            rate_idx = i % len(dilation_rates)
            self.residual_block_list.append(_FireBlock(out_channels, squeeze_channels, expand1x1_channels, expand3x3_channels, dilation=dilation_rates[rate_idx], padding=dilation_rates[rate_idx]))
        
    def forward(self, x):

        for i, layer in enumerate(self.residual_block_list):
            if i==0 and self.in_channels != self.out_channels:
                x = layer(x)
            else:
                residual = layer(x)
                x = x + residual

        return x

class UpsamplingBottleneck(nn.Module):
    """The upsampling bottlenecks upsample the feature map resolution using max
    pooling indices stored from the corresponding downsampling bottleneck.

    Main branch:
    1. 1x1 convolution with stride 1 that decreases the number of channels by
    ``internal_ratio``, also called a projection;
    2. max unpool layer using the max pool indices from the corresponding
    downsampling max pool layer.

    Extension branch:
    1. 1x1 convolution with stride 1 that decreases the number of channels by
    ``internal_ratio``, also called a projection;
    2. transposed convolution (by default, 3x3);
    3. 1x1 convolution which increases the number of channels to
    ``out_channels``, also called an expansion;
    4. dropout as a regularizer.

    Keyword arguments:
    - in_channels (int): the number of input channels.
    - out_channels (int): the number of output channels.
    - internal_ratio (int, optional): a scale factor applied to ``in_channels``
     used to compute the number of channels after the projection. eg. given
     ``in_channels`` equal to 128 and ``internal_ratio`` equal to 2 the number
     of channels after the projection is 64. Default: 4.
    - kernel_size (int, optional): the kernel size of the filters used in the
    convolution layer described above in item 2 of the extension branch.
    Default: 3.
    - padding (int, optional): zero-padding added to both sides of the input.
    Default: 0.
    - dropout_prob (float, optional): probability of an element to be zeroed.
    Default: 0 (no dropout).
    - bias (bool, optional): Adds a learnable bias to the output if ``True``.
    Default: False.
    - relu (bool, optional): When ``True`` ReLU is used as the activation
    function; otherwise, PReLU is used. Default: True.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 internal_ratio=4,
                 kernel_size=3,
                 padding=0,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super().__init__()

        # Check in the internal_scale parameter is within the expected range
        # [1, channels]
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}. "
                               .format(in_channels, internal_ratio))

        internal_channels = in_channels // internal_ratio

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        # Main branch - max pooling followed by feature map (channels) padding
        self.main_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels))

        # Remember that the stride is the same as the kernel_size, just like
        # the max pooling layers
        self.main_unpool1 = nn.MaxUnpool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))

        # Extension branch - 1x1 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution. Number
        # of channels is doubled.

        # 1x1 projection convolution with stride 1
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, internal_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(internal_channels), activation)

        # Transposed convolution
        self.ext_conv2 = nn.Sequential(
            nn.ConvTranspose2d(
                internal_channels,
                internal_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                # output_padding=1,     # Modified by MixXxyZ
                bias=bias), nn.BatchNorm2d(internal_channels), activation)

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels), activation)

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after concatenating the branches
        self.out_activation = activation

    def forward(self, x, max_indices):
        # Main branch shortcut
        """ (original)
        main = self.main_conv1(x)
        main = self.main_unpool1(main, max_indices)
        """
        main = self.main_unpool1(x, max_indices)
        main = self.main_conv1(main)
        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Add main and extension branches
        out = main + ext

        return self.out_activation(out)

class MixNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, init_weights=False):
        super(MixNet, self).__init__()
        
        self.enc1 = nn.Sequential(*[
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False, return_indices=True),
        ])
        # Assume input image size = (224, 224) !!!  224 * (0.5) = 112
        # We use kernel size = 111 which is compatible for GCN module (accept odd number for kernel size)
        # and make the output size equals to the input size
        #self.gcn1 = _GlobalConvModule(16, 8, (111, 111))    
        # self.brm1 = _BoundaryRefineModule(16, dilation_paths=1)
        #############################################
        self.enc2 = nn.Sequential(*[
            _FirePath(16, 64, 4, dilation_rates=[1,2,3,5]),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False, return_indices=True),
        ])
        #self.gcn2 = _GlobalConvModule(64, 32, (55, 55))    # Assume input image size = (224, 224) !!!  224 * (0.5)^2 = 56
        # self.brm2 = _BoundaryRefineModule(64, dilation_paths=1)
        #############################################
        self.enc3 = nn.Sequential(*[
            _FirePath(64, 128, 16, dilation_rates=[1,2,3,5]),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False, return_indices=True),
        ])
        #self.gcn3 = _GlobalConvModule(128, 64, (27, 27))    # Assume input image size = (224, 224) !!!  224 * (0.5)^3 = 28
        # self.brm3 = _BoundaryRefineModule(128, dilation_paths=1)
        #############################################
        self.unpool3 = UpsamplingBottleneck(128, 64, internal_ratio=4, kernel_size=2, padding=0, dropout_prob=0, bias=True, relu=True)  # in_channels = 128 --> from enc3 only !!!
        self.dec3 =  _FirePath(64, 64, 2, dilation_rates=[1,2,3,5])
        # self.brm3_dec = _BoundaryRefineModule(64, dilation_paths=1)
        #############################################
        self.unpool2 = UpsamplingBottleneck(128, 16, internal_ratio=4, kernel_size=2, padding=0, dropout_prob=0, bias=True, relu=True)  # in_channels = 128 --> from enc2 and dec3 (see *1)
        self.dec2 = _FirePath(16, 16, 1, dilation_rates=[1,2,3,5])
        # self.brm2_dec = _BoundaryRefineModule(16, dilation_paths=1)
        #############################################
        self.unpool1 = UpsamplingBottleneck(32, out_channels, internal_ratio=4, kernel_size=2, padding=0, dropout_prob=0, bias=True, relu=True)  # in_channels = 32 --> from enc1 and dec2 (see *2)
        self.dec1 = nn.Sequential(
            *[nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
               nn.BatchNorm2d(out_channels),
               nn.ReLU(inplace=True)]
        )
        # self.brm1_dec = _BoundaryRefineModule(out_channels, squeeze_ratio=1, expand1x1_ratio=0, dilation_paths=1)
        #############################################

        if init_weights:
            self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        enc1, pool_idx1 = self.enc1(x)
        # enc1_skip = self.brm1(self.gcn1(enc1))
        # enc1_skip = self.gcn1(enc1)
        enc2, pool_idx2 = self.enc2(enc1)
        # enc2_skip = self.brm2(self.gcn2(enc2))
        # enc3 = self.brm3(self.gcn3(self.enc3(enc2)))
        # enc2_skip = self.gcn2(enc2)
        # enc3 = self.gcn3(self.enc3(enc2))
        enc3, pool_idx3 = self.enc3(enc2)

        # dec3 = self.brm3_dec(self.dec3(enc3))
        # dec2 = self.brm2_dec(self.dec2(torch.cat([enc2_skip, dec3], 1)))   # *1
        # dec1 = self.brm1_dec(self.dec1(torch.cat([enc1_skip, dec2], 1)))   # *2
        
        dec3 = self.dec3(self.unpool3(enc3, pool_idx3))
        pool_idx2 = torch.cat([pool_idx2, pool_idx2], 1)
        dec2 = self.dec2(self.unpool2(torch.cat([enc2, dec3], 1), pool_idx2))   # *1
        pool_idx1 = torch.cat([pool_idx1, pool_idx1], 1)
        dec1 = self.dec1(self.unpool1(torch.cat([enc1, dec2], 1), pool_idx1))   # *2
        return dec1