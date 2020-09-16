import torch
import math
import torch.nn as nn
import torch.nn.functional as F

# Layer Normalization (Ref. https://arxiv.org/abs/1607.06450) 
class LayerNorm(nn.Module):

    def __init__(self, num_features, eps=1e-12, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):

        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)

        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y

class CascadedRefinement_block(nn.Module):
    def __init__(self,input_channels,output_channels):
        super(CascadedRefinement_block, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=(4,4),padding=(2,2))
        self.lay1 = LayerNorm(output_channels, eps=1e-12, affine=True)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2,inplace=True)

        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=(4,4),padding=(2,2))
        self.lay2 = LayerNorm(output_channels, eps=1e-12, affine=True)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2,inplace=True)

    def forward(self, x):
        x1 = self.relu1(self.lay1(self.conv1(x)))
        x2 = self.relu2(self.lay2(self.conv2(x1)))

        return x2

class EyeMMS(nn.Module):
    def __init__(self,in_channels=1,out_channels=4, init_weights=False):
        super(EyeMMS, self).__init__()

        self.refinement1 = CascadedRefinement_block(input_channels=in_channels, output_channels=32)
        self.refinement2 = CascadedRefinement_block(input_channels=33, output_channels=32)    # 32 + 1
        self.refinement3 = CascadedRefinement_block(input_channels=33, output_channels=16)    # 32 + 1
        self.refinement4 = CascadedRefinement_block(input_channels=17, output_channels=16)    # 16 + 1
        self.refinement5 = CascadedRefinement_block(input_channels=17, output_channels=16)     # 16 + 1
        self.out_layer = nn.Conv2d(16, out_channels, kernel_size=(1,1),padding=(0,0))

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
                m.bias.data.zero_()

    def forward(self, x):

        self.x1_pre = F.interpolate(x,size=(40, 25),mode='nearest')
        self.x1 = self.refinement1(self.x1_pre)
        self.x1_up = F.interpolate(self.x1,size=(80, 50),mode='nearest')
        self.x_down1 = F.interpolate(x,size=(80, 50),mode='nearest')
        
        self.x2_pre = torch.cat((self.x1_up,self.x_down1),dim=1)
        self.x2 = self.refinement2(self.x2_pre)
        self.x2_up = F.interpolate(self.x2,size=(160, 100),mode='nearest')
        self.x_down2 = F.interpolate(x,size=(160, 100),mode='nearest')

        self.x3_pre = torch.cat((self.x2_up,self.x_down2),dim=1)
        self.x3 = self.refinement3(self.x3_pre)
        self.x3_up = F.interpolate(self.x3,size=(320, 200),mode='nearest')
        self.x_down3 = F.interpolate(x,size=(320, 200),mode='nearest')

        self.x4_pre = torch.cat((self.x3_up,self.x_down3),dim=1)
        self.x4 = self.refinement4(self.x4_pre)
        self.x4_up = F.interpolate(self.x4,size=(640, 400),mode='nearest')
        self.x_down4 = F.interpolate(x,size=(640, 400),mode='nearest')

        self.x5_pre = torch.cat((self.x4_up,self.x_down4),dim=1)
        self.x5 = self.refinement5(self.x5_pre)
        self.x5_down = F.interpolate(self.x5,size=(640, 400),mode='nearest')

        self.out = self.out_layer(self.x5_down)
        
        return self.out


