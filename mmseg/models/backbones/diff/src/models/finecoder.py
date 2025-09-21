# Implementation for DIFF for paper 'Diffusion Features to Bridge Domain Gap for Semantic Segmentation'
# By Yuxiang Ji

from torch import nn
from torch.nn import functional as F
import torch.nn.init as init
import os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from archs.detectron2.resnet import BottleneckBlock

class DiftFinecoder_0(nn.Module):
    def __init__(self, in_channel=384, out_channels=[384, 384, 384, 384]):  
        super().__init__()
        self.in_channel = in_channel
        self.out_channels = out_channels
        self.bn = nn.BatchNorm2d(in_channel)
        self.down = nn.MaxPool2d(kernel_size=2, stride=(2, 2))  #resolution 절반

        nn.init.constant_(self.bn.weight, 1)  #bn weight 1
        nn.init.constant_(self.bn.bias, 0)    #bn bias 0으로 초기화
        
    def forward(self, dift_values):  
        dift_values = self.bn(dift_values)
        # Resolution High to Low
        highest_dift = F.interpolate(dift_values, scale_factor=2, mode='bilinear', align_corners=True)
        mid_dift = self.down(dift_values)
        low_dift = self.down(mid_dift)
        return highest_dift, dift_values, mid_dift, low_dift
'''
DiftFinecoder_0: input: [B, C, H, W] 
1. BatchNorm 적용: dift_values
2. Upsampling을 통해 원본보다 해상도 2배 증가: hiest_dift
3. MaxPooling 적용 (원본에서 2배 줄임): mid_dift
4. MaxPooling 적용 (mid_dift에서 2배 줄임): low_dift
return: highest_dift, dift_values, mid_dift, low_dift   -> 멀티 스케일 분석
'''

class DiftFinecoder(nn.Module):
    def __init__(self, in_channel=384, out_channels=[96, 384, 384, 768]):
        super().__init__()
        self.in_channel = in_channel
        self.out_channels = out_channels
        self.bn = nn.BatchNorm2d(in_channel)
        #donw_0: in_channel(384) -> out_channels(384), resolution 2배 감소 conv
        #down_1: in_channels(384) -> out_channels(768), rewolution 2배 감소 conv
        #up: in_channels -> out_channels(96), rewolusion 2배 증가 upconv
        self.down_0 = nn.Conv2d(in_channels=in_channel, out_channels=out_channels[2], kernel_size=2, stride=(2, 2))  
        self.down_1 = nn.Conv2d(in_channels=out_channels[2], out_channels=out_channels[3], kernel_size=2, stride=(2, 2))
        self.up = nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channels[0], kernel_size=2, stride=(2, 2))

        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)
        nn.init.kaiming_normal_(self.down_0.weight)
        nn.init.zeros_(self.down_0.bias)
        nn.init.kaiming_normal_(self.down_1.weight)
        nn.init.zeros_(self.down_1.bias)
        nn.init.kaiming_normal_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        
    
    def forward(self, dift_values):
        dift_values = self.bn(dift_values)
        # Resolution High to Low
        highest_dift = self.up(dift_values)
        mid_dift = self.down_0(dift_values)
        low_dift = self.down_1(mid_dift)
        return highest_dift, dift_values, mid_dift, low_dift
'''
dift_values: batch-norm 적용
highest_dift: dift_values에 2배 up conv
mid_dift: dift_values에 resolution 2배 감소 conv (원본 대비 2배 감소)
low_dift: mid_dift에 rewolution 2배 감소 conv (원본 대비 4배 감소)
return highest_dift, dift_values, mid_dift, low_dift
'''

class DiftUNetFinecoder(nn.Module):   #UNet style finecoder
    def __init__(self, in_channels=384, hidden_dim=[92, 384, 384, 768]):   #encoder/decoder 스테이지에서 사용될 채널 수
        super(DiftUNetFinecoder, self).__init__()

        # Encoder
        # enc1: [B, 384, H, W] -> [B, 384, H/2, W/2]  
        # enc2: [B, 384, H/2, W/2] -> [B, 768, H/4, W/4]
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim[2], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(hidden_dim[2], hidden_dim[3], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Multihead Attention
        # self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)

        # Decoder
        # dec1: [B, 784, H/4, W/4] -> [B, 384, H/2, W/2]  
        # dec2: [B, 384, H/2, W/2] -> [B, 384, H, W]
        # dec3: [B, 384, H, W] -> [B, 768, 2H, 2W]
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim[3], hidden_dim[2], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim[2], hidden_dim[1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim[1], hidden_dim[0], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        # Encoding
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(enc1_out)

        # Decoding
        dec1_out = F.upsample_bilinear(self.dec1(enc2_out), size=(enc1_out.shape[2], enc1_out.shape[3])) + enc1_out
        dec2_out = self.dec2(dec1_out) + x

        # Upsample to match original resolution
        dec3_out = self.dec3(dec2_out)

        return dec3_out, dec2_out, dec1_out, enc2_out
'''
enc1_out: [B, 384, H, W] -> [B, 384, H/2, W/2]  
enc2_out: [B, 384, H/2, W/2] -> [B, 768, H/4, W/4]
dec1_out: [B, 384, H/2, W/2] + enc1_out([B, 384, H/2, W/2])과 element_wise sum(skip connection) = [B, 384, H/2, W/2]
dec2_out: [B, 384, H/2, W/2] -> [B, 384, H, W] + 원본 데이터(skip connection)
dec3_out: [B, 384, H, W] -> [B, 768, 2H, 2W]
return: dec3_out(2배 업샘플링), dec2_out(원본 resolution), dec1_out(2배 감소), enc2_out(4배 감소)
'''

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=32):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.gn1 = nn.GroupNorm(num_groups, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.gn2 = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out += residual
        out = self.relu(out)
        return out
'''
input x[B, C_in, H, W]
-> Con1 -> GN -> RELU -> Conv2 -> GN + x -> RELU
output O[B, C_out, H, W] 
'''

class EnhancedFeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels, num_group=32):
        super(EnhancedFeatureFusion, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)  #2배 업샘플링
        self.resblock = ResBlock(out_channels, out_channels)

    def forward(self, x1, x2=None):
        x1 = self.deconv(x1)
        if x2 is not None:
            x1 = x1 + x2
        x1 = self.resblock(x1)
        return x1
'''
x1 업샘플링 [B, C_in, H, W] -> [B, C_out, 2H, 2W]
x1 resblock
output: [B, C_out, 2H, 2W]
'''

class EnhancedFusionNetwork(nn.Module):
    def __init__(self, in_channels=[768, 384, 192], out_channel=96):
        super(EnhancedFusionNetwork, self).__init__()
        self.fusion1 = EnhancedFeatureFusion(in_channels[0], in_channels[1]) #768, 384
        self.fusion2 = EnhancedFeatureFusion(in_channels[1], in_channels[2]) #384, 192 
        self.fusion3 = EnhancedFeatureFusion(in_channels[2], out_channel)  #192, 96

    def forward(self, x8, x16, x32):
        x = self.fusion1(x32, x16) #x32(768)를 upsample 한 후, x16과 동일 크기로 맞춘 후 sum, 최종 x
        x = self.fusion2(x, x8)
        x = self.fusion3(x)
        return x   #[B, 96, H, W]


class DiftStrideFinecoder(nn.Module):  #out_channel: 96
    def __init__(self, in_channels=[768, 384, 192], hidden_dim_x4=96, num_group=32):
        super().__init__()
        in_channels = in_channels[1:] + [hidden_dim_x4]   #in_channels [384, 192, 96]
        #EnhancedFusionNetwork([384, 192, 96], 96)
        self.fusion_net = EnhancedFusionNetwork(in_channels, hidden_dim_x4) 
        # self.in_x8 = nn.InstanceNorm2d(in_channels[2])
        # self.in_x16 = nn.InstanceNorm2d(in_channels)
        self.gn_x8 = nn.GroupNorm(num_group, in_channels[2])  #96
        self.gn_x16 = nn.GroupNorm(num_group, in_channels[1])  #192
        self.gn_x32 = nn.GroupNorm(num_group, in_channels[0])  #384
        # self.gn_x8 = nn.BatchNorm2d(in_channels[2])
        # self.gn_x16 = nn.BatchNorm2d(in_channels[1])
        # self.gn_x32 = nn.BatchNorm2d(in_channels[0])
        self.apply(self.weights_init)  #Conv: He + biase zero / Norm: (1, 0)

    def forward(self, x8, x16, x32):
        x8 = self.gn_x8(x8)   #96
        x16 = self.gn_x16(x16)  #192
        x32 = self.gn_x32(x32)  #384
        # x4 = self.fusion_net(x, x8, x16, x32)
        # print(x8.shape, x16.shape, x32.shape)
        x4 = self.fusion_net(x8, x16, x32)  #x32 -> +x16 -> resnet refine -> +x8 -> output(x4)
        return x4, x8, x16, x32

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

'''
return
- x32 (H/32, W/32)
- x16 (H/16, W/16)
- x8 (H/8, W/8)
- x4 (H/4, W/4)
'''