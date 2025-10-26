# Obtained from: https://github.com/NVlabs/SegFormer
# Modifications: Model construction with loop
# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
# A copy of the license is available at resources/license_segformer

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead


class MLP(nn.Module):
    """Linear Embedding.
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2).contiguous()   # (B, C, H*W) -> (B, H*W, c)
        x = self.proj(x)   # (B, H*W, C) -> (B, H*W, embed_dim)
        return x


@HEADS.register_module()
class SegFormerHead(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with
    Transformers
    """

    def __init__(self, **kwargs):
        super(SegFormerHead, self).__init__(
            input_transform='multiple_select', **kwargs)

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']
        conv_kernel_size = decoder_params['conv_kernel_size']

        self.linear_c = {}   #각 스케일 i마다 MLP(input_dim = ih_channels[i], embed_dim=embedding_dim)을 만들어 ModuleDict에 저장장
        for i, in_channels in zip(self.in_index, self.in_channels):
            self.linear_c[str(i)] = MLP(
                input_dim=in_channels, embed_dim=embedding_dim)
        self.linear_c = nn.ModuleDict(self.linear_c)
        
        #conv + norm + activation
        self.linear_fuse = ConvModule(
            in_channels=embedding_dim * len(self.in_index),
            out_channels=embedding_dim,
            kernel_size=conv_kernel_size,
            padding=0 if conv_kernel_size == 1 else conv_kernel_size // 2,
            norm_cfg=kwargs['norm_cfg'])

        self.linear_pred = nn.Conv2d(
            embedding_dim, self.num_classes, kernel_size=1)
        
        self.kd_projection = nn.Conv2d(self.num_classes, self.num_classes, kernel_size=1)


    def forward(self, inputs, kd_mode=None):
        x = inputs
        n, _, h, w = x[-1].shape  
        # for f in x:
        #     print(f.shape)

        _c = {}
        ''' 
        in_channels = [32, 64, 160, 256], embed_dim = 256
        x[0] = (B, 32, H/4, W/4)
        x[1] = (B, 64, H/8, W/8)
        x[2] = (B, 160, H/16, W/16)
        x[3] = (B, 256, H/32, W/32)
        
        [MLP 통과 후]
        _c[0] = (B, 256, H/4, W/4)
        _c[1] = (B, 256, H/8, W/8)
        _c[2] = (B, 256, H/16, W/16)
        _c[3] = (B, 256, H/32, W/32)

        모두 X[0]사이즈로 resize 후, 채널방향 concat 후, linear_fuse 통과 [B, embed_dim, H, W]
        '''
        for i in self.in_index:
            # mmcv.print_log(f'{i}: {x[i].shape}, {self.linear_c[str(i)]}')
            _c[i] = self.linear_c[str(i)](x[i]).permute(0, 2, 1).contiguous()
            _c[i] = _c[i].reshape(n, -1, x[i].shape[2], x[i].shape[3])
            if i != 0:
                _c[i] = resize(
                    _c[i],
                    size=x[0].size()[2:],
                    mode='bilinear',
                    align_corners=False)

        _c = self.linear_fuse(torch.cat(list(_c.values()), dim=1))
        
        #features = _c
        #if return_features and hasattr(self, 'kd_projection'):
        #    features = self.kd_projection(features)

        if self.dropout is not None:
            x = self.dropout(_c)
        else:
            x = _c
        x = self.linear_pred(x)   
        
        if kd_mode == 'gram':
            features = self.kd_projection(x)
            return x, features  
        return x