import argparse

from mmcv import Config
from mmcv.cnn import get_model_complexity_info
from mmcv.cnn.utils.flops_counter import flops_to_string, params_to_string

from mmseg.models import build_segmentor
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[2048, 1024],
        help='input image size')
    args = parser.parse_args()
    return args

def sra_flops(h, w, r, dim, num_heads):
    dim_h = dim / num_heads
    n1 = h * w
    n2 = h / r * w / r

    f1 = n1 * dim_h * n2 * num_heads
    f2 = n1 * n2 * dim_h * num_heads

    return f1 + f2


def get_tr_flops(net, input_shape):
    def custom_forward_dummy(self, img):
        """Custom dummy forward function for FLOPs calculation."""
        img_metas = [{'filename': 'dummy.jpg'}]
        seg_logit = self.encode_decode(img, img_metas)
        return seg_logit
    original_forward = net.forward
    net.forward = custom_forward_dummy.__get__(net, net.__class__)
    
    try:
        flops, params = get_model_complexity_info(net, input_shape, as_strings=False)
    finally:
        net.forward = original_forward
    
    _, H, W = input_shape
    backbone = net.backbone
    
    try:
        stage1 = sra_flops(H // 4, W // 4,
                           backbone.block1[0].attn.sr_ratio,
                           backbone.block1[0].attn.dim,
                           backbone.block1[0].attn.num_heads) * len(backbone.block1)
        stage2 = sra_flops(H // 8, W // 8,
                           backbone.block2[0].attn.sr_ratio,
                           backbone.block2[0].attn.dim,
                           backbone.block2[0].attn.num_heads) * len(backbone.block2)
        stage3 = sra_flops(H // 16, W // 16,
                           backbone.block3[0].attn.sr_ratio,
                           backbone.block3[0].attn.dim,
                           backbone.block3[0].attn.num_heads) * len(backbone.block3)
        stage4 = sra_flops(H // 32, W // 32,
                           backbone.block4[0].attn.sr_ratio,
                           backbone.block4[0].attn.dim,
                           backbone.block4[0].attn.num_heads) * len(backbone.block4)
    except:
        stage1 = sra_flops(H // 4, W // 4,
                           backbone.block1[0].attn.squeeze_ratio,
                           64,
                           backbone.block1[0].attn.num_heads) * len(backbone.block1)
        stage2 = sra_flops(H // 8, W // 8,
                           backbone.block2[0].attn.squeeze_ratio,
                           128,
                           backbone.block2[0].attn.num_heads) * len(backbone.block2)
        stage3 = sra_flops(H // 16, W // 16,
                           backbone.block3[0].attn.squeeze_ratio,
                           320,
                           backbone.block3[0].attn.num_heads) * len(backbone.block3)
        stage4 = sra_flops(H // 32, W // 32,
                           backbone.block4[0].attn.squeeze_ratio,
                           512,
                           backbone.block4[0].attn.num_heads) * len(backbone.block4)

    print(f"SRA FLOPs: {stage1 + stage2 + stage3 + stage4}")
    flops += stage1 + stage2 + stage3 + stage4
    return flops_to_string(flops), params_to_string(params)

def main():

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')).cuda()
    model.eval()

    # 수정된 부분: forward_dummy 체크 제거하고 직접 처리
    if hasattr(model.backbone, 'block1'):
        print('#### get transformer flops ####')
        with torch.no_grad():
            flops, params = get_tr_flops(model, input_shape)
    else:
        print('#### get CNN flops ####')
        # CNN의 경우 기본 forward_dummy 사용
        if hasattr(model, 'forward_dummy'):
            model.forward = model.forward_dummy
        flops, params = get_model_complexity_info(model, input_shape)

    split_line = '=' * 30
    print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
        split_line, input_shape, flops, params))
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()


'''
PYTHONPATH=$(pwd):$PYTHONPATH python tools/get_flops.py \
    work_dirs/student/fold1/segformer_224x224_b1/segformer_custom_224x224_fold1.py \
    --shape 256 320
'''