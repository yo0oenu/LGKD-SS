# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

import argparse
import json
import logging
from copy import deepcopy

from mmcv import Config, get_logger
from prettytable import PrettyTable

from mmseg.models import build_segmentor


def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


def count_parameters(model):
    table = PrettyTable(['Modules', 'Parameters'])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, human_format(param)])
        total_params += param
    # print(table)
    print(f'Total Trainable Params: {human_format(total_params)}')
    return total_params


# Run: python -m tools.param_count/ CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$(pwd):$PYTHONPATH python tools/get_param_count.py
if __name__ == '__main__':
    get_logger('mmseg', log_level=logging.ERROR)
    cfg = Config.fromfile('/home/yeonwoo3/DIFF/work_dirs/student/fold1/segformer_288x288_b0/segformer_custom_288x288_fold1.py')
    model = build_segmentor(cfg.model)
    print('Backbone:')
    count_parameters(model.backbone)
    print('Decode Head:')
    count_parameters(model.decode_head)
    print('Total:')
    count_parameters(model)
