dataset_type = 'camvidDataset'
#data_root = '//'  #384 * 512 (H * W)
data_root = '//'
# 데이터 전처리 파이프라인
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], 
    std=[58.395, 57.12, 57.375], 
    to_rgb=True
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(384, 288), ratio_range=(0.8, 1.5), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(288, 288)),
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(288, 288), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(384, 288),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size = (288, 384), pad_val=0, seg_pad_val=255),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=1,       
    workers_per_gpu=1,        
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/train',
        ann_dir='ann/train',
        pipeline=train_pipeline,
        img_suffix='.png',
        seg_map_suffix='_L.png'
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='ann/val',
        pipeline=test_pipeline,
        img_suffix='.png',
        seg_map_suffix='_L.png'
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='ann/val',
        pipeline=test_pipeline,
        img_suffix='.png',
        seg_map_suffix='_L.png'
    )
)
