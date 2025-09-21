dataset_type = 'MultiScaleKDDataset'
data_root = '/home/yeonwoo3/DATA/camvid/384x512_fold_p/fold1/'    #UniScale:  /home/yeonwoo3/DATA/camvid/384x288_fold_p/fold1/
data_root_val = '/home/yeonwoo3/DATA/camvid/384x288_fold_p/fold1/'

crop_size = (384, 384)  #teacher random_crop 사이즈(Input size)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], 
    std=[58.395, 57.12, 57.375], 
    to_rgb=True
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),  # GT 로딩 활성화
    dict(type='Resize', img_scale=(512, 384), ratio_range=(0.8, 1.5), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size),  # 384x384 정사각형 crop
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),  # 384x384 정사각형
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),  
]

# Student 전용 pipeline 
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
            dict(type='Pad', size=(288, 384), pad_val=0, seg_pad_val=255),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# Base dataset configs
base_dataset_cfg_train = dict(
    type='camvidDataset',
    data_root=data_root,
    img_dir='images/train',
    ann_dir='ann/train',
    img_suffix='.png',
    seg_map_suffix='_L.png',
    pipeline = []
)

base_dataset_cfg_val = dict(
    type='camvidDataset',
    data_root=data_root_val,
    img_dir='images/val',
    ann_dir='ann/val',
    img_suffix='.png',
    seg_map_suffix='_L.png',
    pipeline = []
)

base_dataset_cfg_test = dict(
    type='camvidDataset',
    data_root=data_root_val,
    img_dir='images/val',  # val과 test가 같은 폴더
    ann_dir='ann/val',
    img_suffix='.png',
    seg_map_suffix='_L.png'
)


data = dict(
    samples_per_gpu=4,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        base_dataset_cfg=base_dataset_cfg_train,
        train_pipeline=train_pipeline,
        test_pipeline=None,
        student_resolution=(288, 288),  # 384x384 → 288x288 (비율 유지)
        teacher_resolution=crop_size,  # Teacher 해상도
        multi_scale=False
    ),
    val=dict(
        type='camvidDataset',
        data_root=data_root_val,
        img_dir='images/val',
        ann_dir='ann/val',
        img_suffix='.png',
        seg_map_suffix='_L.png',
        pipeline=test_pipeline
    ),
    test=dict(
        type='camvidDataset',
        data_root=data_root_val,
        img_dir='images/val',
        ann_dir='ann/val',
        img_suffix='.png',
        seg_map_suffix='_L.png',
        pipeline=test_pipeline
    )
)