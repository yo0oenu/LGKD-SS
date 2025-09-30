_base_ = [
    '../../_base_/models/segformer.py',
    '../../_base_/datasets/kitti_640x192_f1.py',
    '../../_base_/default_runtime.py'
]

model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/mit_b0.pth',
    diff_train = False,
    backbone=dict(
        type='mit_b0',
        style='pytorch'),
    decode_head=dict(
        type='SegFormerHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=128,
        dropout_ratio=0.1,
        num_classes=11,
        align_corners=False,
        decoder_params=dict(embed_dim=256, conv_kernel_size=1),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))



# 훈련 설정 - 500 epoch (356 데이터, 배치 4)
runner = dict(type='IterBasedRunner', max_iters=40000)  
checkpoint_config = dict(by_epoch=False, interval=40000)
evaluation = dict(interval=1000, metric='mIoU', save_best='mIoU')  

# 옵티마이저 및 학습률
optimizer = dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg = dict(custom_keys = {
                     'pos_block': dict(decay_mult = 0.),
                     'norm': dict(decay_mult = 0.),
                     'head': dict(lr_mult = 10.)
                 }))


# 학습률 스케줄러
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False
)
optimizer_config = dict() 
# 데이터 설정 - batch_size 4
data = dict(
    samples_per_gpu=4,  # batch size
    workers_per_gpu=4
)

work_dir = './work_dirs/student_kitti/fold1/segformer_192x192_b0'
#CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$(pwd):$PYTHONPATH python tools/train.py configs/Student/kitti_640x192_p/fold1.py