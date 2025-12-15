_base_ = [
    '../_base_/models/kd_diff_segformer.py',
    '../_base_/datasets/kd_kitti_1280x384_f2.py', 
    '../_base_/schedules/poly10warm.py',  
    '../_base_/default_runtime.py'
]

teacher_checkpoint = ''

model = dict(
    
    use_kd=True,        # KD
    kd_lamb=1.0,        # KD loss weight
    kd_max_v=10.0,       # KD loss max value
    task_weight=1.0,    # Task loss weight
    kd_temperature=4.0,  # KD temperature
    diff_train=False
)

# DIFF backbone 
norm_cfg = dict(type='BN', requires_grad=True)

optimizer = dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.1,
    paramwise_cfg=dict(
        custom_keys={
            'decode_head': dict(lr_mult=10.0),
            'pos_block': dict(decay_mult = 0.),
            'norm': dict(decay_mult = 0.)
        }
    )
)


lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False
)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(multi_scale=True)
)

optimizer_config = dict()



runner = dict(type='IterBasedRunner', max_iters=40000) 
evaluation = dict(interval=1000, metric='mIoU', save_best = 'mIoU')
checkpoint_config = dict(by_epoch=False, interval=40000)


work_dir = './'

# GPU 설정 추가
gpu_ids = range(0, 1)
