_base_ = [
    '../_base_/models/kd_diff_segformer_pr.py',
    '../_base_/datasets/kd_camvid_512x384_f2.py', 
    '../_base_/schedules/poly10warm.py',  
    '../_base_/default_runtime.py'
]


# Teacher 체크포인트 경로 
teacher_checkpoint = '/

model = dict(
    # KD 파라미터 오버라이드'
    use_kd=True,        # KD
    kd_type='pr',
    kd_lamb=0.01,        # KD loss weight
    task_weight=1.0,    # Task loss weight
    diff_train=False
)

# DIFF backbone 관련 설정
norm_cfg = dict(type='BN', requires_grad=True)

optimizer = dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'decode_head': dict(lr_mult=10.0),
            'pr_loss.proj': dict(lr_mult=10.0),
            'pos_block': dict(decay_mult = 0.),
            'norm': dict(decay_mult = 0.)
        }
    )
)

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

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(multi_scale=True)
)

optimizer_config = dict()

# 체크포인트 및 평가 설정

runner = dict(type='IterBasedRunner', max_iters=30000) 
evaluation = dict(interval=1000, metric='mIoU', save_best = 'mIoU')
checkpoint_config = dict(by_epoch=False, interval=30000)


# 작업 디렉토리
work_dir = './'

# GPU 설정 추가
gpu_ids = range(0, 1)
#PYTHONPATH=$(pwd):$PYTHONPATH python tools/train.py configs/KD/camvid_DIFF2Seg_512t384s_pr_fold2.py
