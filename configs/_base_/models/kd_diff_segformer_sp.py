
teacher_config = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='DIFF',
        style='pytorch',
        batch_size=4,
        init_cfg=dict(type='Pretrained', checkpoint=None)
    ),
    decode_head=dict(
        type='DAFormerHead',
        in_channels=[256, 512, 1024, 2048],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=11,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        decoder_params=dict(
            embed_dims=256,
            embed_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            embed_neck_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            fusion_cfg=dict(
                type='aspp',
                sep=True,
                dilations=(1, 6, 12, 18),
                pool=False,
                act_cfg=dict(type='ReLU'),
                norm_cfg=dict(type='BN', requires_grad=True)
            )
        ),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/mit_b0.pth',
    backbone=dict(type='mit_b0', style='pytorch'),
    decode_head=dict(
        type='SegFormerHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        channels=128,
        dropout_ratio=0.1,
        num_classes=11,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        decoder_params=dict(embed_dim=256, conv_kernel_size=1, use_gram_kd=False),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
    use_kd=True,
    teacher_config=teacher_config,
    kd_type='sp',
    kd_lamb=1.0,
    task_weight=1.0,
    freeze_teacher=True,
    diff_train=False
)

norm_cfg=dict(type='BN', requires_grad=True)
find_unused_parameters=True