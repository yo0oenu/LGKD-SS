# configs/custom/camvid_diff_exp.py
_base_ = [
    '../../_base_/models/daformer_sepaspp_diff.py',        
    '../../_base_/default_runtime.py',                      
    '../../_base_/schedules/adamw.py',                      
    '../../_base_/schedules/poly10warm.py',
    '../../_base_/datasets/kitti_1280x384_f2.py',
]
model = dict(
    diff_train = True,
    use_kd = False,
    pretrained=None,
    backbone=dict(type='DIFF'),         
    decode_head=dict(num_classes=11),   
)

optimizer = dict(
    type='AdamW',
    lr=6e-5,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'head': dict(lr_mult=10.0),
            'backbone.diff_model.aggregation_network.mixing_weights_stride': dict(lr_mult=10.0)
        }
    )
)
optimizer_config = dict()  

runner = dict(type='IterBasedRunner', max_iters=30000)
evaluation = dict(interval=1000, metric='mIoU', save_best='mIoU')
checkpoint_config = dict(by_epoch=False, interval=30000)

work_dir = './work_dirs/1280x384_kitty_fold2'     #work_dir 수정정

#PYTHONPATH=$(pwd):$PYTHONPATH python tools/train.py configs/512x384_p/fold2_F.py
#CUDA_VISIBLE_DEVICES=1