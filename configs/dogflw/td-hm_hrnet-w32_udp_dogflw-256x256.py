# Minimal single-model config for DogFLW (HRNet-W32 + UDP).
# Uses COCO-style keypoints produced by the converter above.

_base_ = []
default_scope = 'mmpose'
backend_args = None
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10, save_best='NME', rule='less'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='PoseVisualizationHook', enable=False),
)
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# ===== CONFIGURE THESE PATHS =====
# Override with: python tools/train.py CONFIG --cfg-options data_root=PATH ann_root=PATH
import os
DOGFLW_ROOT = os.getenv('DOGFLW_ROOT', os.path.expanduser('~/.cache/kagglehub/datasets/georgemartvel/dogflw/versions/1/DogFLW'))
ANN_ROOT    = os.getenv('ANN_ROOT', 'data/dogflw/annotations')

num_keypoints = 46
input_size    = (256, 256)
heatmap_size  = (64, 64)

# Minimal dataset meta. Leave flip_pairs empty unless you fill true L/R pairs.
metainfo = dict(
    dataset_name='dogflw',
    num_keypoints=46,
    keypoint_info={
        i: {'name': f'kp_{i}', 'id': i, 'color': [255, 0, 0], 'type': '', 'swap': ''}
        for i in range(46)
    },
    skeleton_info={},
    joint_weights=[1.0] * 46,
    sigmas=[0.025] * 46,
)

codec = dict(
    type='UDPHeatmap',                # UDP codec (encoder/decoder) for heatmaps
    input_size=input_size,
    heatmap_size=heatmap_size,
    sigma=2
)

train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale', padding=1.0),
    dict(type='RandomBBoxTransform', rotate_factor=30, scale_factor=[0.75, 1.25], shift_factor=0.0),
    dict(type='TopdownAffine', input_size=input_size),
    dict(type='RandomFlip', direction='horizontal', prob=0.0),  # change to 0.5 after you set flip_pairs
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale', padding=1.0),
    dict(type='TopdownAffine', input_size=input_size),
    dict(type='PackPoseInputs')
]

train_dataloader = dict(
    batch_size=16, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset', data_mode='topdown',
        data_root=f'{DOGFLW_ROOT}/train', data_prefix=dict(img='images/'),
        ann_file=f'{ANN_ROOT}/train.json', metainfo=metainfo, pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=16, num_workers=4, persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset', data_mode='topdown', test_mode=True,
        data_root=f'{DOGFLW_ROOT}/test', data_prefix=dict(img='images/'),
        ann_file=f'{ANN_ROOT}/val.json', metainfo=metainfo, pipeline=val_pipeline
    )
)
test_dataloader = val_dataloader

# Evaluate with NME normalized by inter-ocular distance (face-standard).
val_evaluator  = dict(type='NME', norm_mode='keypoint_distance', keypoint_indices=[0, 1])
test_evaluator = val_evaluator

optim_wrapper = dict(optimizer=dict(type='Adam', lr=3e-4))
param_scheduler = [
    dict(type='CosineAnnealingLR', T_max=210, by_epoch=True),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=210, val_interval=5)
val_cfg   = dict(type='ValLoop')
test_cfg  = dict(type='TestLoop')

# ---- Model ----
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(type='PoseDataPreprocessor'),
    backbone=dict(
        type='HRNet', in_channels=3,
        extra=dict(
            stage1=dict(num_modules=1, num_branches=1, block='BOTTLENECK', num_blocks=(4,),    num_channels=(64,)),
            stage2=dict(num_modules=1, num_branches=2, block='BASIC',      num_blocks=(4, 4),  num_channels=(32, 64)),
            stage3=dict(num_modules=4, num_branches=3, block='BASIC',      num_blocks=(4, 4, 4),  num_channels=(32, 64, 128)),
            stage4=dict(num_modules=3, num_branches=4, block='BASIC',      num_blocks=(4, 4, 4, 4), num_channels=(32, 64, 128, 256)),
        ),
        # optional: set a backbone checkpoint path here, or use --load-from (recommended)
        init_cfg=None
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=32, out_channels=num_keypoints,
        deconv_out_channels=None,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec
    ),
    test_cfg=dict(flip_test=False, shift_heatmap=False)
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer   = dict(type='PoseLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True, num_digits=6)
log_level = 'INFO'

