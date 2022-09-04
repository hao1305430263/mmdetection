_base_ = [
    "../_base_/schedules/schedule_1x.py",
    "../_base_/default_runtime.py",
]


# >>> data
dataset_type = "my_CustomDataset"
data_root = ""

train_pipeline = [
    dict(type="my_LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="Resize",  # 变化图像和其注释大小的数据增广的流程。
        img_scale=(768, 512),  # 图像的最大规模。
        keep_ratio=True,
    ),  # 是否保持图像的长宽比。
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Pad", size_divisor=32),  # 填充当前图像到指定大小的数据增广的流程。  # 填充图像可以被当前值整除。
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]

test_pipeline = [
    dict(type="my_LoadImageFromFile"),
    # dict(type="Resize", img_scale=(640, 640), keep_ratio=True),
    # dict(type="RandomFlip", flip_ratio=0.0),
    # # dict(type="ImageToTensor", keys=["img"]),
    # dict(type="DefaultFormatBundle"),
    # dict(
    #     type="Collect",
    #     keys=["img"],
    # ),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(
                type="Normalize",
                mean=[0, 0, 0, 0, 0, 0],
                std=[1, 1, 1, 1, 1, 1],
                to_rgb=False,
            ),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]


data = dict(
    samples_per_gpu=16,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        ann_file0="/home/haoyuanbo/Projects/MOD-haouanbo/datasets/KAIST_sanitized_annotations/annotations/train_visible_sanitized.json",
        ann_file1="/home/haoyuanbo/Projects/MOD-haouanbo/datasets/KAIST_sanitized_annotations/annotations/train_thermal_sanitized.json",
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_file0="/home/haoyuanbo/Projects/MOD-haouanbo/datasets/KAIST_sanitized_annotations/annotations/test_visable_kaist_annotation.json",
        ann_file1="/home/haoyuanbo/Projects/MOD-haouanbo/datasets/KAIST_sanitized_annotations/annotations/test_thermal_kaist_annotation.json",
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file0="/home/haoyuanbo/Projects/MOD-haouanbo/datasets/KAIST_sanitized_annotations/annotations/test_visable_kaist_annotation.json",
        ann_file1="/home/haoyuanbo/Projects/MOD-haouanbo/datasets/KAIST_sanitized_annotations/annotations/test_thermal_kaist_annotation.json",
        pipeline=test_pipeline,
    ),
)

custom_imports = dict(
    imports=["mmdet.models.backbones.my_resnet"], allow_failed_imports=False
)

model = dict(
    type="ATSS",
    backbone=dict(
        type="my_ResNet",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=True,
        style="pytorch",
        # init_cfg0=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
        # init_cfg1=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
    ),
    neck=[
        dict(
            type="FPN",
            in_channels=[512, 1024, 2048, 4096],
            out_channels=512,
            start_level=1,
            add_extra_convs="on_output",
            num_outs=5,
        ),
        dict(type="DyHead", in_channels=512, out_channels=256, num_blocks=3),
    ],
    bbox_head=dict(
        type="ATSSHead",
        num_classes=5,
        in_channels=256,
        stacked_convs=0,
        feat_channels=256,
        anchor_generator=dict(
            type="AnchorGenerator",
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128],
        ),
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder",
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2],
        ),
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0
        ),
        loss_bbox=dict(type="GIoULoss", loss_weight=2.0),
        loss_centerness=dict(
            type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0
        ),
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type="ATSSAssigner", topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type="nms", iou_threshold=0.6),
        max_per_img=100,
    ),
)

# >>> optimizer
evaluation = dict(interval=1, metric="bbox")
optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy="step", warmup="linear", warmup_iters=500, warmup_ratio=0.001, step=[8, 11]
)
runner = dict(type="EpochBasedRunner", max_epochs=120)
checkpoint_config = dict(interval=5)
log_config = dict(interval=50, hooks=[dict(type="TextLoggerHook")])
custom_hooks = [dict(type="NumClassCheckHook")]
dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1), ("val", 1)]
opencv_num_threads = 0
mp_start_method = "fork"
auto_scale_lr = dict(enable=False, base_batch_size=16)
