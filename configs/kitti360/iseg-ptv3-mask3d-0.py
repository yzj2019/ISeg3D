"""
mask3d mask decoder, train from pretrained semantic spunet
load from mask3d-8 ckpt, for single object segmentation
"""

from pointcept.datasets.preprocessing.kitti360.helpers.labels import trainId2label

KITTI360_CLASS_LABELS = {i: trainId2label[i].name for i in trainId2label.keys()}

_base_ = ["../_base_/interseg_default_runtime.py"]

# misc custom setting
batch_size = 12  # bs: total bs in all gpus, 12 on A6000, 8 on 3090
num_worker = 4
mix_prob = 0.8
empty_cache = False
enable_amp = False  # 混合精度?
num_classes = 20
semantic_ignore_label = -1
# segment_background_labels=[0,1]
segment_background_labels = []
instance_ignore_label = -1

# Clicker
train_clicker_cfg = dict(
    type="Clicker3D",
    num_classes=num_classes,
    semantic_ignore_label=semantic_ignore_label,
    instance_ignore_label=instance_ignore_label,
)

test_clicker_cfg = dict(
    type="Clicker3D",
    num_classes=num_classes,
    semantic_ignore_label=semantic_ignore_label,
    instance_ignore_label=instance_ignore_label,
    sample_num=200,
)

# model settings
model = dict(
    type="Mask3dSegmentor",
    pcd_backbone=dict(
        type="DefaultSegmentorFPN",
        backbone=dict(
            type="SpUNet-v1m1",
            in_channels=6,
            num_classes=0,  # 为了不要 backbone.final 这个线性层
            channels=(32, 64, 128, 256, 256, 128, 96, 96),
            layers=(2, 3, 4, 6, 2, 2, 2, 2),
            out_fpn=True,
        ),
        ckpt_path=None,
        criteria=[
            dict(
                type="CrossEntropyLoss",
                loss_weight=1.0,
                ignore_index=semantic_ignore_label,
            )
        ],
    ),
    mask_decoder=dict(
        type="Mask3dMaskDecoder",
        transformer_block_cfg=dict(
            type="Mask3dDecoderBlock",
            embedding_dim=128,
            num_heads=8,
            mlp_dim=1024,
            activation="relu",
            skip_first_layer_pe=False,
            attn_drop=0.0,
            layer_drop=0.0,
            norm_first=False,
        ),
        feature_maps_dims=[256, 256, 128, 96, 96],
        clicks_embedding_dim=96,
        depth=3,
        mask_head_hidden_dims=[128],
        cls_head_hidden_dims=[128],
        num_classes=num_classes,
        with_attn_mask=[False, True, True, True, True],
        enable_final_block=False,
    ),
    loss=dict(
        type="Agile3DLoss",
        clicks_mask_loss_cfg=[
            dict(
                type="BinaryCrossEntropyLoss",
                reduction="mean",
                logits=False,
                loss_weight=5.0,
            ),
            dict(
                type="BinaryDiceLoss",
                exponent=1,
                reduction="mean",
                logits=False,
                loss_weight=2.0,
            ),
        ],
        clicks_cls_loss_cfg=[
            dict(
                type="CrossEntropyLoss",
                loss_weight=2.0,
                #  reduction='none',
                reduction="mean",
                ignore_index=semantic_ignore_label,
            )
        ],
        mask_loss_cfg=[
            dict(
                type="CrossEntropyLoss",
                loss_weight=0.0,
                #  reduction='none',
                reduction="mean",
                ignore_index=semantic_ignore_label,
            )
        ],
    ),
    semantic=True,
    clicks_from_instance=True,
    max_train_iter=3,
    mask_threshold=0.5,
)

# hook
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="ClicksMaker", clicker_cfg=train_clicker_cfg, pcd_only=True),
    dict(
        type="InteractiveSemSegEvaluator", clicker_cfg=train_clicker_cfg, pcd_only=True
    ),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

# Tester
test = dict(
    type="InteractiveInsSegTester",
    pcd_only=True,
    clicker_cfg=test_clicker_cfg,
    semantic_ignore_label=[semantic_ignore_label],
    instance_ignore_label=instance_ignore_label,
    progressive_mode=False,
    random_clicks=False,
    topk_per_scene=100,
    cost_class=1.0,
    cost_focal=1.0,
    cost_dice=1.0,
    iou_thrs=[0.8, 0.85, 0.9],
    noc_thrs=[1, 2, 3, 5, 10, 15],
)

# scheduler settings
epoch = 50
eval_epoch = 50
optimizer = dict(type="AdamW", lr=0.0001, weight_decay=0.0001)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.3,
    anneal_strategy="cos",
    div_factor=25.0,
    final_div_factor=100.0,
)

# dataset settings
dataset_type = "ScanNetFusionDataset"
img_data_root = "data/scannet_img"
pcd_data_root = "data/scannet"

data = dict(
    num_classes=num_classes,
    ignore_index=semantic_ignore_label,
    names=KITTI360_CLASS_LABELS,
    train=dict(
        type=dataset_type,
        image_cfg=dict(
            type="ScanNetImageDataset",
            split="train",
            data_root=img_data_root,
            test_mode=False,
        ),
        pointcloud_cfg=dict(
            type="ScanNetDataset",
            split="train",
            data_root=pcd_data_root,
            transform=[
                dict(type="CenterShift", apply_z=True),
                dict(
                    type="RandomDropout",
                    dropout_ratio=0.2,
                    dropout_application_ratio=0.2,
                ),
                # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
                dict(
                    type="RandomRotate",
                    angle=[-1, 1],
                    axis="z",
                    center=[0, 0, 0],
                    p=0.5,
                ),
                dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
                dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
                dict(type="RandomScale", scale=[0.9, 1.1]),
                # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
                dict(type="RandomFlip", p=0.5),
                dict(type="RandomJitter", sigma=0.005, clip=0.02),
                dict(
                    type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]
                ),
                dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
                dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
                dict(type="ChromaticJitter", p=0.95, std=0.05),
                # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
                # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
                dict(
                    type="GridSample",
                    grid_size=0.02,
                    hash_type="fnv",
                    mode="train",
                    return_discrete_coord=True,
                    keys=("coord", "color", "normal", "segment", "instance"),
                ),
                dict(type="SphereCrop", point_max=100000, mode="random"),
                dict(type="CenterShift", apply_z=False),
                dict(type="NormalizeColor"),
                # dict(type="ShufflePoint"),
                dict(
                    type="InstanceParser",
                    segment_ignore_index=[semantic_ignore_label],
                    instance_ignore_index=instance_ignore_label,
                ),  # 将特定的semantic label在instance中忽略
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=(
                        "coord",
                        "discrete_coord",
                        "segment",
                        "instance",
                        "color",
                        "scene_id",
                    ),
                    feat_keys=("color", "normal"),
                ),
            ],
            test_mode=False,
        ),
    ),
    val=dict(
        type=dataset_type,
        image_cfg=dict(
            type="ScanNetImageDataset",
            split="val",
            data_root=img_data_root,
            test_mode=False,
        ),
        pointcloud_cfg=dict(
            type="ScanNetDataset",
            split="val",
            data_root=pcd_data_root,
            test_mode=False,
            transform=[
                dict(type="CenterShift", apply_z=True),
                dict(
                    type="GridSample",
                    grid_size=0.02,
                    hash_type="fnv",
                    mode="train",
                    return_discrete_coord=True,
                    keys=("coord", "color", "normal", "segment", "instance"),
                ),
                # dict(type="SphereCrop", point_max=1000000, mode="center"),
                dict(type="CenterShift", apply_z=False),
                dict(type="NormalizeColor"),
                dict(
                    type="InstanceParser",
                    segment_ignore_index=[semantic_ignore_label],
                    instance_ignore_index=instance_ignore_label,
                ),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=(
                        "coord",
                        "discrete_coord",
                        "segment",
                        "instance",
                        "instance_centroid",
                        "color",
                        "scene_id",
                    ),
                    feat_keys=("color", "normal"),
                ),
            ],
        ),
    ),
    test=dict(
        type=dataset_type,
        image_cfg=dict(
            type="ScanNetImageDataset",
            split="val",
            data_root=img_data_root,
            test_mode=True,
        ),
        pointcloud_cfg=dict(
            type="ScanNetDataset",
            split="val",
            data_root=pcd_data_root,
            transform=[
                dict(type="CenterShift", apply_z=True),
                dict(type="NormalizeColor"),
                dict(
                    type="InstanceParser",
                    segment_ignore_index=[semantic_ignore_label],
                    instance_ignore_index=instance_ignore_label,
                ),
            ],
            test_mode=True,
            test_cfg=dict(
                voxelize=dict(
                    type="GridSample",
                    grid_size=0.02,
                    hash_type="fnv",
                    mode="test",
                    return_discrete_coord=True,
                    keys=("coord", "color", "normal"),
                ),
                crop=None,
                post_transform=[
                    dict(type="CenterShift", apply_z=False),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=("coord", "discrete_coord", "index", "color", "scene_id"),
                        feat_keys=("color", "normal"),
                    ),
                ],
                aug_transform=[
                    # [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)],
                    # [dict(type="RandomRotateTargetAngle", angle=[1/2], axis="z", center=[0, 0, 0], p=1)],
                    # [dict(type="RandomRotateTargetAngle", angle=[1], axis="z", center=[0, 0, 0], p=1)],
                    # [dict(type="RandomRotateTargetAngle", angle=[3/2], axis="z", center=[0, 0, 0], p=1)],
                    # rescale后需要重新选clicks, 开销有点大
                    # [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1),
                    #  dict(type="RandomScale", scale=[0.95, 0.95])],
                    # [dict(type="RandomRotateTargetAngle", angle=[1 / 2], axis="z", center=[0, 0, 0], p=1),
                    #  dict(type="RandomScale", scale=[0.95, 0.95])],
                    # [dict(type="RandomRotateTargetAngle", angle=[1], axis="z", center=[0, 0, 0], p=1),
                    #  dict(type="RandomScale", scale=[0.95, 0.95])],
                    # [dict(type="RandomRotateTargetAngle", angle=[3 / 2], axis="z", center=[0, 0, 0], p=1),
                    #  dict(type="RandomScale", scale=[0.95, 0.95])],
                    # [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1),
                    #  dict(type="RandomScale", scale=[1.05, 1.05])],
                    # [dict(type="RandomRotateTargetAngle", angle=[1 / 2], axis="z", center=[0, 0, 0], p=1),
                    #  dict(type="RandomScale", scale=[1.05, 1.05])],
                    # [dict(type="RandomRotateTargetAngle", angle=[1], axis="z", center=[0, 0, 0], p=1),
                    #  dict(type="RandomScale", scale=[1.05, 1.05])],
                    # [dict(type="RandomRotateTargetAngle", angle=[3 / 2], axis="z", center=[0, 0, 0], p=1),
                    #  dict(type="RandomScale", scale=[1.05, 1.05])],
                    [dict(type="RandomFlip", p=1)]
                ],
            ),
        ),
    ),
)
