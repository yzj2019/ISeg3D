'''full config, 尝试binary focal loss'''
from pointcept.datasets.preprocessing.scannet.meta_data.scannet200_constants import CLASS_LABELS_200

_base_ = ["../_base_/interseg_default_runtime.py"]

# misc custom setting
batch_size = 12  # bs: total bs in all gpus
mix_prob = 0.8
empty_cache = True
enable_amp = False      # 混合精度?
num_classes=200
ignore_label=-1

# Clicker
train_clicker_cfg = dict(
    type="Clicker3D",
    num_classes=num_classes,
    ignore_label=ignore_label
)

test_clicker_cfg = dict(
    type="Clicker3D",
    num_classes=num_classes,
    ignore_label=ignore_label
)

# model settings
model = dict(
    type="Agile3dSegmentor",
    pcd_backbone=dict(
        type="DefaultSegmentorPretrained",
        backbone=dict(
            type="SpUNet-v1m1",
            in_channels=9,
            num_classes=num_classes,
            channels=(32, 64, 128, 256, 256, 128, 96, 96),
            layers=(2, 3, 4, 6, 2, 2, 2, 2)
        ),
        ckpt_path="/data/shared_workspace/yuzijian/ws/iseg3d/exp/scannet200/semseg-spunet-v1m1-0-base/model/model_best.pth",
        criteria=[
            dict(type="CrossEntropyLoss",
                 loss_weight=1.0,
                 ignore_index=ignore_label)
        ]
    ),
    mask_decoder=dict(
        type="Agile3dMaskDecoder",
        transformer_cfg=dict(
            type="Agile3dDecoderTransformer",
            transformer_block_cfg=dict(
                type="Agile3dDecoderBlock",
                embedding_dim=96,
                num_heads=6,
                mlp_dim=1024,
                attention_downsample_rate=1,
                activation="relu",
                skip_first_layer_pe=False,
                attn_drop = 0.1, layer_drop = 0.1
            ),
            depth=3,
            enable_final_attn=True
        ),
        mask_head_hidden_dims=[96],
        cls_head_hidden_dims=[96],
        num_classes=num_classes,
        norm_fn_name='bn1d'
    ),
    loss=dict(
        type="Agile3DLoss",
        clicks_mask_loss_cfg = dict(
            type="BinaryFocalLoss",
            gamma=2.0,
            alpha=0.5,
            logits=False,
            reduce=True,
            loss_weight=1.0
        ),
        clicks_cls_loss_cfg = dict(
            type="CrossEntropyLoss",
            loss_weight=1.0,
            #  reduction='none',
            reduction='mean',
            ignore_index=ignore_label
        ),
        mask_loss_cfg = dict(
            type="CrossEntropyLoss",
            loss_weight=0.0,
            #  reduction='none',
            reduction='mean',
            ignore_index=ignore_label
        )
    ),
    semantic=True, max_train_iter=3, ignore_label=ignore_label, mask_threshold=0.3
)

# hook
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="ClicksMaker", clicker_cfg=train_clicker_cfg, pcd_only=True),
    dict(type="InteractiveSemSegEvaluator", clicker_cfg=train_clicker_cfg, pcd_only=True),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False)
]

# Tester
test = dict(
    type="InteractiveSemSegTester",
    pcd_only = False,
    progressive_mode = True
)

# scheduler settings
epoch = 300
eval_epoch = 300
optimizer = dict(type="AdamW", lr=0.0001, weight_decay=0.0001)
scheduler = dict(type="OneCycleLR",
                 max_lr=optimizer["lr"],
                 pct_start=0.001,
                 anneal_strategy="cos",
                 div_factor=10.0,
                 final_div_factor=1000.0)

# dataset settings
dataset_type = "ScanNetFusionDataset"
img_data_root = "data/scannet_img"
pcd_data_root = "data/scannet"

data = dict(
    num_classes=num_classes,
    ignore_index=ignore_label,
    names=CLASS_LABELS_200,
    train=dict(
        type=dataset_type,
        image_cfg = dict(
            type="ScanNetImageDataset",
            split="train",
            data_root=img_data_root,
            test_mode=False
        ),
        pointcloud_cfg = dict(
            type="ScanNet200Dataset",
            split="train",
            data_root=pcd_data_root,
            transform=[
                dict(type="CenterShift", apply_z=True),
                dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
                # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
                dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
                dict(type="RandomRotate", angle=[-1/64, 1/64], axis="x", p=0.5),
                dict(type="RandomRotate", angle=[-1/64, 1/64], axis="y", p=0.5),
                dict(type="RandomScale", scale=[0.9, 1.1]),
                # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
                dict(type="RandomFlip", p=0.5),
                dict(type="RandomJitter", sigma=0.005, clip=0.02),
                dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
                dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
                dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
                dict(type="ChromaticJitter", p=0.95, std=0.05),
                # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
                # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
                dict(type="GridSample", grid_size=0.02, hash_type="fnv", mode="train", return_discrete_coord=True),
                dict(type="SphereCrop", point_max=100000, mode="random"),
                dict(type="CenterShift", apply_z=False),
                dict(type="NormalizeColor"),
                # dict(type="ShufflePoint"),
                dict(type="ToTensor"),
                dict(type="Collect", keys=("coord", "discrete_coord", "segment", "color", "scene_id"), feat_keys=("coord", "color", "normal"))
            ],
            test_mode=False
        )
    ),

    val=dict(
        type=dataset_type,
        image_cfg = dict(
            type="ScanNetImageDataset",
            split="val",
            data_root=img_data_root,
            test_mode=False
        ),
        pointcloud_cfg = dict(
            type="ScanNet200Dataset",
            split="val",
            data_root=pcd_data_root,
            test_mode=False,
            transform=[
                dict(type="CenterShift", apply_z=True),
                dict(type="GridSample", grid_size=0.02, hash_type="fnv", mode="train", return_discrete_coord=True),
                # dict(type="SphereCrop", point_max=1000000, mode="center"),
                dict(type="CenterShift", apply_z=False),
                dict(type="NormalizeColor"),
                dict(type="ToTensor"),
                dict(type="Collect", keys=("coord", "discrete_coord", "segment", "color", "scene_id"), feat_keys=("coord", "color", "normal"))
            ]
        )
    ),

    test=dict(
        type=dataset_type,
        image_cfg = dict(
            type="ScanNetImageDataset",
            split="val",
            data_root=img_data_root,
            test_mode=True
        ),
        pointcloud_cfg = dict(
            type="ScanNet200Dataset",
            split="val",
            data_root=pcd_data_root,
            transform=[
                dict(type="NormalizeColor"),
            ],
            test_mode=True,
            test_cfg=dict(
                voxelize=dict(type="GridSample",
                              grid_size=0.02,
                              hash_type="fnv",
                              mode="test",
                              return_discrete_coord=True,
                              keys=("coord", "color", "normal")
                              ),
                crop=None,
                post_transform=[
                    dict(type="CenterShift", apply_z=False),
                    dict(type="ToTensor"),
                    dict(type="Collect", keys=("coord", "discrete_coord", "index", "color", "scene_id"), feat_keys=("coord", "color", "normal"))
                ],
                aug_transform=[
                    [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)],
                    [dict(type="RandomRotateTargetAngle", angle=[1/2], axis="z", center=[0, 0, 0], p=1)],
                    [dict(type="RandomRotateTargetAngle", angle=[1], axis="z", center=[0, 0, 0], p=1)],
                    [dict(type="RandomRotateTargetAngle", angle=[3/2], axis="z", center=[0, 0, 0], p=1)]
                ]
            )
        )
    ),
)
