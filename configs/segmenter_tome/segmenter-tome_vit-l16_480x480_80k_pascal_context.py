_base_ = [
    # "./training_scheme.py",
    "../_base_/models/segmenter_vit-b16.py",
    "../_base_/datasets/pascal_context_meanstd0.5.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_80k.py",
]

model = dict(
    pretrained="pretrain/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz",
    backbone=dict(
        type="ToMeViT",
        img_size=(480, 480),
        patch_size=16,
        in_channels=3,
        embed_dims=1024,
        num_layers=24,
        num_heads=16,
        mlp_ratio=4,
        out_indices=(23),
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        with_cls_token=True,
        final_norm=True,
        norm_cfg=dict(type="LN", eps=1e-6),
        act_cfg=dict(type="GELU"),
        norm_eval=False,
        interpolate_mode="bicubic",
        tome_r=0.2,
    ),
    neck=dict(type="UseIndexSingleOutNeck", index=-1,),
    decode_head=dict(
        n_cls=60, n_layers=2, d_encoder=1024, n_heads=16, d_model=1024, d_ff=4 * 1024,
    ),
    test_cfg=dict(mode="slide", crop_size=(480, 480), stride=(320, 320)),
)

optimizer = dict(
    _delete_=True,
    type="AdamW",
    lr=0.00001,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            "pos_embed": dict(decay_mult=0.0),
            "cls_token": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
)

lr_config = dict(
    _delete_=True,
    policy="poly",
    warmup="linear",
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)