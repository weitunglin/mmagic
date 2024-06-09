"""
params 8.682 M, FLOPs 7.606 G

iter 33200 uie/MAE: 0.0584  uie/MSE: 0.0061  uie/SSIM: 0.9215  uie/PSNR: 23.5876  data_time: 0.1315  time: 0.6182
Start evalutaing T90
PSNR:  23.26439414457024
SSIM:  0.9084961349111366
MSE:  0.006392932273526553
UCIQE:  0.617646223794257
UIQM:  3.0484763800040464
NIQE:  5.581812865081653
URanker: 2.215339424771567
MUSIQ: 52.2026805029975
Start evalutaing C60
UCIQE:  0.5867716214313753
UIQM:  2.8687242172755725
NIQE:  6.46677250381111
URanker: 1.5107721330287556
MUSIQ: 48.025184567769365
Start evalutaing UCCS
UCIQE:  0.5618736255691569
UIQM:  3.0532465552891996
NIQE:  4.752065231873926
URanker: 1.3774680538102984
MUSIQ: 31.201950352986653

iter 37650 uie/MAE: 0.0587  uie/MSE: 0.0062  uie/SSIM: 0.9236  uie/PSNR: 23.5332
Start evalutaing T90
PSNR:  23.20493805018966
SSIM:  0.9101273139006643
MSE:  0.006456596438282841
UCIQE:  0.6183598856129596
UIQM:  3.005769978421612
NIQE:  5.57675808847364
URanker: 2.209049194306135
MUSIQ: 52.28248744540744
Start evalutaing C60
UCIQE:  0.5866200627564432
UIQM:  2.7904716684743276
NIQE:  6.511373322766993
URanker: 1.506754536430041
MUSIQ: 48.08399906158447
Start evalutaing UCCS
UCIQE:  0.5619259755651814
UIQM:  3.037282983130936
NIQE:  4.7152205458605705
URanker: 1.4017370195810994
MUSIQ: 31.422313912709555
"""

_base_ = [
    '../_base_/default_runtime.py',
    './uieb.py',
    './lwmamba.py'
]

ver = 'final'
experiment_name = f'lwmamba_uieb_{ver}'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

model = dict(
    type='BaseEditModel',
    generator=dict(
        type='MM_VSSM',
        depths=[1]*3,
        dims=128,
        pixel_branch=True,
        pixel_block_num=2,
        pixel_bi_scan=False,
        pixel_d_state=12,
        bpe=True,
        bi_scan=True,
        final_refine=False,
        merge_attn=True,
        pos_embed=True,
        last_skip=False,
        patch_size=4,
        mamba_up=True,
        unet_down=False,
        unet_up=False,
        conv_down=False,
        no_act_branch=True,
    ),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
    )
)

batch_size = 16
train_dataloader = dict(batch_size=batch_size)
val_dataloader = dict(batch_size=batch_size)

optim_wrapper = dict(
    dict(
        type='AmpOptimWrapper',
        optimizer=dict(type='AdamW', lr=0.0004, betas=(0.9, 0.999), weight_decay=0.5)))

max_epochs = 800
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-3, by_epoch=True, begin=0, end=20),
    dict(type='CosineAnnealingLR', by_epoch=True, begin=20, end=max_epochs, T_max=max_epochs, convert_to_iter_based=True)]

train_cfg = dict(by_epoch=True, max_epochs=max_epochs)

visualizer = dict(
    vis_backends=[dict(type='LocalVisBackend'), dict(type='WandbVisBackend', init_kwargs=dict(project='seamamba', name=ver))])

auto_scale_lr = dict(enable=False)
default_hooks = dict(logger=dict(interval=5))
custom_hooks = [dict(type='BasicVisualizationHook', interval=5)]

find_unused_parameter=False

# Test Scripts
# visualizer = dict(
#     type='ConcatImageVisualizer',
#     fn_key='img_path',
#     img_keys=['pred_img'],
#     bgr2rgb=True)


# custom_hooks = [
#     dict(type='BasicVisualizationHook', interval=1)]
