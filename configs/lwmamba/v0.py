"""
params 19.122 M, FLOPs 7.210 G
"""

_base_ = [
    '../_base_/default_runtime.py',
    './uieb.py',
    './lwmamba.py'
]

ver = 'v0'
experiment_name = f'lwmamba_uieb_{ver}'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

model = dict(
    type='BaseEditModel',
    generator=dict(
        type='MM_VSSM',
        depths=[2,2,2,2],
        dims=96,
        patch_size=4,
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
        optimizer=dict(type='AdamW', lr=0.0002, betas=(0.9, 0.999), weight_decay=0.5)))

max_epochs = 400
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-3, by_epoch=True, begin=0, end=15),
    dict(type='CosineAnnealingLR', by_epoch=True, begin=15, T_max=max_epochs, convert_to_iter_based=True)]

train_cfg = dict(by_epoch=True, max_epochs=max_epochs)

visualizer = dict(
    vis_backends=[dict(type='LocalVisBackend'), dict(type='WandbVisBackend', init_kwargs=dict(project='seamamba', name=ver))])

auto_scale_lr = dict(enable=False)
default_hooks = dict(logger=dict(interval=10))
custom_hooks = [dict(type='BasicVisualizationHook', interval=6)]

find_unused_parameter = True