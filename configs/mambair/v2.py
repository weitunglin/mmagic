_base_ = [
    './seamamba.py'
]

ver = 'v2'
experiment_name = f'seamamba_uieb_{ver}'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

# model settings
model = dict(
    type='BaseEditModel',
    generator=dict(
        type='MM_VSSM',
        depths=[1]*4,
        dims=[48]*4,
        ver=ver,
    ))

optim_wrapper = dict(
    dict(
        type='OptimWrapper',
        optimizer=dict(type='AdamW', lr=0.00015, betas=(0.9, 0.999), weight_decay=0.5)))

max_epochs = 400
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=True, begin=0,
        end=15),
    dict(type='CosineAnnealingLR', by_epoch=True, T_max=max_epochs, convert_to_iter_based=True),]

train_cfg = dict(by_epoch=True, max_epochs=max_epochs)
