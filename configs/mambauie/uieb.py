img_scale = (256, 256)

pipeline = [
    dict(
        type='LoadImageFromFile',
        key='img',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(
        type='LoadImageFromFile',
        key='gt',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(
        type='Resize',
        keys=['img', 'gt'],
        scale=img_scale,
    ),
    dict(type='PackInputs')
]

data_root = '/home/allen/workspace/seamamba/data/uieb_t90/'

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='BasicImageDataset',
        metainfo=dict(dataset_type='UIEB', task_name='denoising'),
        data_root=data_root+'train',
        data_prefix=dict(img='raw-890', gt='reference-890'),
        pipeline=pipeline
    ),
)

val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicImageDataset',
        metainfo=dict(dataset_type='UIEB', task_name='denoising'),
        data_root=data_root+'valid',
        data_prefix=dict(img='raw-890', gt='reference-890'),
        pipeline=pipeline
    ),
)

test_dataloader = val_dataloader

evaluator = [
    dict(type='MAE', prefix='uie'),
    dict(type='MSE', prefix='uie'),
    dict(type='SSIM', convert_to='Y', prefix='uie'),
    dict(type='PSNR', convert_to='Y', prefix='uie'),
]

test_evaluator = val_evaluator = evaluator