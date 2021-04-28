_base_ = '../../../base.py'
# model settings
model = dict(
    type='Classification',
    pretrained=None,
    with_sobel=False,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN'),
        frozen_stages=4),
    head=dict(
        type='MultiClsClassificationHead', with_avg_pool=True, in_channels=2048,
        num_classes=20))
# dataset settings
data_source_cfg = dict(
    type='XRay',
    memcached=True,
    mclient_path='/mnt/lustre/share/memcached_client')

data_train_list = 'data/VOCdevkit/VOC2007/meta/trainval_labeled.txt'
data_train_root = 'data/VOCdevkit/VOC2007/JPEGImages'
data_test_list = 'data/VOCdevkit/VOC2007/meta/test_labeled.txt'
data_test_root = 'data/VOCdevkit/VOC2007/JPEGImages'
dataset_type = 'MultiLabelClassificationDataset'
img_norm_cfg = dict(mean=[0.4574, 0.4321, 0.3970],  std =[0.2777, 0.2769, 0.2846])

train_pipeline = [
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomHorizontalFlip'),
]
test_pipeline = [
    dict(type='Resize', size=256),
    dict(type='CenterCrop', size=224),
]
# prefetch
prefetch = False
if not prefetch:
    train_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])
    test_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])
data = dict(
    imgs_per_gpu=32,  # total 32*8=256, 8GPU linear cls
    workers_per_gpu=5,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline=train_pipeline,
        prefetch=prefetch),
    val=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_test_list, root=data_test_root, **data_source_cfg),
        pipeline=test_pipeline,
        prefetch=prefetch))
# additional hooks
custom_hooks = [
    dict(
        type='MultiValidateHook',
        dataset=data['val'],
        initial=True,
        interval=1,
        imgs_per_gpu=128,
        workers_per_gpu=4,
        prefetch=prefetch,
        img_norm_cfg=img_norm_cfg,
        )
]
# optimizer
optimizer = dict(type='SGD', lr=30./8, momentum=0.9, weight_decay=0.)
optimizer_config = dict(update_interval=8)
# learning policy
lr_config = dict(policy='step', step=[60, 80])
checkpoint_config = dict(interval=10)
# runtime settings
total_epochs = 100
