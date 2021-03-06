_base_ = '../../base.py'
# model settings
model = dict(
    type='MOCO',
    pretrained=None,
    queue_len=65536,
    feat_dim=128,
    momentum=0.999,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN')),
    neck=dict(
        type='LinearNeck',
        in_channels=2048,
        out_channels=128,
        with_avg_pool=True),
    head=dict(type='ContrastiveHead', temperature=0.07))
# dataset settings
data_source_cfg = dict(
    type='ImageNet',
    memcached=True,
    mclient_path='/mnt/lustre/share/memcached_client')
#data_train_list = 'data/imagenet/meta/train.txt'
#data_train_root = 'data/imagenet/train'
# img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

#data_train_list = 'data/VOCdevkit/VOC2007/meta/train.txt'
#data_train_root = 'data/VOCdevkit/VOC2007/JPEGImages'
#data_test_list = 'data/VOCdevkit/VOC2007/meta/test.txt'
#data_test_root = 'data/VOCdevkit/VOC2007/JPEGImages'

data_test_list = 'data/isic2017/meta/test.txt'
data_test_root = 'data/isic2017/test'
data_train_list = 'data/isic2017/meta/train.txt'
data_train_root = 'data/isic2017/train'
#isic2017 Normalization Config
img_norm_cfg = dict(mean=[0.670, 0.585, 0.589], std=[0.177, 0.194, 0.230])

# data_test_list = 'data/x_ray_dataset/test_list.txt'
# data_test_root = 'data/x_ray_dataset/images'
# data_train_list = 'data/x_ray_dataset/train_val_list.txt'
# data_train_root = 'data/x_ray_dataset/images'

dataset_type = 'ContrastiveDataset'

train_pipeline = [
    dict(type='RandomResizedCrop', size=224, scale=(0.2, 1.)),
    dict(type='RandomGrayscale', p=0.2),
    dict(
        type='ColorJitter',
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.4),
    dict(type='RandomHorizontalFlip'),
]
# prefetch
prefetch = False
if not prefetch:
    train_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])

data = dict(
    imgs_per_gpu=32,  # total 32*8=256
    workers_per_gpu=4,
    drop_last=True,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline=train_pipeline,
        prefetch=prefetch,
    ),    val = dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_test_list, root=data_test_root,
            **data_source_cfg),
        pipeline=train_pipeline,
        prefetch=prefetch,
    )
)

# additional hooks
custom_hooks = [dict(
        type='NewValidateHook',
        dataset=data['val'],
        initial=True,
        interval=1,
        imgs_per_gpu=32,
        workers_per_gpu=5,
        prefetch=prefetch,
        img_norm_cfg=img_norm_cfg)
]
# optimizer
optimizer = dict(type='SGD', lr=0.03/8, weight_decay=0.0001, momentum=0.9)
optimizer_config = dict(update_interval=8)
# learning policy
lr_config = dict(policy='step', step=[120, 160])
checkpoint_config = dict(interval=10)
# runtime settings
total_epochs = 100
