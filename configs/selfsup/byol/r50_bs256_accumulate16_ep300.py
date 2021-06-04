import copy
_base_ = '../../base.py'
# model settings
model = dict(
    type='BYOL',
    pretrained=None,
    base_momentum=0.99,
    pre_conv=True,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='SyncBN')),
    neck=dict(
        type='NonLinearNeckSimCLR',
        in_channels=2048,
        hid_channels=4096,
        out_channels=256,
        num_layers=2,
        sync_bn=True,
        with_bias=True,
        with_last_bn=False,
        with_avg_pool=True),
    head=dict(type='LatentPredictHead',
              size_average=True,
              predictor=dict(type='NonLinearNeckSimCLR',
                             in_channels=256, hid_channels=4096,
                             out_channels=256, num_layers=2, sync_bn=True,
                             with_bias=True, with_last_bn=False, with_avg_pool=False)))
# dataset settings
data_source_cfg = dict(
    type='ImageNet',
    memcached=True,
    mclient_path='/mnt/lustre/share/memcached_client')
#data_train_list = 'data/imagenet/meta/train.txt'
#data_train_root = 'data/imagenet/train'

# data_train_list = 'data/VOCdevkit/VOC2007/meta/train.txt'
# data_train_root = 'data/VOCdevkit/VOC2007/JPEGImages'
# data_test_list = 'data/VOCdevkit/VOC2007/meta/test.txt'
# data_test_root = 'data/VOCdevkit/VOC2007/JPEGImages'

# data_test_list = 'data/isic2017/meta/test.txt'
# data_test_root = 'data/isic2017/test'
# data_train_list = 'data/isic2017/meta/train.txt'
# data_train_root = 'data/isic2017/train'

data_test_list = 'data/x_ray_dataset/test_list.txt'
data_test_root = 'data/x_ray_dataset/images'
data_train_list = 'data/x_ray_dataset/train_val_list.txt'
data_train_root = 'data/x_ray_dataset/images'

dataset_type = 'BYOLDataset'
#ImageNet Normalization Config
#img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

#isic2017 Normalization Config
# img_norm_cfg = dict(mean=[0.670, 0.585, 0.589], std=[0.177, 0.194, 0.230])
#x-ray dataset config
img_norm_cfg = dict(mean=[0.5245, 0.5245, 0.5245], std =[0.2589, 0.2589, 0.2589])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224, interpolation=3),
    dict(type='RandomHorizontalFlip'),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='GaussianBlur',
                sigma_min=0.1,
                sigma_max=2.0)
        ],
        p=1.),
    dict(type='RandomAppliedTrans',
         transforms=[dict(type='Solarization')], p=0.),
]
# prefetch
prefetch = False
if not prefetch:
    train_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])
train_pipeline1 = copy.deepcopy(train_pipeline)
train_pipeline2 = copy.deepcopy(train_pipeline)
train_pipeline2[4]['p'] = 0.1 # gaussian blur
train_pipeline2[5]['p'] = 0.2 # solarization

data = dict(
    imgs_per_gpu=16,  # total 32*8(gpu)*16(interval)=4096
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline1=train_pipeline1,
        pipeline2=train_pipeline2,
        prefetch=prefetch,
    ),
    val = dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_test_list, root=data_test_root,
            **data_source_cfg),
        pipeline1=train_pipeline1,
        pipeline2=train_pipeline2,
        prefetch=prefetch,
    )
)
# additional hooks
update_interval = 16*16  # interval for accumulate gradient
custom_hooks = [
    dict(type='BYOLHook', end_momentum=1., update_interval=update_interval),    dict(
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
optimizer = dict(type='LARS', lr=4.8/8, weight_decay=0.000001, momentum=0.9,
                 paramwise_options={
                    '(bn|gn)(\d+)?.(weight|bias)': dict(weight_decay=0., lars_exclude=True),
                    'bias': dict(weight_decay=0., lars_exclude=True)}) #lr=4.8/8,
# apex
use_fp16 = False
optimizer_config = dict(update_interval=update_interval, use_fp16=use_fp16)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.,
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=0.0001, # cannot be 0
    warmup_by_epoch=True)
checkpoint_config = dict(interval=10)
# runtime settings
total_epochs = 200
