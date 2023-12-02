import torch
import torch.nn as nn
import albumentations

class CFG:
    kernel_type = '1007_1bonev2_effv2s_224_30_6ch_augv2_mixupp5_drl3_rov1p2_bs2_lr23e6_eta23e6_75ep_kidney'
    load_kernel = None
    load_last = True

    n_folds = 5
    backbone = 'resnet50d'
    image_size = 224
    n_slice_per_c = 20
    in_chans = 6

    init_lr = 23e-5
    eta_min = 23e-6
    lw = [15, 1]
    batch_size = 8
    drop_rate = 0.
    drop_rate_last = 0.3
    drop_path_rate = 0.
    p_mixup = 0.5
    p_rand_order = 0.2
    p_rand_order_v1 = 0.2

    csv_dir = '../../data/rsna-2023-abdominal-trauma-detection'
    data_dir = '../../results/data/Segmentation/data/kidney'
    data_train_dir = '../../data/rsna-2023-abdominal-trauma-detection/train_images'
    classification_objects = 'kidney'
    use_amp = True
    num_workers = 4
    out_dim = 3

    device = torch.device('cuda')
    bce = nn.BCEWithLogitsLoss(reduction='none')
    n_epochs = 1

    transforms_train = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.Transpose(p=0.5),
        albumentations.RandomBrightness(limit=0.1, p=0.7),
        albumentations.ShiftScaleRotate(shift_limit=0.3, scale_limit=0.3, rotate_limit=45, border_mode=4, p=0.7),

        albumentations.OneOf([
            albumentations.MotionBlur(blur_limit=3),
            albumentations.MedianBlur(blur_limit=3),
            albumentations.GaussianBlur(blur_limit=(3, 5)),
            albumentations.GaussNoise(var_limit=(3.0, 9.0)),
        ], p=0.5),
        albumentations.OneOf([
            albumentations.OpticalDistortion(distort_limit=1.),
            albumentations.GridDistortion(num_steps=5, distort_limit=1.),
        ], p=0.5),

        albumentations.Cutout(max_h_size=int(image_size * 0.5), max_w_size=int(image_size * 0.5), num_holes=1, p=0.5),
    ])

    transforms_valid = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
    ])

