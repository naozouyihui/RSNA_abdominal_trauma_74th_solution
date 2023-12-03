import monai.transforms as transforms

class CFG:
    kernel_type = 'timm3d_v2s_unet4b_128_128_128_dsv2_flip12_shift333p7_gd1p5_mixup1_lr1e3_20x50ep'
    load_kernel = None
    load_last = True
    n_blocks = 4
    n_folds = 5
    backbone = 'tf_efficientnetv2_s_in21ft1k'
    checkpoint = 'assets/public_pretrains/resnet18d_ra2-48a79e06.pth'
    pretrained = True
    image_sizes = [128, 128, 128]
    n_slice_per_c = 20
    n_ch = 5
    init_lr = 3e-3
    batch_size = 4
    drop_rate = 0.
    drop_path_rate = 0.
    loss_weights = [1, 1]
    p_mixup = 0.1

    csv_dir = '../../data/rsna-2023-abdominal-trauma-detection'
    data_dir = '../../data/rsna-2023-abdominal-trauma-detection'
    data_train_dir = '../../data/rsna-2023-abdominal-trauma-detection/train_images'
    use_amp = True
    num_workers = 2
    out_dim = 5

    n_epochs = 50

    transforms_train = transforms.Compose([
        transforms.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
        transforms.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=2),
        transforms.RandAffined(keys=["image", "mask"],
                               translate_range=[int(x * y) for x, y in zip(image_sizes, [0.3, 0.3, 0.3])],
                               padding_mode='zeros', prob=0.7),
        transforms.RandGridDistortiond(keys=("image", "mask"), prob=0.5, distort_limit=(-0.01, 0.01), mode="nearest"),
    ])

    transforms_valid = transforms.Compose([
    ])



