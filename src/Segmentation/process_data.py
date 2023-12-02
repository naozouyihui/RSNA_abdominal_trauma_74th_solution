import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import cv2
import pydicom
import threading
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import segmentation_models_pytorch as smp
from conv3d_same import Conv3dSame
import torch
import timm
from timm.models.layers.conv2d_same import Conv2dSame
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from config import CFG


device = torch.device('cuda')
torch.backends.cudnn.benchmark = True


msk_size = CFG.image_sizes[0]
image_size_cls = 224


batch_size_seg = 1
num_workers = 2

# data_dir = '../../data/data/Segmentation/data'
# os.makedirs(data_dir, exist_ok=True)

data_liver_dir = '../../results/data/Segmentation/data/liver'
os.makedirs(data_liver_dir, exist_ok=True)

data_spleen_dir = '../../results/data/Segmentation/data/spleen'
os.makedirs(data_spleen_dir, exist_ok=True)

data_kidney_dir = '../../results/data/Segmentation/data/kidney'
os.makedirs(data_kidney_dir, exist_ok=True)

data_bowel_dir = '../../results/data/Segmentation/data/bowel'
os.makedirs(data_bowel_dir, exist_ok=True)

def get_subdirectories_with_paths(path, level=2):
    subdirectories = []
    directories = []
    for name in os.listdir(path):
        full_path = os.path.join(path, name)
        if os.path.isdir(full_path):
            if level == 1:
                subdirectories.append(name)
                directories.append(full_path)
            else:
                temp1, temp2 = get_subdirectories_with_paths(full_path, level - 1)
                subdirectories.extend(temp1)
                directories.extend(temp2)
    return subdirectories, directories


series_id, image_folder = get_subdirectories_with_paths(f'{CFG.data_train_dir}', 2)
df = pd.DataFrame({
    'series_id': series_id,
    'image_folder': image_folder,
})

# df = df.head(20)
#
# print(df)

# print(df['image_folder'].apply(lambda x: (x.split('\\')[-2])))

df['patient_id'] = df['image_folder'].apply(lambda x: int(x.split('\\')[-2].split("/")[-1]))

# print(len(df))
#
df = df.head(20)
#
# print(df)


def load_dicom(path):
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    data = cv2.resize(data, (CFG.image_sizes[0], CFG.image_sizes[1]), interpolation=cv2.INTER_AREA)
    return data


def load_dicom_line_par(path):
    t_paths = sorted(glob(os.path.join(path, "*")), key=lambda x: int(x.split('\\')[-1].split(".")[0]))

    n_scans = len(t_paths)
    #     print(n_scans)
    indices = np.quantile(list(range(n_scans)), np.linspace(0., 1., CFG.image_sizes[2])).round().astype(int)
    t_paths = [t_paths[i] for i in indices]

    images = []
    for filename in t_paths:
        images.append(load_dicom(filename))
    images = np.stack(images, -1)

    images = images - np.min(images)
    images = images / (np.max(images) + 1e-4)
    images = (images * 255).astype(np.uint8)

    return images


class SegTestDataset(Dataset):

    def __init__(self, df):
        self.df = df.reset_index()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image = load_dicom_line_par(row.image_folder)
        if image.ndim < 4:
            image = np.expand_dims(image, 0)
        image = image.astype(np.float32).repeat(3, 0)  # to 3ch
        image = image / 255.
        return torch.tensor(image).float()


dataset_seg = SegTestDataset(df)
loader_seg = torch.utils.data.DataLoader(dataset_seg, batch_size=batch_size_seg, shuffle=False, num_workers=num_workers)


def convert_3d(module):
    module_output = module
    if isinstance(module, torch.nn.BatchNorm2d):
        module_output = torch.nn.BatchNorm3d(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
        )
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig

    elif isinstance(module, Conv2dSame):
        module_output = Conv3dSame(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size[0],
            stride=module.stride[0],
            padding=module.padding[0],
            dilation=module.dilation[0],
            groups=module.groups,
            bias=module.bias is not None,
        )
        module_output.weight = torch.nn.Parameter(module.weight.unsqueeze(-1).repeat(1, 1, 1, 1, module.kernel_size[0]))

    elif isinstance(module, torch.nn.Conv2d):
        module_output = torch.nn.Conv3d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size[0],
            stride=module.stride[0],
            padding=module.padding[0],
            dilation=module.dilation[0],
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode
        )
        module_output.weight = torch.nn.Parameter(module.weight.unsqueeze(-1).repeat(1, 1, 1, 1, module.kernel_size[0]))

    elif isinstance(module, torch.nn.MaxPool2d):
        module_output = torch.nn.MaxPool3d(
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            ceil_mode=module.ceil_mode,
        )
    elif isinstance(module, torch.nn.AvgPool2d):
        module_output = torch.nn.AvgPool3d(
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            ceil_mode=module.ceil_mode,
        )

    for name, child in module.named_children():
        module_output.add_module(
            name, convert_3d(child)
        )
    del module

    return module_output



class TimmSegModel(nn.Module):
    def __init__(self, backbone, segtype='unet'):
        super(TimmSegModel, self).__init__()

        self.encoder = timm.create_model(
            backbone,
            in_chans=3,
            features_only=True,
            drop_rate=CFG.drop_rate,
            drop_path_rate=CFG.drop_path_rate,
            pretrained=CFG.pretrained
        )
        g = self.encoder(torch.rand(1, 3, 64, 64))
        encoder_channels = [1] + [_.shape[1] for _ in g]
        decoder_channels = [256, 128, 64, 32, 16]
        if segtype == 'unet':
            self.decoder = smp.unet.decoder.UnetDecoder(
                encoder_channels=encoder_channels[:CFG.n_blocks+1],
                decoder_channels=decoder_channels[:CFG.n_blocks],
                n_blocks=CFG.n_blocks,
            )

        self.segmentation_head = nn.Conv2d(decoder_channels[CFG.n_blocks-1], CFG.out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        global_features = [0] + self.encoder(x)[:CFG.n_blocks]
        seg_features = self.decoder(*global_features)
        seg_features = self.segmentation_head(seg_features)
        return seg_features

models_seg = []

kernel_type = 'timm3d_v2s_unet4b_128_128_128_dsv2_flip12_shift333p7_gd1p5_mixup1_lr1e3_20x50ep'
backbone = 'tf_efficientnetv2_s_in21ft1k'
model_dir_seg = '../../results/models/Segmentation/train'
n_blocks = 4
for fold in range(5):
    model = TimmSegModel(backbone)
    model = convert_3d(model)
    model = model.to(device)
    load_model_file = os.path.join(model_dir_seg, f'{kernel_type}_fold{fold}_best.pth')
    sd = torch.load(load_model_file)
    if 'model_state_dict' in sd.keys():
        sd = sd['model_state_dict']
    sd = {k[7:] if k.startswith('module.') else k: sd[k] for k in sd.keys()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    models_seg.append(model)

len(models_seg)


def load_bone(msk, cid, t_paths, cropped_images):
    n_scans = len(t_paths)
    bone = []
    try:
        msk_b = msk[cid] > 0.2
        msk_c = msk[cid] > 0.05

        x = np.where(msk_b.sum(1).sum(1) > 0)[0]
        y = np.where(msk_b.sum(0).sum(1) > 0)[0]
        z = np.where(msk_b.sum(0).sum(0) > 0)[0]

        if len(x) == 0 or len(y) == 0 or len(z) == 0:
            x = np.where(msk_c.sum(1).sum(1) > 0)[0]
            y = np.where(msk_c.sum(0).sum(1) > 0)[0]
            z = np.where(msk_c.sum(0).sum(0) > 0)[0]

        x1, x2 = max(0, x[0] - 1), min(msk.shape[1], x[-1] + 1)
        y1, y2 = max(0, y[0] - 1), min(msk.shape[2], y[-1] + 1)
        z1, z2 = max(0, z[0] - 1), min(msk.shape[3], z[-1] + 1)
        zz1, zz2 = int(z1 / msk_size * n_scans), int(z2 / msk_size * n_scans)

        inds = np.linspace(zz1, zz2-1, CFG.n_slice_per_c).astype(int)
        inds_ = np.linspace(z1, z2-1, CFG.n_slice_per_c).astype(int)
        for sid, (ind, ind_) in enumerate(zip(inds, inds_)):

            msk_this = msk[cid, :, :, ind_]

            images = []
            for i in range(-CFG.n_ch//2+1, CFG.n_ch//2+1):
                try:
                    dicom = pydicom.read_file(t_paths[ind+i])
                    images.append(dicom.pixel_array)
                except:
                    images.append(np.zeros((512, 512)))

            data = np.stack(images, -1)
            data = data - np.min(data)
            data = data / (np.max(data) + 1e-4)
            data = (data * 255).astype(np.uint8)
            msk_this = msk_this[x1:x2, y1:y2]
            xx1 = int(x1 / msk_size * data.shape[0])
            xx2 = int(x2 / msk_size * data.shape[0])
            yy1 = int(y1 / msk_size * data.shape[1])
            yy2 = int(y2 / msk_size * data.shape[1])
            data = data[xx1:xx2, yy1:yy2]
            data = np.stack([cv2.resize(data[:, :, i], (image_size_cls, image_size_cls), interpolation=cv2.INTER_LINEAR) for i in range(CFG.n_ch)], -1)
            msk_this = (msk_this * 255).astype(np.uint8)
            msk_this = cv2.resize(msk_this, (image_size_cls, image_size_cls), interpolation=cv2.INTER_LINEAR)

            data = np.concatenate([data, msk_this[:, :, np.newaxis]], -1)
            bone.append(torch.tensor(data))

    except:
        bone.clear()
        for sid in range(CFG.n_slice_per_c):
            bone.append(torch.ones((image_size_cls, image_size_cls, CFG.n_ch+1)).int())

    cropped_images[cid] = torch.stack(bone, 0)


def load_cropped_images(msk, image_folder):
    threads = [None] * 4
    cropped_images = [None] * 4

    t_paths = sorted(glob(os.path.join(image_folder, "*")), key=lambda x: int(x.split('\\')[-1].split(".")[0]))
    for cid in range(4):
        threads[cid] = threading.Thread(target=load_bone, args=(msk, cid, t_paths, cropped_images))
        threads[cid].start()
    for cid in range(4):
        threads[cid].join()

    return cropped_images


def main():
    bar = tqdm(loader_seg)
    with torch.no_grad():
        for batch_id, (images) in enumerate(bar):
            images = images.cuda()

            # SEG
            pred_masks = []
            for model in models_seg:
                pmask = model(images).sigmoid()
                pred_masks.append(pmask)
            pred_masks = torch.stack(pred_masks, 0).mean(0)
            proress_masks = torch.zeros((pred_masks.shape[0], 4, 128, 128, 128)).cuda()
            proress_masks[0][0] += pred_masks[0][0]
            proress_masks[0][1] += pred_masks[0][1]
            proress_masks[0][2] += torch.add(pred_masks[0][2], pred_masks[0][3])
            proress_masks[0][3] += pred_masks[0][4]
            proress_masks = proress_masks.cpu().numpy()

            # Build cls input
            proress_images = [None] * 4
            for i in range(proress_masks.shape[0]):
                row = df.iloc[batch_id * batch_size_seg + i]
                proress_images = load_cropped_images(proress_masks[i], row.image_folder) # (4, 30, 6, 224, 224)


            patient_directory_path = os.path.join(data_liver_dir, str(row.patient_id))
            os.makedirs(patient_directory_path, exist_ok=True)
            series_directory_path = os.path.join(patient_directory_path, str(row.series_id))
            os.makedirs(series_directory_path, exist_ok=True)
            for ind in range(CFG.n_slice_per_c):
                np.save(os.path.join(series_directory_path, f'{row.series_id}_liver_{ind}.npy'), proress_images[0][ind].cpu().numpy())

            patient_directory_path = os.path.join(data_spleen_dir, str(row.patient_id))
            os.makedirs(patient_directory_path, exist_ok=True)
            series_directory_path = os.path.join(patient_directory_path, str(row.series_id))
            os.makedirs(series_directory_path, exist_ok=True)
            for ind in range(CFG.n_slice_per_c):
                np.save(os.path.join(series_directory_path, f'{row.series_id}_spleen_{ind}.npy'), proress_images[1][ind].cpu().numpy())

            patient_directory_path = os.path.join(data_kidney_dir, str(row.patient_id))
            os.makedirs(patient_directory_path, exist_ok=True)
            series_directory_path = os.path.join(patient_directory_path, str(row.series_id))
            os.makedirs(series_directory_path, exist_ok=True)
            for ind in range(CFG.n_slice_per_c):
                np.save(os.path.join(series_directory_path, f'{row.series_id}_kidney_{ind}.npy'), proress_images[2][ind].cpu().numpy())

            patient_directory_path = os.path.join(data_bowel_dir, str(row.patient_id))
            os.makedirs(patient_directory_path, exist_ok=True)
            series_directory_path = os.path.join(patient_directory_path, str(row.series_id))
            os.makedirs(series_directory_path, exist_ok=True)
            for ind in range(CFG.n_slice_per_c):
                np.save(os.path.join(series_directory_path, f'{row.series_id}_bowel_{ind}.npy'), proress_images[3][ind].cpu().numpy())


if __name__ == '__main__':
    main()

