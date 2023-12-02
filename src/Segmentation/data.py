import os
import cv2
import pydicom
import numpy as np
from glob import glob
import nibabel as nib
from config import CFG
import torch
from torch.utils.data import Dataset
from monai.transforms import Resize


revert_list = [
]

R = Resize(CFG.image_sizes)

def load_dicom(path):
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    data = cv2.resize(data, (CFG.image_sizes[0], CFG.image_sizes[1]), interpolation=cv2.INTER_LINEAR)
    return data


def load_dicom_line_par(path):
    t_paths = sorted(glob(os.path.join(path, "*")),
                     key=lambda x: int(x.split('\\')[-1].split(".")[0]))

    n_scans = len(t_paths)
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


def load_sample(row, has_mask=True):
    image = load_dicom_line_par(row.image_folder)
    if image.ndim < 4:
        image = np.expand_dims(image, 0).repeat(3, 0)  # to 3ch

    if has_mask:
        mask_org = nib.load(row.mask_file).get_fdata()
        shape = mask_org.shape
        mask_org = mask_org.transpose(1, 0, 2)[::-1, :, ::-1]  # (d, w, h)
        mask = np.zeros((5, shape[1], shape[0], shape[2]))
        for cid in range(5):
            mask[cid] = (mask_org == (cid + 1))
        mask = mask.astype(np.uint8) * 255
        mask = R(mask).numpy()

        return image, mask
    else:
        return image


class SEGDataset(Dataset):
    def __init__(self, df, mode, transform):
        self.df = df.reset_index()
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image, mask = load_sample(row, has_mask=True)

#        if row.patient_id in revert_list:
#            mask = mask[:, :, :, ::-1]

        res = self.transform({'image': image, 'mask': mask})
        image = res['image'] / 255.
        mask = res['mask']
        mask = (mask > 127).astype(np.float32)
        image, mask = torch.tensor(image).float(), torch.tensor(mask).float()

        return image, mask