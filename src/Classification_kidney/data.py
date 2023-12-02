import os
import numpy as np
from config import CFG
import torch
import random
from torch.utils.data import Dataset


class CLSDataset(Dataset):
    def __init__(self, df, mode, transform):

        self.df = df.reset_index()
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]

        images = []

        for ind in list(range(CFG.n_slice_per_c)):
            filepath = os.path.join(CFG.data_dir, f'{row.patient_id}', f'{row.series_id}', f'{row.series_id}_{CFG.classification_objects}_{ind}.npy')
            image = np.load(filepath)
            image = image.astype(np.uint8)
            image = self.transform(image=image)['image']
            image = image.transpose(2, 0, 1).astype(np.float32) / 255.
            images.append(image)
        images = np.stack(images, 0)

        if self.mode != 'test':
            images = torch.tensor(images).float()
            label = []
            label.append(row.kidney_healthy)
            label.append(row.kidney_low)
            label.append(row.kidney_high)
            labels = torch.tensor(label * CFG.n_slice_per_c).float()
            if self.mode == 'train' and random.random() < CFG.p_rand_order_v1:
                indices = torch.randperm(images.size(0))
                images = images[indices]




            return images, labels
        else:
            return torch.tensor(images).float()
