import sys
sys.path.append('./')
import os
import gc
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold
from config import CFG
from model import TimmSegModel, convert_3d
from data import SEGDataset
from loss import bce_dice, multilabel_dice_score

import torch
import torch.optim as optim
import torch.cuda.amp as amp
from torch.utils.data import DataLoader

DEBUG = False
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True


log_dir = '../../results/logs/Segmentation/train'
model_dir = '../../results/models/Segmentation/train'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)


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


series_id, image_folder = get_subdirectories_with_paths(f'{CFG.data_dir}/train_images/', 2)
mask_files = os.listdir(f'{CFG.data_dir}/segmentations')
df_mask = pd.DataFrame({
    'mask_file': mask_files,
})
df_train = pd.DataFrame({
    'series_id': series_id,
    'image_folder': image_folder,
})
df_mask['series_id'] = df_mask['mask_file'].apply(lambda x: x[:-4])
df_mask['mask_file'] = df_mask['mask_file'].apply(lambda x: os.path.join(f'{CFG.data_dir}', 'segmentations', x))
df = df_train.merge(df_mask, on='series_id', how='left')
df['mask_file'].fillna('', inplace=True)

df_seg = df.query('mask_file != ""').reset_index(drop=True)


kf = KFold(5)
df_seg['fold'] = -1
for fold, (train_idx, valid_idx) in enumerate(kf.split(df_seg, df_seg)):
    df_seg.loc[valid_idx, 'fold'] = fold

criterion = bce_dice


def mixup(input, truth, clip=[0, 1]):
    indices = torch.randperm(input.size(0))
    shuffled_input = input[indices]
    shuffled_labels = truth[indices]

    lam = np.random.uniform(clip[0], clip[1])
    input = input * lam + shuffled_input * (1 - lam)
    return input, truth, shuffled_labels, lam


def train_func(model, loader_train, optimizer, scaler=None):
    model.train()
    train_loss = []
    bar = tqdm(loader_train)
    for images, gt_masks in bar:
        optimizer.zero_grad()
        images = images.cuda()
        gt_masks = gt_masks.cuda()

        do_mixup = False
        if random.random() < CFG.p_mixup:
            do_mixup = True
            images, gt_masks, gt_masks_sfl, lam = mixup(images, gt_masks)

        with amp.autocast():
            logits = model(images)
            loss = criterion(logits, gt_masks)
            if do_mixup:
                loss2 = criterion(logits, gt_masks_sfl)
                loss = loss * lam + loss2 * (1 - lam)

        train_loss.append(loss.item())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bar.set_description(f'smth:{np.mean(train_loss[-30:]):.4f}')

    return np.mean(train_loss)


def valid_func(model, loader_valid):
    model.eval()
    valid_loss = []
    ths = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    batch_metrics = [[]] * 7
    bar = tqdm(loader_valid)
    with torch.no_grad():
        for images, gt_masks in bar:
            images = images.cuda()
            gt_masks = gt_masks.cuda()

            logits = model(images)
            loss = criterion(logits, gt_masks)
            valid_loss.append(loss.item())
            for thi, th in enumerate(ths):
                for i in range(logits.shape[0]):
                    tmp = multilabel_dice_score(
                        y_pred=logits[i].sigmoid().cpu(),
                        y_true=gt_masks[i].cpu(),
                        threshold=0.5,
                    )
                    batch_metrics[thi].extend(tmp)
            bar.set_description(f'smth:{np.mean(valid_loss[-30:]):.4f}')

    metrics = [np.mean(this_metric) for this_metric in batch_metrics]
    print('best th:', ths[np.argmax(metrics)], 'best dc:', np.max(metrics))

    return np.mean(valid_loss), np.max(metrics)


def run(fold):
    log_file = os.path.join(log_dir, f'{CFG.kernel_type}.txt')
    model_file = os.path.join(model_dir, f'{CFG.kernel_type}_fold{fold}_best.pth')

    train_ = df_seg[df_seg['fold'] != fold].reset_index(drop=True)
    valid_ = df_seg[df_seg['fold'] == fold].reset_index(drop=True)
    dataset_train = SEGDataset(train_, 'train', transform=CFG.transforms_train)
    dataset_valid = SEGDataset(valid_, 'valid', transform=CFG.transforms_valid)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=CFG.batch_size, shuffle=True,
                                               num_workers=CFG.num_workers)
    loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=CFG.batch_size, shuffle=False,
                                               num_workers=CFG.num_workers)

    model = TimmSegModel(CFG.backbone)
    model = convert_3d(model)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=CFG.init_lr)
    scaler = torch.cuda.amp.GradScaler()
    metric_best = 0.

    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, CFG.n_epochs)

    print(len(dataset_train), len(dataset_valid))

    for epoch in range(1, CFG.n_epochs + 1):
        scheduler_cosine.step(epoch - 1)

        print(time.ctime(), 'Epoch:', epoch)

        train_loss = train_func(model, loader_train, optimizer, scaler)
        valid_loss, metric = valid_func(model, loader_valid)

        content = time.ctime() + ' ' + f'Fold {fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid loss: {valid_loss:.5f}, metric: {metric:.6f}.'
        print(content)
        with open(log_file, 'a') as appender:
            appender.write(content + '\n')

        if metric > metric_best:
            print(f'metric_best ({metric_best:.6f} --> {metric:.6f}). Saving model ...')
            torch.save(model.state_dict(), model_file)
            metric_best = metric

        # Save Last
        if not DEBUG:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict() if scaler else None,
                    'score_best': metric_best,
                },
                model_file.replace('_best', '_last')
            )

    del model
    torch.cuda.empty_cache()
    gc.collect()


def main():
    run(0)
    run(1)
    run(2)
    run(3)
    run(4)


if __name__ == '__main__':
    main()
