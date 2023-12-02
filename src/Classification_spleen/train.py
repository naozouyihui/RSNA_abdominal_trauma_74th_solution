import os
import gc
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from config import CFG
from model import TimmModelType2
from data import CLSDataset
from loss import criterion
from sklearn.model_selection import KFold
import torch
import torch.optim as optim
import torch.cuda.amp as amp

DEBUG = False
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True


log_dir = '../../results/logs/spleen/train'
model_dir = '../../results/models/spleen/train'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

df_train = pd.read_csv(os.path.join(CFG.csv_dir, 'train.csv'))


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

series_id, image_folder = get_subdirectories_with_paths(f'{CFG.data_dir}/', 2)
df_segmentation = pd.DataFrame({
    'series_id': series_id,
    'image_folder': image_folder,
})

df_segmentation['patient_id'] = df_segmentation['image_folder'].apply(lambda x: int(x.split('\\')[-2].split("/")[-1]))

# 将 df_train['patient_id'] 列转换为整数类型
df_train['patient_id'] = df_train['patient_id'].astype(int)

# 将 df_segmentation['patient_id'] 列转换为整数类型
df_segmentation['patient_id'] = df_segmentation['patient_id'].astype(int)



df = df_segmentation.merge(df_train, on='patient_id', how='left')

df = df.drop(df[df['series_id'] == '58351'].index)

#df.to_csv('submission.csv', index=False)

print(len(df))

# df = df.head(20)

df = df.reset_index()

kf = KFold(5)
df['fold'] = -1
for fold, (train_idx, valid_idx) in enumerate(kf.split(df, df)):
    df.loc[valid_idx, 'fold'] = fold


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
    for images, targets in bar:
        optimizer.zero_grad()
        images = images.cuda()
        targets = targets.cuda()
        do_mixup = False
        if random.random() < CFG.p_mixup:
            do_mixup = True
            images, targets, targets_mix, lam = mixup(images, targets)

        with amp.autocast():
            logits = model(images)
            loss = criterion(logits, targets)
            if do_mixup:
                loss11 = criterion(logits, targets_mix)
                loss = loss * lam  + loss11 * (1 - lam)
        train_loss.append(loss.item())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bar.set_description(f'smth:{np.mean(train_loss[-30:]):.4f}')

    return np.mean(train_loss)


def valid_func(model, loader_valid):
    model.eval()
    valid_loss = []
    gts = []
    outputs = []
    bar = tqdm(loader_valid)
    with torch.no_grad():
        for images, targets in bar:
            images = images.cuda()
            targets = targets.cuda()

            logits = model(images)
            loss = criterion(logits, targets)

            gts.append(targets.cpu())
            outputs.append(logits.cpu())
            valid_loss.append(loss.item())

            bar.set_description(f'smth:{np.mean(valid_loss[-30:]):.4f}')

    outputs = torch.cat(outputs)
    gts = torch.cat(gts)
    valid_loss = criterion(outputs, gts).item()

    return valid_loss


def run(fold):

    log_file = os.path.join(log_dir, f'{CFG.kernel_type}.txt')
    model_file = os.path.join(model_dir, f'{CFG.kernel_type}_fold{fold}_best.pth')

    train_ = df[df['fold'] != fold].reset_index(drop=True)
    valid_ = df[df['fold'] == fold].reset_index(drop=True)
    dataset_train = CLSDataset(train_, 'train', transform=CFG.transforms_train)
    dataset_valid = CLSDataset(valid_, 'valid', transform=CFG.transforms_valid)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers, drop_last=True)
    loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)

    model = TimmModelType2(CFG.backbone, pretrained=True)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=CFG.init_lr)
    scaler = torch.cuda.amp.GradScaler() if CFG.use_amp else None
    metric_best = np.inf

    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, CFG.n_epochs, eta_min=CFG.eta_min)

    print(len(dataset_train), len(dataset_valid))

    for epoch in range(1, CFG.n_epochs+1):
        scheduler_cosine.step(epoch-1)

        print(time.ctime(), 'Epoch:', epoch)

        train_loss = train_func(model, loader_train, optimizer, scaler)
        valid_loss = valid_func(model, loader_valid)
        metric = valid_loss

        content = time.ctime() + ' ' + f'Fold {fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid loss: {valid_loss:.5f}, metric: {metric:.6f}.'
        print(content)
        with open(log_file, 'a') as appender:
            appender.write(content + '\n')

        if metric < metric_best:
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
