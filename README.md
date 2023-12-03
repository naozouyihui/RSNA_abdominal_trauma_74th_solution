# 74th Place Solution : RSNA 2023 Abdominal Trauma Detection Competition

[RSNA 2023 Abdominal Trauma Detection Competition](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/overview)

------

## Introduction

Team: [naocanzouyihui ](https://www.kaggle.com/naocanzouyihui)
Rank: `74/1123` Solo Winner - [LeaderBoard](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/leaderboard)

[Solution Summary](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/448208)

Hardware:

- CPU : `13th Gen Intel(R) Core(TM) i7-13700K 8 cores 24 threads`
- memory : `64GB`
- GPU : `NVIDIA GeForce RTX 3090 GPU with 32 GB VRAM`
- number of GPUs : `1`

OS/platform : `Windows 11`

------

## Solution Summary

This repository contains code to reproduce the 6th place solution, achieving private LB 0.40.

------

## Prerequisites

- Clone the repository

- Setup the environment:
  `pip install -r requirements.txt`

- Download the data in the `data` folder:

  - download [Competition data](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/data) and change folder name to `rsna-2023-abdominal-trauma-detection`

    ```
    data/dataset
    ├── image_level_labels.csv
    ├── sample_submission.csv
    ├── segmentations [206 entries exceeds filelimit, not opening dir]
    ├── test_dicom_tags.parquet
    ├── test_images
    ├── test_series_meta.csv
    ├── train.csv
    ├── train_dicom_tags.parquet
    ├── train_images
    └── train_series_meta.csv
    ```

## Run The pipeline

### Train model

`cd bash` : to run bash file

#### 1. Train 3D segmentation model

1. `cd src/Segmentation` : save transformed images and masks to train segmentation model to save time

2. `python train.py` : to train segmentation model

3. `python train.py` : to train segmentation model

   - segmentation model folder

     ```
     ├── results
     │   ├── segmentations
     │   │   └── test
     ```

     

   - data structure of segmented output will be :

     ```
     ├── segmentations           # by seg_save_cache.sh
     │   └── s_128
     ├── segmented               # by seg_output.sh
     │   ├── bowel
     │   ├── bowel_slices
     │   ├── left_kidney
     │   ├── liver
     │   ├── right_kidney
     │   └── spleen
     ```

     

#### 2. Train Organ and Bowel Model

1. `source train_organ_bowel.sh` : to train organ and bowel model
