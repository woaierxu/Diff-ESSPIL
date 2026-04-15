# RNCL:Refinement-based Noise-conditioned Learning for Semi-Supervised Medical Image Segmentation


## Introduction
Pytorch implementation of our paper "Refinement based Noise conditioned Learning for Semi Superior Medical Image Segmentation" published in Expert Systems with Applications 2026.


## Recommended environment:
Please run the following commands.

```
conda create -n rncl python=3.8
conda activate rncl
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Data preparation
You can get Left Atrium (LA) dataset from [UA-MT](https://github.com/yulequan/UA-MT/tree/master/data).

You can get Pancreas-CT  (Pancreas) dataset from [Google Drive](https://drive.google.com/file/d/1gipbEAnziAXIWXZZwqE7y2VnTgMrSvXu/view?usp=drive_link) or [Baidu Netdisk](https://pan.baidu.com/s/15SmPKaqeAtT2WV2FcDR2rQ?pwd=rncl).

You can get Kidney Tumor Segmentation 2019 (KiTS19) dataset from [Google Drive](https://drive.google.com/file/d/1FjzVDcEDNchrxqPs5wzoiOTLRHSpJ1HF/view?usp=drive_link) or [Baidu Netdisk](https://pan.baidu.com/s/15SmPKaqeAtT2WV2FcDR2rQ?pwd=rncl).

## Train

### LA Training
```
python train_la_rncl_lab10.py --root_path your_LA_dataset_root_path
```
If the number of labeled data needs to be adjusted, the **labelnum** parameter needs to be modified. For LA, **labelnum** of 8 represents 10% label data, and **labelnum** of 16 represents 20% label data.

### KiTS19 Training
```
python train_kits19_rncl_lab10.py --root_path your_KiTS19_dataset_root_path
```
If the number of labeled data needs to be adjusted, the **labelnum** parameter needs to be modified. For KiTS19, **labelnum** of 19 represents 10% label data, and **labelnum** of 38 represents 20% label data.

### Pancreas Training
pancreas/train_pancreas_rncl_lab10.py
**data_root** parameter needs to be modified.
If the number of labeled data needs to be adjusted, the **label_percent** parameter needs to be modified. For Pancreas, **label_percent** of 10 represents 10% label data, and **label_percent** of 20 represents 20% label data.
```
python train_pancreas_rncl_lab10.py
```

## Test

### LA Test
```
python test_la.py
```
### KiTS19 Training
```
python test_kits19.py
```

### Pancreas Training
```
python test_pancreas.py
```

## Citations
```

```







