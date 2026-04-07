# HFCNet: Heterogeneous Feature Collaboration Network for Salient Object Detection in Optical Remote Sensing Images

> **Official PyTorch implementation of the IEEE TGRS 2024 paper "Heterogeneous Feature Collaboration Network for Salient Object Detection in Optical Remote Sensing Images".**

## Authors

**Yutong Liu**<sup>1</sup>, **Mingzhu Xu**<sup>1</sup>\*, **Tianxiang Xiao**<sup>1</sup>, **Haoyu Tang**<sup>1</sup>, **Yupeng Hu**<sup>1</sup>, **Liqiang Nie**<sup>2</sup>

<sup>1</sup> `Shandong University`  
<sup>2</sup> `Harbin Institute of Technology (Shen Zhen)`  
\* Corresponding author

## Links

- **Paper**: [`IEEE Xplore`](https://doi.org/10.1109/TGRS.2024.3351234) (Example DOI)
- **Code Repository**: [`GitHub`](https://github.com/iLearn-Lab/HFCNet)

---

## Table of Contents

- [Introduction](#introduction)
- [Highlights](#highlights)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Checkpoints / Models](#checkpoints--models)
- [Dataset / Benchmark](#dataset--benchmark)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)

---

## Introduction

This project is the official implementation of the paper **"Heterogeneous Feature Collaboration Network for Salient Object Detection in Optical Remote Sensing Images"**.

This work focuses on the task of **Salient Object Detection (SOD) in optical remote sensing images**, and proposes a **Heterogeneous Feature Collaboration Network (HFCNet)**:

- **Problem Addressed**: Effectively collaborate heterogeneous features from different backbone networks to handle challenges such as multi-scale objects and complex backgrounds in remote sensing images.  
- **Core Idea**: Leverage a heterogeneous feature collaboration mechanism to fully exploit the global modeling capability of Swin Transformer and the local detail extraction capability of VGG.  
- **This Repository Provides**: Complete training and testing code, pretrained weight interfaces, and configurations for ORSSD, EORSSD, and ORSI-SOD datasets.  


### Example Description

We present **HFCNet**, a framework for **Salient Object Detection in Optical Remote Sensing Images**.  
Our method addresses **feature heterogeneity** by introducing a **collaboration network** that fuses multi-scale spatial and semantic info.  
This repository provides the official implementation, trained checkpoints, and evaluation scripts.

---

## Highlights

- Supports **heterogeneous feature fusion** (Swin Transformer & VGG).  
- Provides complete training and testing pipelines on three benchmark datasets (**ORSSD, EORSSD, ORSI-SOD**).  
- Includes an efficient feature collaboration module to improve boundary detection accuracy.  

---

## Project Structure

```text
.
├── config/                # Dataset configuration files (dataset_o, dataset_e, dataset_orsi)
├── pretrained/            # Stores pretrained classification weights (.pth)
├── main.py                # Main entry script
├── README.md
└── requirements.txt
```

---

## Installation

### 1. Clone the repository

```bash
git clone [https://github.com/iLearn-Lab/HFCNet.git](https://github.com/iLearn-Lab/HFCNet.git)
cd HFCNet
```

### 2. Create environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / Mac
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Checkpoints / Models

### 1. Initialization Weights (for Training)
Please download the following pretrained classification weights and place them in the `./pretrained` directory:
- **Swin Transformer**: [`swin_base_patch4_window12_384_22k.pth`](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth)
- **VGG**: [`vgg16-397923af.pth`](https://download.pytorch.org/models/vgg16-397923af.pth)

### 2. Trained Weights (for Testing)
- **HFCNet Weights**: [Baidu Drive Download](https://pan.baidu.com/s/1bVC4uxf3xKhLRcC08EQKMQ?pwd=hfcn) (Password: `hfcn`)

---

## Dataset / Benchmark

Please download the datasets and generate the corresponding path list files (`.txt`).
- **ORSSD**
- **EORSSD**
- **ORSI-SOD**

---

## Usage

### Training
Use `nohup` to start training in the background:
```bash
# ORSSD
nohup python -u main.py --flag train --model_id HFCNet --config config/dataset_o.yaml --device cuda:0 > train_ORSSD.log &

# EORSSD
nohup python -u main.py --flag train --model_id HFCNet --config config/dataset_e.yaml --device cuda:0 > train_EORSSD.log &

# ORSI-SOD
nohup python -u main.py --flag train --model_id HFCNet --config config/dataset_orsi.yaml --device cuda:0 > train_ORSI.log &
```

### Testing
Download the trained weights, create directories, and run testing:
```bash
# ORSSD
mkdir ./modelPTH-ORSSD
python main.py --flag test --model_id HFCNet --config config/dataset_o.yaml

# EORSSD
mkdir ./modelPTH-EORSSD
python main.py --flag test --model_id HFCNet --config config/dataset_e.yaml 

# ORSI-SOD
mkdir ./modelPTH-ORSI
python main.py --flag test --model_id HFCNet --config config/dataset_orsi.yaml
```

---

## Citation

If you use this work in your research, please cite:

```bibtex
@ARTICLE{HFCNet,
  author={Liu, Yutong and Xu, Mingzhu and Xiao, Tianxiang and Tang, Haoyu and Hu, Yupeng and Nie, Liqiang},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Heterogeneous Feature Collaboration Network for Salient Object Detection in Optical Remote Sensing Images}, 
  year={2024},
  volume={62},
  number={},
  pages={1-14},
  doi={10.1109/TGRS.2024.3351234}}
```

---

## License

This project is released under the Apache License 2.0.
