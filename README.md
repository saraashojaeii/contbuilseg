# Building Segmentation Project

This repository contains a modularized implementation of building segmentation using four different deep learning models:

1. **SegFormer** - A transformer-based model for semantic segmentation
2. **UNet** - A convolutional neural network with encoder-decoder architecture that includes contour awareness
3. **DeepLabV3+** - An advanced semantic segmentation model with atrous spatial pyramid pooling (ASPP) and ResNet50 backbone
4. **HRNet** - High-Resolution Network that maintains high-resolution representations throughout the network
<!-- 
## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Models](#models)
  - [SegFormer](#segformer)
  - [UNet](#unet)
  - [DeepLabV3+](#deeplabv3)
  - [HRNet](#hrnet)
  - [Model Comparison](#model-comparison)
- [Training](#training)
  - [Training SegFormer](#training-segformer)
  - [Training UNet](#training-unet)
  - [Training DeepLabV3+](#training-deeplabv3)
  - [Training HRNet](#training-hrnet)
- [Evaluation](#evaluation)
- [Requirements](#requirements)
- [Citation](#citation)
- [License](#license) -->
<!-- 
## Features

✨ **Multiple State-of-the-Art Models** - Choose from 4 different architectures (SegFormer, UNet, DeepLabV3+, HRNet)  
🎯 **Contour-Aware Training** - Optional contour prediction for enhanced boundary detection  
📊 **Automatic Visualization** - Built-in prediction visualization and metric logging  
🔄 **Wandb Integration** - Track experiments with Weights & Biases  
💾 **Smart Checkpointing** - Automatic model saving with dataset-specific organization  
🛠️ **Modular Design** - Easy to extend and customize for your needs  
📈 **Comprehensive Metrics** - IoU, F1-score, precision, recall, and more -->

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/contbuilseg.git
cd contbuilseg

# Install dependencies
pip install -r requirements.txt

# Train a model (example with UNet)
python scripts/train_unet.py \
    --data_dir /path/to/datasets \
    --dataset_name "your_dataset" \
    --use_contours \
    --batch_size 8 \
    --epochs 100

# Evaluate the model
python scripts/evaluate_model.py \
    --model_type unet \
    --checkpoint ./checkpoints/your_dataset_models/your_dataset_latest.pth \
    --data_dir /path/to/datasets/your_dataset \
    --set test
```
<!-- 
## Repository Structure

```
building-segmentation/
├── data/                     # Data handling modules
│   ├── dataset.py            # Dataset classes for both models
│   └── data_utils.py         # Utilities for data processing
│
├── models/                   # Model implementations
│   ├── segformer.py          # SegFormer model wrapper
│   ├── unet.py               # UNet model implementation
│   ├── deeplabv3plus.py      # DeepLabV3+ model implementation
│   ├── hrnet.py              # HRNet model implementation
│   └── blocks.py             # Common building blocks
│
├── training/                 # Training modules
│   ├── train.py              # Base trainer class
│   ├── segformer_trainer.py  # SegFormer specific training
│   ├── unet_trainer.py       # UNet specific training
│   ├── deeplabv3plus_trainer.py  # DeepLabV3+ specific training
│   └── hrnet_trainer.py      # HRNet specific training
│
├── evaluation/               # Evaluation tools
│   ├── metrics.py            # Metrics computation (IoU, F1, etc.)
│   └── visualization.py      # Visualization utilities
│
├── utils/                    # Utility functions
│   └── losses.py             # Loss functions
│
├── scripts/                  # Executable scripts
│   ├── train_segformer.py    # Train SegFormer model
│   ├── train_unet.py         # Train UNet model
│   ├── train_deeplabv3plus.py # Train DeepLabV3+ model
│   ├── train_hrnet.py        # Train HRNet model
│   └── evaluate_model.py     # Evaluate trained models
│
├── notebooks/                # Jupyter notebooks (optional)
│   ├── segformer_exploration.ipynb
│   └── unet_exploration.ipynb
│
├── requirements.txt          # Package dependencies
└── README.md                 # This file
``` -->

## Dataset

The project is designed for building segmentation datasets with the following structure:

```
data/
├── train/             # Training images
├── train_labels/      # Training masks
├── val/               # Validation images
├── val_labels/        # Validation masks
├── test/              # Test images
└── test_labels/       # Test masks
```

For contour-aware segmentation with UNet, optional contour maps can be provided in folders like `train_contours/`.

### Generating Contour-based Masks

The repository includes a utility script `contour_distance_transform.py` for generating inverted saturated distance transform masks from binary segmentation labels. These contour-based masks can be used for contour-aware training or as additional supervision signals.

#### Usage
<!-- 
##### Single Image Processing

```bash
# Process a single binary mask
python contour_distance_transform.py --input mask.png --output contour_mask.png --saturation 50
``` -->

#### Batch Directory Processing

```bash
# Process all images in a directory
python contour_distance_transform.py --input_dir ./dataset/labels --output_dir ./dataset/contours --saturation 20

# Alternative syntax (backward compatible)
python contour_distance_transform.py --input ./input_directory --output ./output_directory --saturation 30
```

### Parameters

- `--input` / `-i`: Input binary mask image path or directory
- `--output` / `-o`: Output image path or directory
- `--input_dir`: Input directory containing binary masks (explicit)
- `--output_dir`: Output directory for results (explicit)
- `--saturation` / `-s`: Saturation value for distance transform (default: 50)
- `--demo`: Run with demo sample mask


The script automatically creates the output directory if it doesn't exist and preserves the original filenames. Progress is displayed during batch processing, showing which files are being processed and the final completion status.
<!-- 
## Models

### SegFormer

The SegFormer model uses the Hugging Face transformers implementation and is fine-tuned for building segmentation. It supports multi-class segmentation (background, building, boundary).

### UNet

The UNet model is a custom implementation with encoder and decoder blocks. It can output both mask and contour predictions for building segmentation tasks.

### DeepLabV3+

DeepLabV3+ is an advanced semantic segmentation architecture that uses:
- **ResNet50 backbone** with pretrained ImageNet weights
- **Atrous Spatial Pyramid Pooling (ASPP)** for multi-scale feature extraction
- **Encoder-decoder structure** with skip connections
- **Dual output heads** for both mask and contour prediction

### HRNet

High-Resolution Network (HRNet) maintains high-resolution representations throughout the network by:
- **Parallel multi-resolution branches** that process features at different scales
- **Repeated multi-scale fusion** to exchange information between branches
- **High-resolution feature maps** preserved from beginning to end
- **Dual output heads** for both mask and contour prediction -->
<!-- 
## Model Comparison

| Model | Strengths | Best For | Computational Cost |
|-------|-----------|----------|-------------------|
| **SegFormer** | Transformer-based, multi-class support | Large datasets, boundary detection | Medium-High |
| **UNet** | Simple, fast, effective | Quick experiments, baseline | Low |
| **DeepLabV3+** | Multi-scale features, pretrained backbone | General purpose, transfer learning | Medium |
| **HRNet** | High-resolution preservation, fine details | Precise boundaries, small buildings | High |

**Recommendations:**
- **Start with UNet** for quick prototyping and baseline results
- **Use DeepLabV3+** for best balance of performance and efficiency
- **Choose HRNet** when boundary precision is critical
- **Try SegFormer** for multi-class segmentation tasks -->

## Training Sample

<!-- ### Training SegFormer

```bash
python scripts/train_segformer.py \
    --data_dir /path/to/data \
    --model_name nvidia/mit-b0 \
    --num_labels 3 \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --epochs 100 \
    --save_every 10 \
    --output_dir ./outputs \
    --model_save_dir ./checkpoints/segformer
``` -->

### Training UNet

The UNet training script now supports automatic dataset organization and model/prediction saving with dataset-specific prefixes.

**Basic Training Command:**
```bash
python scripts/train_unet.py --data_dir /path/to/your/base/data/directory --dataset_name "massachusetts" --use_contours --batch_size 8 --learning_rate 1e-4 --epochs 100 --save_every 10 --mask_weight 0.7 --contour_weight 0.3 --output_dir ./outputs --model_save_dir ./checkpoints
```

**Multiple Dataset Examples:**
```bash
# Train on Massachusetts dataset
python scripts/train_unet.py --data_dir /path/to/datasets --dataset_name "massachusetts" --use_contours --epochs 100

# Train on WHU dataset
python scripts/train_unet.py --data_dir /path/to/datasets --dataset_name "whu" --use_contours --epochs 100

# Train on custom dataset
python scripts/train_unet.py --data_dir /path/to/datasets --dataset_name "custom_buildings" --use_contours --epochs 100
```

**Expected Directory Structure:**
The script expects your data to be organized as `data_dir/{dataset_name}/`:
```
/path/to/datasets/
├── massachusetts/
│   ├── train/
│   ├── train_labels/
│   ├── train_contours/
│   ├── val/
│   ├── val_labels/
│   ├── val_contours/
│   ├── test/
│   └── test_labels/
├── whu/
│   ├── train/
│   ├── train_labels/
│   └── ...
└── custom_buildings/
    ├── train/
    ├── train_labels/
    └── ...
```

**Output Organization:**
Each dataset will create organized output folders:
```
checkpoints/
├── massachusetts_models/
│   ├── massachusetts_epoch_001.pth
│   ├── massachusetts_epoch_002.pth
│   └── massachusetts_latest.pth
├── massachusetts_predictions/
│   ├── epoch_001/
│   │   ├── sample_01/
│   │   │   ├── input.png
│   │   │   ├── mask_gt.png
│   │   │   ├── mask_pred.png
│   │   │   └── combined_visualization.png
│   │   └── sample_02/...
│   └── epoch_002/...
└── whu_models/
    └── ...
```
<!-- 
**Features:**
- ✅ Automatic model checkpoint saving at each epoch
- ✅ Validation prediction visualization saving
- ✅ wandb integration for metrics and visualization logging
- ✅ Dataset-specific organization
- ✅ Support for both mask-only and mask+contour training -->
<!-- 
### Training DeepLabV3+

The DeepLabV3+ training script follows the same structure as UNet with dataset-specific organization.

**Basic Training Command:**
```bash
python scripts/train_deeplabv3plus.py \
    --data_dir /path/to/datasets \
    --dataset_name "massachusetts" \
    --use_contours \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --epochs 100 \
    --save_every 10 \
    --mask_weight 0.7 \
    --contour_weight 0.3 \
    --output_dir ./outputs \
    --model_save_dir ./checkpoints
```

**Key Features:**
- ResNet50 backbone with ImageNet pretrained weights
- ASPP module for multi-scale context
- Supports contour-aware training
- Same dataset organization as UNet

### Training HRNet

The HRNet training script also supports automatic dataset organization and contour-aware training.

**Basic Training Command:**
```bash
python scripts/train_hrnet.py \
    --data_dir /path/to/datasets \
    --dataset_name "massachusetts" \
    --use_contours \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --epochs 100 \
    --save_every 10 \
    --mask_weight 0.7 \
    --contour_weight 0.3 \
    --output_dir ./outputs \
    --model_save_dir ./checkpoints
```

**Key Features:**
- Maintains high-resolution representations throughout
- Multi-scale parallel processing
- Excellent for preserving fine details in building boundaries
- Supports contour-aware training -->

## Evaluation

To evaluate a trained model:

```bash
python scripts/evaluate_model.py \
    --model_type segformer \  # or unet, deeplabv3plus, hrnet
    --checkpoint /path/to/checkpoint \
    --data_dir /path/to/data \
    --set test \
    --output_dir ./evaluation_results \
    --save_predictions \
    --visualize \
    --num_vis_samples 5
```

**Supported model types:** `segformer`, `unet`, `deeplabv3plus`, `hrnet`

<!-- ## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
``` -->

<!-- ## Citation

If you use this code for your research, please consider citing:

```
@software{building_segmentation,
  author = {Your Name},
  title = {Building Segmentation with Multiple Deep Learning Architectures},
  year = {2024},
  url = {https://github.com/yourusername/building-segmentation}
}
``` -->

## License

This project is licensed under the MIT License - see the LICENSE file for details.
