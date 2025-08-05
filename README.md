# Building Segmentation Project

This repository contains a modularized implementation of building segmentation using two different deep learning models:

1. **SegFormer** - A transformer-based model for semantic segmentation
2. **UNet** - A convolutional neural network with encoder-decoder architecture that includes contour awareness

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
│   └── blocks.py             # Common building blocks
│
├── training/                 # Training modules
│   ├── train.py              # Base trainer class
│   ├── segformer_trainer.py  # SegFormer specific training
│   └── unet_trainer.py       # UNet specific training
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
│   └── evaluate_model.py     # Evaluate trained models
│
├── notebooks/                # Jupyter notebooks (optional)
│   ├── segformer_exploration.ipynb
│   └── unet_exploration.ipynb
│
├── requirements.txt          # Package dependencies
└── README.md                 # This file
```

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

##### Single Image Processing

```bash
# Process a single binary mask
python contour_distance_transform.py --input mask.png --output contour_mask.png --saturation 50
```

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

## Models

### SegFormer

The SegFormer model uses the Hugging Face transformers implementation and is fine-tuned for building segmentation. It supports multi-class segmentation (background, building, boundary).

### UNet

The UNet model is a custom implementation with encoder and decoder blocks. It can output both mask and contour predictions for building segmentation tasks.

## Training

### Training SegFormer

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
```

### Training UNet

```bash
python scripts/train_unet.py --data_dir /root/home/pvc/building_segmentation_datasets/massachusetts/ --use_contours --contour_dir /root/home/pvc/building_segmentation_datasets/massachusetts/ --batch_size 8 --learning_rate 1e-4 --epochs 100 --save_every 10 --mask_weight 0.7 --contour_weight 0.3 --output_dir ./outputs --model_save_dir ./checkpoints/unet
```

## Evaluation

To evaluate a trained model:

```bash
python scripts/evaluate_model.py \
    --model_type segformer \
    --checkpoint /path/to/checkpoint \
    --data_dir /path/to/data \
    --set test \
    --output_dir ./evaluation_results \
    --save_predictions \
    --visualize \
    --num_vis_samples 5
```

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Citation

If you use this code for your research, please consider citing:

```
@software{building_segmentation,
  author = {Your Name},
  title = {Building Segmentation with SegFormer and UNet},
  year = {2024},
  url = {https://github.com/yourusername/building-segmentation}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
