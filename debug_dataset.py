#!/usr/bin/env python
"""Debug script to check dataset values"""

import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from data.dataset import CustomDataset
from torchvision import transforms
import glob

def find_files_with_extensions(directory, extensions):
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, f'*.{ext}')))
    return sorted(files)

# Setup
data_dir = "/root/home/pvc/building_segmetation_datasets/"
dataset_name = "massachusetts"
image_extensions = ['png', 'jpg', 'jpeg', 'tif', 'tiff']

dataset_dir = os.path.join(data_dir, dataset_name)
train_img_paths = find_files_with_extensions(os.path.join(dataset_dir, 'train'), image_extensions)
train_mask_paths = find_files_with_extensions(os.path.join(dataset_dir, 'train_labels'), image_extensions)
train_contour_paths = None
train_contour_dir = os.path.join(dataset_dir, 'train_contours')
if os.path.isdir(train_contour_dir):
    train_contour_paths = find_files_with_extensions(train_contour_dir, image_extensions)

transform = transforms.Compose([transforms.ToTensor()])

# Create dataset
train_dataset = CustomDataset(
    train_img_paths, 
    train_mask_paths, 
    train_contour_paths, 
    transform=transform
)

print(f"Dataset size: {len(train_dataset)}")
print(f"Has contours: {train_contour_paths is not None}")

# Check first few samples
for i in range(min(3, len(train_dataset))):
    print(f"\n=== Sample {i} ===")
    if train_contour_paths:
        image, mask, contour = train_dataset[i]
        print(f"Image shape: {image.shape}, dtype: {image.dtype}")
        print(f"Image min: {image.min():.4f}, max: {image.max():.4f}")
        print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
        print(f"Mask min: {mask.min():.4f}, max: {mask.max():.4f}")
        print(f"Mask unique values: {mask.unique()[:10]}")  # First 10 unique values
        print(f"Contour shape: {contour.shape}, dtype: {contour.dtype}")
        print(f"Contour min: {contour.min():.4f}, max: {contour.max():.4f}")
        print(f"Contour unique values: {contour.unique()[:10]}")
    else:
        image, mask = train_dataset[i]
        print(f"Image shape: {image.shape}, dtype: {image.dtype}")
        print(f"Image min: {image.min():.4f}, max: {image.max():.4f}")
        print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
        print(f"Mask min: {mask.min():.4f}, max: {mask.max():.4f}")
        print(f"Mask unique values: {mask.unique()[:10]}")
