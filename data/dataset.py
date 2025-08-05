"""
Dataset classes for building segmentation tasks.
"""

import os
import glob
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class DataPrep(Dataset):
    """
    Dataset class for SegFormer model used for semantic segmentation of buildings.
    Loads RGB images and grayscale masks, and processes them using SegformerImageProcessor.
    """
    def __init__(self, image_paths, label_paths1, image_processor):
        """
        Initialize the DataPrep dataset.
        
        Args:
            image_paths: List of paths to input images
            label_paths1: List of paths to corresponding mask images
            image_processor: SegformerImageProcessor for preprocessing
        """
        self.image_paths = image_paths
        self.label1_paths = label_paths1
        self.image_processor = image_processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get a preprocessed image and mask pair.
        
        Returns:
            Encoded inputs containing pixel values and labels processed by SegformerImageProcessor
        """
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.label1_paths[idx]).convert('L')
        mask = np.array(mask).astype(np.uint8)
        mask = Image.fromarray(mask)
        
        encoded_inputs = self.image_processor(image, segmentation_maps=mask, return_tensors="pt")
        
        # Remove batch dimension
        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()
        
        return encoded_inputs


class CustomDataset(Dataset):
    """
    Dataset class for UNet model used for segmentation of buildings with both mask and contour outputs.
    Loads RGB images, binary masks, and contour maps.
    """
    def __init__(self, image_paths, label1_paths, label2_paths=None, transform=None):
        """
        Initialize the CustomDataset.
        
        Args:
            image_paths: List of paths to input images
            label1_paths: List of paths to corresponding mask images
            label2_paths: Optional list of paths to contour images
            transform: Optional transforms to be applied on the data
        """
        self.image_paths = image_paths
        self.label1_paths = label1_paths
        self.label2_paths = label2_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get an image, mask, and optionally contour.
        
        Returns:
            Tuple containing (image, mask) or (image, mask, contour)
        """
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.label1_paths[idx]).convert('L')
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            
        if self.label2_paths is None:
            return image, mask
            
        contour = Image.open(self.label2_paths[idx]).convert('L')
        if self.transform:
            contour = self.transform(contour)
            
        return image, mask, contour


def find_files_with_extensions(directory: str, extensions: List[str]) -> List[str]:
    """
    Find files with any of the specified extensions in a directory.
    
    Args:
        directory: Directory to search in
        extensions: List of file extensions to search for (without dots)
    
    Returns:
        Sorted list of file paths
    """
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, f'*.{ext}')))
    return sorted(files)


def get_data_paths(data_dir: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Get paths to images and masks for train, validation, and test sets.
    
    Args:
        data_dir: Root directory containing data folders
    
    Returns:
        Dictionary containing paths for train, val and test splits
    """
    # Supported image extensions
    image_extensions = ['png', 'jpg', 'jpeg', 'tif', 'tiff']
    
    paths = {
        'train': {
            'images': find_files_with_extensions(os.path.join(data_dir, 'train'), image_extensions),
            'masks': find_files_with_extensions(os.path.join(data_dir, 'train_labels'), image_extensions),
        },
        'val': {
            'images': find_files_with_extensions(os.path.join(data_dir, 'val'), image_extensions),
            'masks': find_files_with_extensions(os.path.join(data_dir, 'val_labels'), image_extensions),
        },
        'test': {
            'images': find_files_with_extensions(os.path.join(data_dir, 'test'), image_extensions),
            'masks': find_files_with_extensions(os.path.join(data_dir, 'test_labels'), image_extensions),
        }
    }
    
    return paths


def create_dataloaders(dataset_train, dataset_val, dataset_test=None, 
                      batch_size=8, num_workers=0):
    """
    Create DataLoaders for training, validation and optionally test datasets.
    
    Args:
        dataset_train: Training dataset
        dataset_val: Validation dataset
        dataset_test: Test dataset (optional)
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        
    Returns:
        Dictionary containing dataloaders
    """
    from torch.utils.data import DataLoader
    
    loaders = {
        'train': DataLoader(dataset_train, batch_size=batch_size, 
                           shuffle=True, num_workers=num_workers),
        'val': DataLoader(dataset_val, batch_size=1, 
                         shuffle=False, num_workers=num_workers)
    }
    
    if dataset_test is not None:
        loaders['test'] = DataLoader(dataset_test, batch_size=1,
                                    shuffle=False, num_workers=num_workers)
    
    return loaders
