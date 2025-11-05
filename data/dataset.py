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
    def __init__(self, image_paths, label_paths1, image_processor, contour_paths: Optional[List[str]] = None, processor_kwargs: Optional[Dict] = None):
        """
        Initialize the DataPrep dataset.
        
        Args:
            image_paths: List of paths to input images
            label_paths1: List of paths to corresponding mask images
            image_processor: SegformerImageProcessor for preprocessing
        """
        self.image_paths = image_paths
        self.label1_paths = label_paths1
        self.contour_paths = contour_paths
        self.image_processor = image_processor
        # Optional kwargs passed to image_processor during call (e.g., size or do_resize)
        self.processor_kwargs = processor_kwargs or {'do_resize': False}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get a preprocessed image and mask pair.
        
        Returns:
            Encoded inputs containing pixel values and labels processed by SegformerImageProcessor
        """
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask_img = Image.open(self.label1_paths[idx]).convert('L')
        contour_img = None
        if self.contour_paths is not None:
            contour_img = Image.open(self.contour_paths[idx]).convert('L')

        # Process image to tensor
        proc = self.image_processor(image, return_tensors="pt", **self.processor_kwargs)
        pixel_values = proc["pixel_values"].squeeze(0)  # CxHxW
        target_h, target_w = pixel_values.shape[-2], pixel_values.shape[-1]

        # Prepare mask as float tensor in [0,1], resized to pixel_values size (NEAREST to keep labels crisp)
        mask_arr = np.array(mask_img).astype(np.float32) / 255.0
        mask_resized = Image.fromarray((mask_arr * 255).astype(np.uint8)).resize((target_w, target_h), resample=Image.NEAREST)
        mask_tensor = torch.from_numpy(np.array(mask_resized).astype(np.float32) / 255.0).unsqueeze(0)

        sample = {
            'pixel_values': pixel_values,
            'mask': mask_tensor,
        }

        # Prepare contour if available as float tensor, resized similarly (NEAREST keeps the intensity plateaus stable)
        if contour_img is not None:
            contour_arr = np.array(contour_img).astype(np.float32) / 255.0
            contour_resized = Image.fromarray((contour_arr * 255).astype(np.uint8)).resize((target_w, target_h), resample=Image.NEAREST)
            contour_tensor = torch.from_numpy(np.array(contour_resized).astype(np.float32) / 255.0).unsqueeze(0)
            sample['contours'] = contour_tensor

        return sample


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
            # Convert mask to tensor and normalize to [0, 1]
            mask_np = np.array(mask, dtype=np.float32)
            # Normalize to [0, 1] if needed
            if mask_np.max() > 1.0:
                mask_np = mask_np / 255.0
            mask = torch.from_numpy(mask_np).unsqueeze(0)  # Add channel dimension
            # Clamp to ensure [0, 1] range
            mask = torch.clamp(mask, 0.0, 1.0)
        else:
            # If no transform, still ensure proper format
            mask_np = np.array(mask, dtype=np.float32)
            if mask_np.max() > 1.0:
                mask_np = mask_np / 255.0
            mask = torch.from_numpy(mask_np).unsqueeze(0)
            mask = torch.clamp(mask, 0.0, 1.0)
            
        if self.label2_paths is None:
            return image, mask
            
        contour = Image.open(self.label2_paths[idx]).convert('L')
        if self.transform:
            # Convert contour to tensor and normalize
            contour_np = np.array(contour, dtype=np.float32)
            if contour_np.max() > 1.0:
                contour_np = contour_np / 255.0
            contour = torch.from_numpy(contour_np).unsqueeze(0)
            contour = torch.clamp(contour, 0.0, 1.0)
        else:
            contour_np = np.array(contour, dtype=np.float32)
            if contour_np.max() > 1.0:
                contour_np = contour_np / 255.0
            contour = torch.from_numpy(contour_np).unsqueeze(0)
            contour = torch.clamp(contour, 0.0, 1.0)
            
        return image, mask, contour


class TiledDataPrep(Dataset):
    """
    Dataset that yields fixed-size tiles from large images and corresponding masks (and optional contours).
    Useful for very large inputs like INRIA (e.g., 5000x5000) to avoid GPU OOM.
    """
    def __init__(
        self,
        image_paths: List[str],
        label_paths1: List[str],
        image_processor,
        contour_paths: Optional[List[str]] = None,
        tile_size: int = 512,
        stride: Optional[int] = None,
        processor_kwargs: Optional[Dict] = None,
    ):
        self.image_paths = image_paths
        self.label1_paths = label_paths1
        self.contour_paths = contour_paths
        self.image_processor = image_processor
        self.tile_size = tile_size
        self.stride = stride or tile_size
        # For tiles we do not resize by default; tiles already match model input
        self.processor_kwargs = processor_kwargs or {'do_resize': False}

        # Build tile index: list of (img_idx, y, x)
        self._tiles = []
        for i, img_path in enumerate(self.image_paths):
            with Image.open(img_path) as im:
                w, h = im.size
            # compute grid with full coverage
            y_positions = list(range(0, max(1, h - self.tile_size + 1), self.stride))
            x_positions = list(range(0, max(1, w - self.tile_size + 1), self.stride))
            if len(y_positions) == 0:
                y_positions = [0]
            if len(x_positions) == 0:
                x_positions = [0]
            if y_positions[-1] + self.tile_size < h:
                y_positions.append(max(0, h - self.tile_size))
            if x_positions[-1] + self.tile_size < w:
                x_positions.append(max(0, w - self.tile_size))
            for top in y_positions:
                for left in x_positions:
                    self._tiles.append((i, top, left))

    def __len__(self) -> int:
        return len(self._tiles)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_idx, top, left = self._tiles[idx]

        image = Image.open(self.image_paths[img_idx]).convert('RGB')
        mask_img = Image.open(self.label1_paths[img_idx]).convert('L')
        contour_img = None
        if self.contour_paths is not None:
            contour_img = Image.open(self.contour_paths[img_idx]).convert('L')

        # Crop tiles (pad if tile exceeds boundary)
        tile_box = (left, top, left + self.tile_size, top + self.tile_size)
        image_tile = self._crop_with_pad(image, tile_box)
        mask_tile = self._crop_with_pad(mask_img, tile_box)
        contour_tile = self._crop_with_pad(contour_img, tile_box) if contour_img is not None else None

        # Process image
        proc = self.image_processor(image_tile, return_tensors="pt", **self.processor_kwargs)
        pixel_values = proc["pixel_values"].squeeze(0)  # CxHxW
        target_h, target_w = pixel_values.shape[-2], pixel_values.shape[-1]

        # Prepare mask (nearest to preserve labels)
        mask_arr = np.array(mask_tile).astype(np.float32) / 255.0
        mask_resized = Image.fromarray((mask_arr * 255).astype(np.uint8)).resize((target_w, target_h), resample=Image.NEAREST)
        mask_tensor = torch.from_numpy(np.array(mask_resized).astype(np.float32) / 255.0).unsqueeze(0)

        sample = {
            'pixel_values': pixel_values,
            'mask': mask_tensor,
        }

        if contour_tile is not None:
            contour_arr = np.array(contour_tile).astype(np.float32) / 255.0
            contour_resized = Image.fromarray((contour_arr * 255).astype(np.uint8)).resize((target_w, target_h), resample=Image.NEAREST)
            contour_tensor = torch.from_numpy(np.array(contour_resized).astype(np.float32) / 255.0).unsqueeze(0)
            sample['contours'] = contour_tensor

        return sample

    @staticmethod
    def _crop_with_pad(img: Optional[Image.Image], box: tuple) -> Image.Image:
        """Crop the region, padding with zeros if box exceeds image bounds."""
        if img is None:
            return None
        left, top, right, bottom = box
        w, h = img.size
        # Compute intersection
        inter_left = max(0, left)
        inter_top = max(0, top)
        inter_right = min(w, right)
        inter_bottom = min(h, bottom)

        crop = img.crop((inter_left, inter_top, inter_right, inter_bottom))
        # If exact size, return
        target_w = right - left
        target_h = bottom - top
        if crop.size == (target_w, target_h):
            return crop

        # Otherwise, paste into a black canvas
        canvas = Image.new(img.mode, (target_w, target_h))
        paste_x = inter_left - left
        paste_y = inter_top - top
        canvas.paste(crop, (paste_x, paste_y))
        return canvas


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
