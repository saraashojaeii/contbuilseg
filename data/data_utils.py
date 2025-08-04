"""
Utility functions for data processing and augmentation for building segmentation.
"""

import cv2
import numpy as np
import torch
from PIL import Image
import os


def load_binary_mask(mask_path):
    """
    Load a grayscale mask and convert to binary.
    
    Args:
        mask_path: Path to the mask image
        
    Returns:
        Binary mask as numpy array
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    binary_mask = binary_mask / 255.0
    return binary_mask


def apply_otsu_thresholding(tensor, is_mask=True):
    """
    Apply Otsu thresholding to a tensor.
    
    Args:
        tensor: A 2D or 3D (1 channel) PyTorch tensor of network outputs, with values in [0, 1]
        is_mask: A boolean indicating if the tensor is a mask (if False, assumes contour)
        
    Returns:
        Threshold value determined by Otsu's method
    """
    # Ensure the tensor is on CPU and convert it to a numpy array
    tensor_np = tensor.cpu().detach().numpy()
    
    # Convert the probabilities to a suitable format for Otsu's thresholding
    tensor_np_scaled = (tensor_np * 255).astype(np.uint8)
    th, _ = cv2.threshold(tensor_np_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def extract_connected_components(mask):
    """
    Extract connected components from a binary mask.
    
    Args:
        mask: Binary mask as numpy array
        
    Returns:
        Number of components and labeled mask
    """
    num_labels, labels = cv2.connectedComponents(mask)
    return num_labels, labels


def create_contour_from_mask(mask):
    """
    Create contour map from binary mask.
    
    Args:
        mask: Binary mask as numpy array
        
    Returns:
        Contour map as numpy array
    """
    # Ensure binary
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create empty contour map
    contour_map = np.zeros_like(mask_uint8)
    
    # Draw contours
    cv2.drawContours(contour_map, contours, -1, 255, 1)
    
    return contour_map / 255.0


def split_data(image_paths, mask_paths, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Split data into train, validation and test sets.
    
    Args:
        image_paths: List of paths to images
        mask_paths: List of paths to masks
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_state: Random seed
        
    Returns:
        Dictionary containing split paths
    """
    from sklearn.model_selection import train_test_split
    
    # First split train and temp (val + test)
    train_img, temp_img, train_mask, temp_mask = train_test_split(
        image_paths, mask_paths, test_size=val_ratio+test_ratio, random_state=random_state
    )
    
    # Calculate the test ratio relative to the temp set
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    
    # Split temp into val and test
    val_img, test_img, val_mask, test_mask = train_test_split(
        temp_img, temp_mask, test_size=relative_test_ratio, random_state=random_state
    )
    
    return {
        'train': {'images': train_img, 'masks': train_mask},
        'val': {'images': val_img, 'masks': val_mask},
        'test': {'images': test_img, 'masks': test_mask}
    }
