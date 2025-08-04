"""
Visualization utilities for building segmentation results.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.patches as mpatches


def visualize_sample(image, mask, pred=None, contour=None, pred_contour=None, figsize=(15, 10)):
    """
    Visualize a sample image with its ground truth mask and predictions.
    
    Args:
        image: Input image (numpy array or tensor)
        mask: Ground truth mask (numpy array or tensor)
        pred: Predicted mask (optional, numpy array or tensor)
        contour: Ground truth contour (optional, numpy array or tensor)
        pred_contour: Predicted contour (optional, numpy array or tensor)
        figsize: Figure size
    """
    # Convert tensors to numpy arrays
    if isinstance(image, torch.Tensor):
        image = image.cpu().detach().numpy()
        if image.shape[0] == 3:  # CHW to HWC
            image = np.transpose(image, (1, 2, 0))
    
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().detach().numpy()
        if len(mask.shape) == 3 and mask.shape[0] == 1:  # CHW to HW
            mask = mask.squeeze(0)
    
    if isinstance(pred, torch.Tensor) and pred is not None:
        pred = pred.cpu().detach().numpy()
        if len(pred.shape) == 3 and pred.shape[0] == 1:  # CHW to HW
            pred = pred.squeeze(0)
    
    if isinstance(contour, torch.Tensor) and contour is not None:
        contour = contour.cpu().detach().numpy()
        if len(contour.shape) == 3 and contour.shape[0] == 1:  # CHW to HW
            contour = contour.squeeze(0)
    
    if isinstance(pred_contour, torch.Tensor) and pred_contour is not None:
        pred_contour = pred_contour.cpu().detach().numpy()
        if len(pred_contour.shape) == 3 and pred_contour.shape[0] == 1:  # CHW to HW
            pred_contour = pred_contour.squeeze(0)
    
    # Normalize image if needed
    if image.max() > 1:
        image = image / 255.0
    
    # Binarize masks if needed
    if mask is not None and mask.max() > 1:
        mask = mask / 255.0
    
    if pred is not None and pred.max() > 1:
        pred = pred / 255.0
    
    if contour is not None and contour.max() > 1:
        contour = contour / 255.0
    
    if pred_contour is not None and pred_contour.max() > 1:
        pred_contour = pred_contour / 255.0
    
    # Determine number of subplots
    n_plots = 1 + (mask is not None) + (pred is not None) + (contour is not None) + (pred_contour is not None)
    fig_cols = min(n_plots, 3)
    fig_rows = (n_plots + fig_cols - 1) // fig_cols
    
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=figsize)
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    plot_idx = 1
    
    # Plot ground truth mask
    if mask is not None:
        axes[plot_idx].imshow(mask, cmap='gray')
        axes[plot_idx].set_title('Ground Truth Mask')
        axes[plot_idx].axis('off')
        plot_idx += 1
    
    # Plot predicted mask
    if pred is not None:
        axes[plot_idx].imshow(pred, cmap='gray')
        axes[plot_idx].set_title('Predicted Mask')
        axes[plot_idx].axis('off')
        plot_idx += 1
    
    # Plot ground truth contour
    if contour is not None:
        axes[plot_idx].imshow(contour, cmap='gray')
        axes[plot_idx].set_title('Ground Truth Contour')
        axes[plot_idx].axis('off')
        plot_idx += 1
    
    # Plot predicted contour
    if pred_contour is not None:
        axes[plot_idx].imshow(pred_contour, cmap='gray')
        axes[plot_idx].set_title('Predicted Contour')
        axes[plot_idx].axis('off')
    
    # Hide any unused axes
    for i in range(plot_idx + 1, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_prediction_overlay(image, mask=None, pred=None, alpha=0.5, figsize=(12, 5)):
    """
    Visualize image with overlaid ground truth and/or prediction.
    
    Args:
        image: Input image (numpy array or tensor)
        mask: Ground truth mask (optional, numpy array or tensor)
        pred: Predicted mask (optional, numpy array or tensor)
        alpha: Transparency level for overlays
        figsize: Figure size
    """
    # Convert tensors to numpy arrays
    if isinstance(image, torch.Tensor):
        image = image.cpu().detach().numpy()
        if image.shape[0] == 3:  # CHW to HWC
            image = np.transpose(image, (1, 2, 0))
    
    if isinstance(mask, torch.Tensor) and mask is not None:
        mask = mask.cpu().detach().numpy()
        if len(mask.shape) == 3 and mask.shape[0] == 1:  # CHW to HW
            mask = mask.squeeze(0)
    
    if isinstance(pred, torch.Tensor) and pred is not None:
        pred = pred.cpu().detach().numpy()
        if len(pred.shape) == 3 and pred.shape[0] == 1:  # CHW to HW
            pred = pred.squeeze(0)
    
    # Normalize image if needed
    if image.max() > 1:
        image = image / 255.0
    
    # Binarize masks if needed
    if mask is not None and mask.max() > 1:
        mask = mask / 255.0
    
    if pred is not None and pred.max() > 1:
        pred = pred / 255.0
    
    # Create RGB mask overlays
    if mask is not None:
        mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
        mask_rgb[:, :, 1] = mask  # Green for ground truth
    
    if pred is not None:
        pred_rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.float32)
        pred_rgb[:, :, 0] = pred  # Red for predictions
    
    # Create figure
    n_plots = 1 + (mask is not None) + (pred is not None)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = np.array([axes])
    
    # Plot original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    plot_idx = 1
    
    # Plot image with ground truth overlay
    if mask is not None:
        axes[plot_idx].imshow(image)
        axes[plot_idx].imshow(mask_rgb, alpha=alpha)
        axes[plot_idx].set_title('Ground Truth Overlay')
        axes[plot_idx].axis('off')
        
        # Add legend
        gt_patch = mpatches.Patch(color='green', label='Ground Truth')
        axes[plot_idx].legend(handles=[gt_patch], loc='upper right')
        
        plot_idx += 1
    
    # Plot image with prediction overlay
    if pred is not None:
        axes[plot_idx].imshow(image)
        axes[plot_idx].imshow(pred_rgb, alpha=alpha)
        axes[plot_idx].set_title('Prediction Overlay')
        axes[plot_idx].axis('off')
        
        # Add legend
        pred_patch = mpatches.Patch(color='red', label='Prediction')
        axes[plot_idx].legend(handles=[pred_patch], loc='upper right')
    
    plt.tight_layout()
    plt.show()


def visualize_metrics_over_epochs(metrics_df, figsize=(12, 8)):
    """
    Visualize metrics over epochs.
    
    Args:
        metrics_df: DataFrame containing metrics over epochs
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    metrics = ['IoU', 'Dice/F1', 'Precision', 'Recall']
    
    for i, metric in enumerate(metrics):
        axes[i].plot(metrics_df['Epoch'], metrics_df[metric], marker='o')
        axes[i].set_title(f'{metric} over Epochs')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(metric)
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.show()


def visualize_segmentation_comparison(image_paths, mask_paths, pred_paths, n_samples=3, figsize=(15, 15)):
    """
    Visualize a comparison of ground truth and predictions for multiple images.
    
    Args:
        image_paths: List of paths to input images
        mask_paths: List of paths to ground truth masks
        pred_paths: List of paths to predicted masks
        n_samples: Number of samples to visualize
        figsize: Figure size
    """
    n_samples = min(n_samples, len(image_paths))
    
    fig, axes = plt.subplots(n_samples, 3, figsize=figsize)
    
    for i in range(n_samples):
        # Load image and masks
        image = np.array(Image.open(image_paths[i]))
        mask = np.array(Image.open(mask_paths[i]).convert('L'))
        pred = np.array(Image.open(pred_paths[i]).convert('L'))
        
        # Normalize
        if image.max() > 1:
            image = image / 255.0
        
        if mask.max() > 1:
            mask = mask / 255.0
        
        if pred.max() > 1:
            pred = pred / 255.0
        
        # Display
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f'Sample {i+1}: Original')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title(f'Sample {i+1}: Ground Truth')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_title(f'Sample {i+1}: Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
