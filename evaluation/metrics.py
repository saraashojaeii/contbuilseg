"""
Metrics for evaluating building segmentation models.
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import cv2


def compute_iou(pred_mask, gt_mask):
    """
    Compute Intersection over Union (IoU) for binary masks.
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
        
    Returns:
        IoU score
    """
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().detach().numpy()
    
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.cpu().detach().numpy()
    
    # Convert to binary if needed
    if np.max(pred_mask) > 1:
        pred_mask = pred_mask / 255.0
    if np.max(gt_mask) > 1:
        gt_mask = gt_mask / 255.0
    
    # Binarize using threshold of 0.5
    pred_mask = (pred_mask > 0.5).astype(np.float32)
    gt_mask = (gt_mask > 0.5).astype(np.float32)
    
    # Compute intersection and union
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    # Compute IoU
    iou = intersection / (union + 1e-6)
    
    return iou


def compute_dice_coefficient(pred_mask, gt_mask):
    """
    Compute Dice coefficient (F1 score) for binary masks.
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
        
    Returns:
        Dice coefficient
    """
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().detach().numpy()
    
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.cpu().detach().numpy()
    
    # Convert to binary if needed
    if np.max(pred_mask) > 1:
        pred_mask = pred_mask / 255.0
    if np.max(gt_mask) > 1:
        gt_mask = gt_mask / 255.0
    
    # Binarize using threshold of 0.5
    pred_mask = (pred_mask > 0.5).astype(np.float32)
    gt_mask = (gt_mask > 0.5).astype(np.float32)
    
    # Compute intersection
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    
    # Compute Dice coefficient
    dice = (2 * intersection) / (pred_mask.sum() + gt_mask.sum() + 1e-6)
    
    return dice


def compute_precision_recall(pred_mask, gt_mask):
    """
    Compute precision and recall for binary masks.
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
        
    Returns:
        Tuple of (precision, recall)
    """
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().detach().numpy()
    
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.cpu().detach().numpy()
    
    # Convert to binary if needed
    if np.max(pred_mask) > 1:
        pred_mask = pred_mask / 255.0
    if np.max(gt_mask) > 1:
        gt_mask = gt_mask / 255.0
    
    # Binarize using threshold of 0.5
    pred_mask = (pred_mask > 0.5).astype(np.float32)
    gt_mask = (gt_mask > 0.5).astype(np.float32)
    
    # Compute intersection (true positives)
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    
    # Compute precision and recall
    precision = intersection / (pred_mask.sum() + 1e-6)
    recall = intersection / (gt_mask.sum() + 1e-6)
    
    return precision, recall


def compute_metrics(pred_mask, gt_mask):
    """
    Compute all segmentation metrics for binary masks.
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
        
    Returns:
        Dictionary of metrics
    """
    iou = compute_iou(pred_mask, gt_mask)
    dice = compute_dice_coefficient(pred_mask, gt_mask)
    precision, recall = compute_precision_recall(pred_mask, gt_mask)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    return {
        'iou': iou,
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def compute_metrics_batch(pred_masks, gt_masks):
    """
    Compute metrics for a batch of masks.
    
    Args:
        pred_masks: Batch of predicted masks
        gt_masks: Batch of ground truth masks
        
    Returns:
        Dictionary of average metrics
    """
    batch_size = len(pred_masks)
    metrics = {
        'iou': 0,
        'dice': 0,
        'precision': 0,
        'recall': 0,
        'f1': 0
    }
    
    for i in range(batch_size):
        batch_metrics = compute_metrics(pred_masks[i], gt_masks[i])
        for k, v in batch_metrics.items():
            metrics[k] += v
    
    # Average metrics
    for k in metrics:
        metrics[k] /= batch_size
    
    return metrics


def evaluate_model(model, data_loader, device='cuda'):
    """
    Evaluate a model on a dataset.
    
    Args:
        model: PyTorch model
        data_loader: DataLoader for evaluation data
        device: Device to use ('cuda' or 'cpu')
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    metrics = {
        'iou': 0,
        'dice': 0,
        'precision': 0,
        'recall': 0,
        'f1': 0
    }
    
    num_batches = len(data_loader)
    with torch.no_grad():
        for batch in data_loader:
            # Handle different dataset formats
            if isinstance(batch, dict):  # SegFormer dataset format
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(pixel_values=pixel_values, labels=labels)
                logits = outputs.logits
                pred_masks = torch.argmax(logits, dim=1)
                batch_metrics = compute_metrics_batch(pred_masks, labels)
            else:  # UNet dataset format
                images, masks = batch[0].to(device), batch[1].to(device)
                outputs = model(images)
                pred_masks = outputs[0] if isinstance(outputs, tuple) else outputs
                batch_metrics = compute_metrics_batch(pred_masks, masks)
            
            for k, v in batch_metrics.items():
                metrics[k] += v
    
    # Average metrics
    for k in metrics:
        metrics[k] /= num_batches
    
    return metrics


def create_metrics_table(metrics_list, epoch_list):
    """
    Create a pandas DataFrame table for metrics across epochs.
    
    Args:
        metrics_list: List of metric dictionaries
        epoch_list: List of epoch numbers
        
    Returns:
        DataFrame with metrics
    """
    # Create dictionary for DataFrame
    metrics_dict = {
        'Epoch': epoch_list,
        'IoU': [m['iou'] for m in metrics_list],
        'Dice/F1': [m['dice'] for m in metrics_list],
        'Precision': [m['precision'] for m in metrics_list],
        'Recall': [m['recall'] for m in metrics_list]
    }
    
    # Create DataFrame
    df = pd.DataFrame(metrics_dict)
    
    return df
