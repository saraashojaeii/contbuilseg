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
    # Pixel accuracy
    acc = compute_pixel_accuracy(pred_mask, gt_mask)
    
    return {
        'iou': iou,
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': acc
    }


def _to_numpy_binary(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = x.astype(np.float32)
    if x.max() > 1:
        x = x / 255.0
    return (x > 0.5).astype(np.uint8)


def compute_pixel_accuracy(pred_mask, gt_mask):
    """
    Pixel accuracy between two binary masks.
    """
    pred = _to_numpy_binary(pred_mask)
    gt = _to_numpy_binary(gt_mask)
    return (pred == gt).mean().item() if hasattr((pred==gt).mean(), 'item') else float((pred==gt).mean())


def _label_instances(binary_mask):
    """
    Label connected components in a binary mask using 8-connectivity.
    Returns labels image and number of components (excluding background).
    """
    mask = _to_numpy_binary(binary_mask)
    num, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    # num includes background label 0
    return labels, num - 1


def _component_centroids(labels):
    """
    Compute centroids for labeled components (label>0). Returns list of (y, x).
    """
    centroids = []
    max_label = labels.max()
    for lab in range(1, max_label + 1):
        ys, xs = np.where(labels == lab)
        if ys.size == 0:
            continue
        cy = ys.mean()
        cx = xs.mean()
        centroids.append((cy, cx))
    return centroids


def compute_merge_rate(pred_mask, gt_mask):
    """
    Merge rate = 2*M / (N_gt + N_pred), where M is the number of predicted
    instances that overlap with two or more GT instances (i.e., predicted merges).
    """
    pred = _to_numpy_binary(pred_mask)
    gt = _to_numpy_binary(gt_mask)
    pred_labels, n_pred = _label_instances(pred)
    gt_labels, n_gt = _label_instances(gt)

    merges = 0
    for lab in range(1, pred_labels.max() + 1):
        coords = (pred_labels == lab)
        overlapping_gts = np.unique(gt_labels[coords])
        overlapping_gts = overlapping_gts[overlapping_gts > 0]
        if overlapping_gts.size >= 2:
            merges += 1
    denom = max(n_gt + n_pred, 1)
    return (2.0 * merges) / denom


def compute_centroid_prf(pred_mask, gt_mask):
    """
    Centroid-based precision/recall/F1.
    - A predicted instance is correct if its centroid lies inside any GT instance.
    - Recall counts GT instances that contain at least one predicted centroid.
    """
    pred = _to_numpy_binary(pred_mask)
    gt = _to_numpy_binary(gt_mask)
    pred_labels, n_pred = _label_instances(pred)
    gt_labels, n_gt = _label_instances(gt)

    # Centroids of predictions
    pred_centroids = _component_centroids(pred_labels)

    # Map centroid correctness
    tp = 0
    matched_gt_labels = set()
    for (cy, cx) in pred_centroids:
        y = int(round(cy))
        x = int(round(cx))
        if 0 <= y < gt_labels.shape[0] and 0 <= x < gt_labels.shape[1]:
            lab = gt_labels[y, x]
            if lab > 0:
                tp += 1
                matched_gt_labels.add(int(lab))

    fp = max(n_pred - tp, 0)
    fn = max(n_gt - len(matched_gt_labels), 0)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return precision, recall, f1


def compute_all_metrics(pred_mask, gt_mask):
    """
    Convenience wrapper returning all requested metrics.
    """
    base = compute_metrics(pred_mask, gt_mask)
    base['merge_rate'] = compute_merge_rate(pred_mask, gt_mask)
    cprec, crec, cf1 = compute_centroid_prf(pred_mask, gt_mask)
    base['centroid_precision'] = cprec
    base['centroid_recall'] = crec
    base['centroid_f1'] = cf1
    return base


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
