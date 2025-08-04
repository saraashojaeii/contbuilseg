#!/usr/bin/env python
"""
Script to evaluate a trained model and visualize predictions.
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.unet import get_unet_model
from models.segformer import get_segformer_model
from data.dataset import CustomDataset, DataPrep
from evaluation.metrics import compute_metrics
from evaluation.visualization import (
    visualize_sample, 
    visualize_prediction_overlay, 
    visualize_metrics_over_epochs,
    visualize_segmentation_comparison
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate model and visualize predictions")
    
    # Model parameters
    parser.add_argument("--model_type", type=str, choices=["unet", "segformer"], required=True,
                        help="Type of model to evaluate")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    
    # Data paths
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory containing data folders")
    parser.add_argument("--set", type=str, choices=["train", "val", "test"], default="test",
                        help="Dataset split to evaluate on")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Whether to save prediction masks")
    parser.add_argument("--visualize", action="store_true",
                        help="Whether to visualize predictions")
    parser.add_argument("--num_vis_samples", type=int, default=5,
                        help="Number of samples to visualize")
    
    # Device settings
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for evaluation (cuda or cpu)")
    
    return parser.parse_args()


def evaluate_unet(model, dataloader, device, output_dir, save_predictions=False, visualize=False, num_vis=5):
    """
    Evaluate a UNet model.
    
    Args:
        model: UNet model
        dataloader: DataLoader for evaluation data
        device: Device to use for evaluation
        output_dir: Directory to save results
        save_predictions: Whether to save prediction masks
        visualize: Whether to visualize predictions
        num_vis: Number of samples to visualize
    
    Returns:
        DataFrame with evaluation metrics
    """
    model.eval()
    metrics_list = []
    
    if save_predictions:
        os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # Process input based on batch format
            if len(batch) == 3:
                images, masks, contours = batch
                images = images.to(device)
                masks = masks.to(device)
                contours = contours.to(device)
                
                # Forward pass
                mask_pred, contour_pred = model(images)
                
                # Compute metrics for both mask and contour
                mask_metrics = compute_metrics(mask_pred.cpu(), masks.cpu())
                contour_metrics = compute_metrics(contour_pred.cpu(), contours.cpu())
                
                # Combine metrics
                metrics = {
                    'sample_idx': i,
                    'mask_iou': mask_metrics['iou'],
                    'mask_dice': mask_metrics['dice'],
                    'mask_precision': mask_metrics['precision'],
                    'mask_recall': mask_metrics['recall'],
                    'contour_iou': contour_metrics['iou'],
                    'contour_dice': contour_metrics['dice'],
                    'contour_precision': contour_metrics['precision'],
                    'contour_recall': contour_metrics['recall']
                }
                
                # Save predictions
                if save_predictions:
                    mask_np = (mask_pred.cpu().numpy() > 0.5).astype(np.uint8) * 255
                    contour_np = (contour_pred.cpu().numpy() > 0.5).astype(np.uint8) * 255
                    
                    np.save(
                        os.path.join(output_dir, 'predictions', f'mask_pred_{i}.npy'),
                        mask_np
                    )
                    np.save(
                        os.path.join(output_dir, 'predictions', f'contour_pred_{i}.npy'),
                        contour_np
                    )
                
                # Visualize sample
                if visualize and i < num_vis:
                    visualize_sample(
                        images[0].cpu().numpy().transpose(1, 2, 0),
                        masks[0].cpu().numpy().squeeze(),
                        mask_pred[0].cpu().numpy().squeeze(),
                        contours[0].cpu().numpy().squeeze(),
                        contour_pred[0].cpu().numpy().squeeze(),
                        figsize=(15, 10)
                    )
                    plt.savefig(os.path.join(output_dir, f'vis_sample_{i}.png'))
                    plt.close()
            else:
                images, masks = batch
                images = images.to(device)
                masks = masks.to(device)
                
                # Forward pass
                outputs = model(images)
                if isinstance(outputs, tuple):
                    mask_pred = outputs[0]  # Use only the mask prediction
                else:
                    mask_pred = outputs
                
                # Compute metrics
                metrics = compute_metrics(mask_pred.cpu(), masks.cpu())
                metrics['sample_idx'] = i
                
                # Save predictions
                if save_predictions:
                    mask_np = (mask_pred.cpu().numpy() > 0.5).astype(np.uint8) * 255
                    np.save(
                        os.path.join(output_dir, 'predictions', f'mask_pred_{i}.npy'),
                        mask_np
                    )
                
                # Visualize sample
                if visualize and i < num_vis:
                    visualize_sample(
                        images[0].cpu().numpy().transpose(1, 2, 0),
                        masks[0].cpu().numpy().squeeze(),
                        mask_pred[0].cpu().numpy().squeeze(),
                        figsize=(15, 5)
                    )
                    plt.savefig(os.path.join(output_dir, f'vis_sample_{i}.png'))
                    plt.close()
            
            metrics_list.append(metrics)
    
    # Create a DataFrame with metrics
    metrics_df = pd.DataFrame(metrics_list)
    
    # Compute average metrics
    avg_metrics = metrics_df.mean().to_dict()
    print("\nAverage Metrics:")
    for k, v in avg_metrics.items():
        if k != 'sample_idx':
            print(f"{k}: {v:.4f}")
    
    # Save metrics to CSV
    metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
    
    return metrics_df


def evaluate_segformer(model, dataloader, device, output_dir, save_predictions=False, visualize=False, num_vis=5):
    """
    Evaluate a SegFormer model.
    
    Args:
        model: SegFormer model
        dataloader: DataLoader for evaluation data
        device: Device to use for evaluation
        output_dir: Directory to save results
        save_predictions: Whether to save prediction masks
        visualize: Whether to visualize predictions
        num_vis: Number of samples to visualize
    
    Returns:
        DataFrame with evaluation metrics
    """
    model.eval()
    metrics_list = []
    
    if save_predictions:
        os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # Move batch to device
            for k, v in batch.items():
                batch[k] = v.to(device)
            
            # Forward pass
            outputs = model(pixel_values=batch['pixel_values'])
            logits = outputs.logits
            
            # Get predictions (class with highest probability)
            predictions = torch.argmax(logits, dim=1)
            
            # Calculate metrics for each class
            all_class_metrics = {}
            for class_id in range(3):  # Assuming 3 classes: background, building, boundary
                # Create binary masks for this class
                pred_mask = (predictions == class_id).float()
                gt_mask = (batch['labels'] == class_id).float()
                
                # Compute metrics
                class_metrics = compute_metrics(pred_mask.cpu(), gt_mask.cpu())
                
                # Store metrics with class prefix
                class_prefix = 'bg_' if class_id == 0 else 'building_' if class_id == 1 else 'boundary_'
                for k, v in class_metrics.items():
                    all_class_metrics[f"{class_prefix}{k}"] = v
            
            # Add sample index
            all_class_metrics['sample_idx'] = i
            
            # Save predictions
            if save_predictions:
                pred_np = predictions.cpu().numpy().astype(np.uint8)
                np.save(
                    os.path.join(output_dir, 'predictions', f'pred_{i}.npy'),
                    pred_np
                )
            
            # Visualize sample
            if visualize and i < num_vis:
                # Convert predictions to RGB for visualization
                pred_vis = np.zeros((predictions.shape[1], predictions.shape[2], 3), dtype=np.float32)
                for c in range(3):
                    pred_vis[predictions[0].cpu() == c, c] = 1.0
                
                # Convert ground truth to RGB for visualization
                gt_vis = np.zeros((batch['labels'].shape[1], batch['labels'].shape[2], 3), dtype=np.float32)
                for c in range(3):
                    gt_vis[batch['labels'][0].cpu() == c, c] = 1.0
                
                # Visualize
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.imshow(batch['pixel_values'][0].cpu().permute(1, 2, 0))
                plt.title('Input Image')
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.imshow(gt_vis)
                plt.title('Ground Truth')
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.imshow(pred_vis)
                plt.title('Prediction')
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'vis_sample_{i}.png'))
                plt.close()
            
            metrics_list.append(all_class_metrics)
    
    # Create a DataFrame with metrics
    metrics_df = pd.DataFrame(metrics_list)
    
    # Compute average metrics
    avg_metrics = metrics_df.mean().to_dict()
    print("\nAverage Metrics:")
    for k, v in avg_metrics.items():
        if k != 'sample_idx':
            print(f"{k}: {v:.4f}")
    
    # Save metrics to CSV
    metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
    
    return metrics_df


def main():
    """Main function to evaluate model."""
    args = parse_args()
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, args.model_type, args.set)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the appropriate model and dataset based on model_type
    if args.model_type == "unet":
        # Load UNet model
        model = get_unet_model()
        model.load_state_dict(torch.load(args.checkpoint)['model_state_dict'])
        model.to(args.device)
        
        # Define transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        # Get data paths
        img_paths = sorted(glob.glob(os.path.join(args.data_dir, args.set, '*.tiff')))
        mask_paths = sorted(glob.glob(os.path.join(args.data_dir, f"{args.set}_labels", '*.tif')))
        
        # Check if contour paths exist
        contour_dir = os.path.dirname(args.data_dir)
        contour_paths = None
        if os.path.exists(os.path.join(contour_dir, f"{args.set}_contours")):
            contour_paths = sorted(glob.glob(os.path.join(contour_dir, f"{args.set}_contours", '*.tif')))
        
        # Create dataset and dataloader
        dataset = CustomDataset(img_paths, mask_paths, contour_paths, transform=transform)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
        
        # Evaluate model
        metrics_df = evaluate_unet(
            model,
            dataloader,
            args.device,
            output_dir,
            args.save_predictions,
            args.visualize,
            args.num_vis_samples
        )
        
    elif args.model_type == "segformer":
        # Load SegFormer model
        from transformers import SegformerForSemanticSegmentation
        
        model = SegformerForSemanticSegmentation.from_pretrained(args.checkpoint)
        model.to(args.device)
        
        # Load image processor
        from transformers import SegformerImageProcessor
        image_processor = SegformerImageProcessor.from_pretrained(args.checkpoint)
        
        # Get data paths
        img_paths = sorted(glob.glob(os.path.join(args.data_dir, args.set, '*.tiff')))
        mask_paths = sorted(glob.glob(os.path.join(args.data_dir, f"{args.set}_labels", '*.tif')))
        
        # Create dataset and dataloader
        dataset = DataPrep(img_paths, mask_paths, image_processor)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
        
        # Evaluate model
        metrics_df = evaluate_segformer(
            model,
            dataloader,
            args.device,
            output_dir,
            args.save_predictions,
            args.visualize,
            args.num_vis_samples
        )
    
    print(f"Evaluation completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
