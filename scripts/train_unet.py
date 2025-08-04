#!/usr/bin/env python
"""
Script to train a UNet model for building segmentation with optional contour awareness.
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import glob

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.unet import get_unet_model
from data.dataset import CustomDataset
from training.unet_trainer import UNetTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train UNet model for building segmentation")
    
    # Data paths
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory containing data folders")
    parser.add_argument("--use_contours", action="store_true",
                        help="Whether to use contour maps for training")
    parser.add_argument("--contour_dir", type=str, default=None,
                        help="Directory containing contour maps (if use_contours=True)")
    
    # Model parameters
    parser.add_argument("--in_channels", type=int, default=3,
                        help="Number of input channels (3 for RGB)")
    parser.add_argument("--out_channels_mask", type=int, default=1,
                        help="Number of output channels for mask")
    parser.add_argument("--out_channels_contour", type=int, default=1,
                        help="Number of output channels for contour")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save model every N epochs")
    parser.add_argument("--mask_weight", type=float, default=0.7,
                        help="Weight for mask loss")
    parser.add_argument("--contour_weight", type=float, default=0.3,
                        help="Weight for contour loss")
    
    # Output paths
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Directory to save outputs")
    parser.add_argument("--model_save_dir", type=str, default="./checkpoints/unet",
                        help="Directory to save model checkpoints")
    
    # Device settings
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (cuda or cpu)")
    
    return parser.parse_args()


def main():
    """Main function to train UNet model."""
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_save_dir, exist_ok=True)
    
    # Initialize model
    model = get_unet_model(
        in_channels=args.in_channels,
        out_channels_mask=args.out_channels_mask,
        out_channels_contour=args.out_channels_contour
    )
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Get data paths
    train_img_paths = sorted(glob.glob(os.path.join(args.data_dir, 'train', '*.tiff')))
    train_mask_paths = sorted(glob.glob(os.path.join(args.data_dir, 'train_labels', '*.tif')))
    
    val_img_paths = sorted(glob.glob(os.path.join(args.data_dir, 'val', '*.tiff')))
    val_mask_paths = sorted(glob.glob(os.path.join(args.data_dir, 'val_labels', '*.tif')))
    
    test_img_paths = sorted(glob.glob(os.path.join(args.data_dir, 'test', '*.tiff')))
    test_mask_paths = sorted(glob.glob(os.path.join(args.data_dir, 'test_labels', '*.tif')))
    
    # Get contour paths if needed
    train_contour_paths = None
    val_contour_paths = None
    test_contour_paths = None
    
    if args.use_contours and args.contour_dir:
        train_contour_paths = sorted(glob.glob(os.path.join(args.contour_dir, 'train_contours', '*.tif')))
        val_contour_paths = sorted(glob.glob(os.path.join(args.contour_dir, 'val_contours', '*.tif')))
        test_contour_paths = sorted(glob.glob(os.path.join(args.contour_dir, 'test_contours', '*.tif')))
        
        print(f"Found {len(train_contour_paths)} training contours, {len(val_contour_paths)} validation contours, "
              f"and {len(test_contour_paths)} test contours")
    
    print(f"Found {len(train_img_paths)} training images, {len(val_img_paths)} validation images, "
          f"and {len(test_img_paths)} test images")
    
    # Create datasets
    train_dataset = CustomDataset(
        train_img_paths, 
        train_mask_paths, 
        train_contour_paths, 
        transform=transform
    )
    val_dataset = CustomDataset(
        val_img_paths, 
        val_mask_paths, 
        val_contour_paths,
        transform=transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=1
    )
    
    # Initialize trainer
    trainer = UNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        learning_rate=args.learning_rate,
        model_save_dir=args.model_save_dir,
        mask_weight=args.mask_weight,
        contour_weight=args.contour_weight
    )
    
    # Train model
    trainer.train(num_epochs=args.epochs, save_every=args.save_every)


if __name__ == "__main__":
    main()
