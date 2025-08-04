#!/usr/bin/env python
"""
Script to train a SegFormer model for building segmentation.
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
import glob

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.segformer import get_segformer_model
from data.dataset import DataPrep
from training.segformer_trainer import SegFormerTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train SegFormer model for building segmentation")
    
    # Data paths
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory containing data folders")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="nvidia/mit-b0",
                        help="Pretrained model name from HuggingFace")
    parser.add_argument("--num_labels", type=int, default=3,
                        help="Number of segmentation classes")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save model every N epochs")
    
    # Output paths
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Directory to save outputs")
    parser.add_argument("--model_save_dir", type=str, default="./checkpoints/segformer",
                        help="Directory to save model checkpoints")
    
    # Device settings
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (cuda or cpu)")
    
    return parser.parse_args()


def main():
    """Main function to train SegFormer model."""
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_save_dir, exist_ok=True)
    
    # Initialize SegFormer model
    segformer = get_segformer_model(
        model_name=args.model_name,
        num_labels=args.num_labels
    )
    
    model = segformer.get_model()
    image_processor = segformer.get_image_processor()
    
    # Get data paths
    train_img_paths = sorted(glob.glob(os.path.join(args.data_dir, 'train', '*.tiff')))
    train_mask_paths = sorted(glob.glob(os.path.join(args.data_dir, 'train_labels', '*.tif')))
    
    val_img_paths = sorted(glob.glob(os.path.join(args.data_dir, 'val', '*.tiff')))
    val_mask_paths = sorted(glob.glob(os.path.join(args.data_dir, 'val_labels', '*.tif')))
    
    test_img_paths = sorted(glob.glob(os.path.join(args.data_dir, 'test', '*.tiff')))
    test_mask_paths = sorted(glob.glob(os.path.join(args.data_dir, 'test_labels', '*.tif')))
    
    print(f"Found {len(train_img_paths)} training images, {len(val_img_paths)} validation images, "
          f"and {len(test_img_paths)} test images")
    
    # Create datasets
    train_dataset = DataPrep(train_img_paths, train_mask_paths, image_processor)
    val_dataset = DataPrep(val_img_paths, val_mask_paths, image_processor)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    
    # Initialize trainer
    trainer = SegFormerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        learning_rate=args.learning_rate,
        model_save_dir=args.model_save_dir
    )
    
    # Train model
    trainer.train(num_epochs=args.epochs, save_every=args.save_every)


if __name__ == "__main__":
    main()
