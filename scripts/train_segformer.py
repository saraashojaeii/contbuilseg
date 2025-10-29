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
                        help="Base directory containing dataset folders (dataset will be at data_dir/dataset_name)")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Name of the dataset (used for organizing saved models and predictions)")
    parser.add_argument("--use_contours", action="store_true",
                        help="Whether to use contour maps for training (currently not used by SegFormer)")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="nvidia/mit-b0",
                        help="Pretrained model name from HuggingFace")
    parser.add_argument("--num_labels", type=int, default=1,
                        help="Number of segmentation classes (1 for binary building mask)")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Optional Hugging Face access token for private/gated models")
    parser.add_argument("--hf_revision", type=str, default=None,
                        help="Optional model revision (e.g., a specific tag/commit)")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save model every N epochs")
    parser.add_argument("--mask_weight", type=float, default=0.7,
                        help="Weight for mask loss")
    parser.add_argument("--contour_weight", type=float, default=0.3,
                        help="Weight for contour loss")
    parser.add_argument("--merge_weight", type=float, default=0.0,
                        help="Weight for merge separation loss (proxy for merge rate)")
    parser.add_argument("--merge_boundary_width", type=int, default=1,
                        help="Boundary width (pixels) used by merge separation loss")
    
    # Output paths
    parser.add_argument("--output_dir", type=str, default="/root/home/pvc/conbuildseg_results/",
                        help="Directory to save outputs")
    parser.add_argument("--model_save_dir", type=str, default="/root/home/pvc/conbuildseg_results/checkpoints/segformer",
                        help="Directory to save model checkpoints")

    parser.add_argument('--num_workers', type=int, default=1)
    
    # Device settings
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (cuda or cpu)")
    
    # Weights & Biases
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="building_seg", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Optional W&B run name")
    
    return parser.parse_args()


def find_files_with_extensions(directory, extensions):
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


def main():
    """Main function to train SegFormer model."""
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_save_dir, exist_ok=True)
    
    # Initialize SegFormer model
    chosen_model_name = args.model_name
    if chosen_model_name.strip().lower() == "segformer":
        print("[warn] 'segformer' is not a valid model id on Hugging Face; defaulting to 'nvidia/mit-b0'.")
        chosen_model_name = "nvidia/mit-b0"
    if args.num_labels != 1:
        print("[warn] Current trainer is binary-only. Overriding num_labels to 1.")
        args.num_labels = 1
    segformer = get_segformer_model(
        model_name=chosen_model_name,
        num_labels=args.num_labels,
        hf_token=args.hf_token,
        revision=args.hf_revision,
    )
    
    model = segformer.get_model()
    image_processor = segformer.get_image_processor()
    
    # Supported image extensions
    image_extensions = ['png', 'jpg', 'jpeg', 'tif', 'tiff']
    
    # Construct dataset-specific directory path
    dataset_dir = os.path.join(args.data_dir, args.dataset_name)
    print(f"Dataset directory: {dataset_dir}")
    
    # Get data paths
    train_img_paths = find_files_with_extensions(os.path.join(dataset_dir, 'train'), image_extensions)
    train_mask_paths = find_files_with_extensions(os.path.join(dataset_dir, 'train_labels'), image_extensions)
    
    val_img_paths = find_files_with_extensions(os.path.join(dataset_dir, 'val'), image_extensions)
    val_mask_paths = find_files_with_extensions(os.path.join(dataset_dir, 'val_labels'), image_extensions)
    
    test_img_paths = find_files_with_extensions(os.path.join(dataset_dir, 'test'), image_extensions)
    test_mask_paths = find_files_with_extensions(os.path.join(dataset_dir, 'test_labels'), image_extensions)
    
    # Optional contour paths if present
    train_contour_paths = None
    val_contour_paths = None
    train_contour_dir = os.path.join(dataset_dir, 'train_contours')
    val_contour_dir = os.path.join(dataset_dir, 'val_contours')
    if os.path.isdir(train_contour_dir):
        train_contour_paths = find_files_with_extensions(train_contour_dir, image_extensions)
    if os.path.isdir(val_contour_dir):
        val_contour_paths = find_files_with_extensions(val_contour_dir, image_extensions)
    
    print(f"Found {len(train_img_paths)} training images, {len(val_img_paths)} validation images, "
          f"and {len(test_img_paths)} test images")
    
    # Create datasets
    train_dataset = DataPrep(train_img_paths, train_mask_paths, image_processor, contour_paths=train_contour_paths)
    val_dataset = DataPrep(val_img_paths, val_mask_paths, image_processor, contour_paths=val_contour_paths)
    
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
        model_save_dir=args.model_save_dir,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        dataset_name=args.dataset_name,
        mask_weight=args.mask_weight,
        contour_weight=args.contour_weight,
        merge_weight=args.merge_weight,
        merge_boundary_width=args.merge_boundary_width
    )
    
    # Train model
    trainer.train(num_epochs=args.epochs, save_every=args.save_every)


if __name__ == "__main__":
    main()
