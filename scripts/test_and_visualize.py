"""
Test a trained model on test images and create color-coded visualizations.
"""
import os
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

# Import model classes
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.buildformer import DualHeadBuildFormer
from models.unet import UNet
from models.segformer import DualHeadSegFormer
from data.dataset import find_files_with_extensions
from transformers import SegformerImageProcessor


def calculate_and_color_buildings(ground_truth, prediction):
    """
    Color-code buildings based on ground truth and predictions.
    
    Returns:
        - binary_ground_truth: Binarized ground truth
        - binary_prediction: Binarized prediction
        - output_image: Color-coded visualization (BGR format)
    """
    # Threshold images to binary
    _, binary_ground_truth = cv2.threshold(ground_truth, 128, 255, cv2.THRESH_BINARY)
    _, binary_prediction = cv2.threshold(prediction, 128, 255, cv2.THRESH_BINARY)

    # Label the buildings
    num_labels_ground_truth, labels_ground_truth = cv2.connectedComponents(binary_ground_truth)
    num_labels_prediction, labels_prediction = cv2.connectedComponents(binary_prediction)
    
    # Create an output image with 3 channels (for BGR)
    output_image = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)

    # Define colors in BGR
    ground_truth_color = (0, 255, 255)  # Yellow for ground truth
    non_conjoined_color = (255, 0, 0)  # Blue for non-conjoined predictions
    conjoined_color = (255, 255, 0)  # Cyan for conjoined predictions
    false_positive_color = (0, 0, 255)  # Red for false positives

    # Color the ground truth buildings in yellow
    for label_gt in range(1, num_labels_ground_truth):
        gt_building_mask = (labels_ground_truth == label_gt)
        output_image[gt_building_mask] = ground_truth_color

    # Color the prediction buildings
    for label_pred in range(1, num_labels_prediction):
        pred_building_mask = (labels_prediction == label_pred)
        overlapping_gt_labels = np.unique(labels_ground_truth[pred_building_mask])
        
        # Check if the prediction is a false positive
        if len(overlapping_gt_labels) == 1 and overlapping_gt_labels[0] == 0:
            # Color the false positive in red
            output_image[pred_building_mask] = false_positive_color
        else:
            # Check if the prediction is conjoined
            is_conjoined = len(overlapping_gt_labels) - 1 > 1  # Subtract 1 for the background label
            # Color the building in cyan if it's conjoined, otherwise in blue
            color = conjoined_color if is_conjoined else non_conjoined_color
            output_image[pred_building_mask] = color

    return binary_ground_truth, binary_prediction, output_image


def load_model(checkpoint_path, model_type, device):
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint (.pth file)
        model_type: Type of model ('buildformer', 'unet', 'segformer')
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode
    """
    if model_type == 'buildformer':
        model = DualHeadBuildFormer(
            in_channels=3,
            out_channels_mask=1,
            out_channels_contour=1
        )
    elif model_type == 'unet':
        model = UNet(
            in_channels=3,
            out_channels_mask=1,
            out_channels_contour=1
        )
    elif model_type == 'segformer':
        model = DualHeadSegFormer(
            model_name='nvidia/mit-b0',  # Default backbone
            num_labels=1
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            # Full training checkpoint
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            # Alternative format
            state_dict = checkpoint['state_dict']
        else:
            # Direct state dict
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"✓ Loaded {model_type} model from {checkpoint_path}")
    return model


def predict_image(model, image_path, device, model_type):
    """
    Run inference on a single image.
    
    Args:
        model: Trained model
        image_path: Path to input image
        device: Device to run inference on
        model_type: Type of model
        
    Returns:
        prediction: Binary mask prediction (numpy array, 0-255)
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Preprocess based on model type
    if model_type == 'segformer':
        # Use HuggingFace processor for SegFormer
        processor = SegformerImageProcessor.from_pretrained('nvidia/mit-b0')
        inputs = processor(images=image, return_tensors='pt')
        image_tensor = inputs['pixel_values'].to(device)
    else:
        # Use standard transforms for UNet/BuildFormer
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        if model_type == 'buildformer':
            mask_logits, _ = model(image_tensor)
            prediction = torch.sigmoid(mask_logits)
        elif model_type == 'unet':
            mask_logits, _ = model(image_tensor)
            prediction = torch.sigmoid(mask_logits)
        elif model_type == 'segformer':
            mask_logits, _ = model(image_tensor)
            prediction = torch.sigmoid(mask_logits)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    # Convert to numpy (0-1 range from sigmoid)
    prediction = prediction.squeeze().cpu().numpy()
    
    # Normalize to full 0-1 range for better contrast
    pred_min = prediction.min()
    pred_max = prediction.max()
    if pred_max > pred_min:
        prediction = (prediction - pred_min) / (pred_max - pred_min)
    
    # Convert to 0-255
    prediction = (prediction * 255).astype(np.uint8)
    
    return prediction


def main():
    parser = argparse.ArgumentParser(description='Test model and create color-coded visualizations')
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--model_type', type=str, required=True, 
                        choices=['buildformer', 'unet', 'segformer'],
                        help='Type of model')
    
    # Data parameters
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Name of dataset (e.g., massachusetts, whu, inria)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root directory containing datasets')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save predictions and visualizations')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binarizing predictions (0-1)')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, args.model_type, device)
    
    # Setup paths
    dataset_dir = os.path.join(args.data_dir, args.dataset_name)
    test_img_dir = os.path.join(dataset_dir, 'test')
    test_mask_dir = os.path.join(dataset_dir, 'test_labels')
    
    # Create output directories
    pred_dir = os.path.join(args.output_dir, 'predictions')
    colored_dir = os.path.join(args.output_dir, 'color_coded')
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(colored_dir, exist_ok=True)
    
    print(f"\nTest images directory: {test_img_dir}")
    print(f"Test masks directory: {test_mask_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"  - Predictions: {pred_dir}")
    print(f"  - Color-coded: {colored_dir}\n")
    
    # Get test image paths
    image_extensions = ['png', 'jpg', 'jpeg', 'tif', 'tiff']
    test_img_paths = find_files_with_extensions(test_img_dir, image_extensions)
    test_mask_paths = find_files_with_extensions(test_mask_dir, image_extensions)
    
    if len(test_img_paths) == 0:
        print(f"ERROR: No test images found in {test_img_dir}")
        return
    
    print(f"Found {len(test_img_paths)} test images")
    print(f"Using threshold: {args.threshold} ({int(args.threshold * 255)}/255)\n")
    
    # Process each test image
    for img_idx, img_path in enumerate(tqdm(test_img_paths, desc="Processing test images")):
        # Get image name
        img_name = os.path.basename(img_path)
        img_base = os.path.splitext(img_name)[0]
        
        # Find corresponding ground truth mask
        gt_path = None
        for mask_path in test_mask_paths:
            if os.path.splitext(os.path.basename(mask_path))[0] == img_base:
                gt_path = mask_path
                break
        
        if gt_path is None:
            print(f"Warning: No ground truth found for {img_name}, skipping...")
            continue
        
        # Generate prediction (returns float in 0-255 range)
        prediction = predict_image(model, img_path, device, args.model_type)
        
        # Debug: Print prediction statistics for first few images
        if img_idx < 3:
            print(f"\n{img_name} - Prediction stats:")
            print(f"  Min: {prediction.min()}, Max: {prediction.max()}, Mean: {prediction.mean():.2f}")
        
        # Save raw prediction (grayscale)
        pred_save_path = os.path.join(pred_dir, f"{img_base}_pred.png")
        cv2.imwrite(pred_save_path, prediction)
        
        # Apply threshold to get binary mask
        threshold_value = int(args.threshold * 255)  # Convert 0-1 to 0-255
        _, prediction_binary = cv2.threshold(prediction, threshold_value, 255, cv2.THRESH_BINARY)
        
        # Debug: Print how many pixels are above threshold
        if img_idx < 3:
            pixels_above = (prediction_binary == 255).sum()
            total_pixels = prediction_binary.size
            print(f"  Pixels above threshold: {pixels_above}/{total_pixels} ({100*pixels_above/total_pixels:.1f}%)")
        
        # Load ground truth
        ground_truth = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize prediction to match ground truth if needed
        if prediction_binary.shape != ground_truth.shape:
            prediction_binary = cv2.resize(prediction_binary, (ground_truth.shape[1], ground_truth.shape[0]))
        
        # Create color-coded visualization
        try:
            binary_gt, binary_pred, colored_image = calculate_and_color_buildings(ground_truth, prediction_binary)
            
            # Save color-coded visualization
            colored_save_path = os.path.join(colored_dir, f"{img_base}_colored.png")
            cv2.imwrite(colored_save_path, colored_image)
            
            # Optionally save binary masks as well
            binary_gt_path = os.path.join(colored_dir, f"{img_base}_gt_binary.png")
            binary_pred_path = os.path.join(colored_dir, f"{img_base}_pred_binary.png")
            cv2.imwrite(binary_gt_path, binary_gt)
            cv2.imwrite(binary_pred_path, binary_pred)
            
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            continue
    
    print(f"\n✓ Processing complete!")
    print(f"  - Raw predictions saved to: {pred_dir}")
    print(f"  - Color-coded visualizations saved to: {colored_dir}")
    print(f"\nColor legend:")
    print(f"  - Yellow: Ground truth buildings")
    print(f"  - Blue: Correctly predicted non-conjoined buildings")
    print(f"  - Cyan: Conjoined building predictions (merged buildings)")
    print(f"  - Red: False positive predictions")


if __name__ == '__main__':
    main()
