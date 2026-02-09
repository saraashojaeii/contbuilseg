"""
Generate mask predictions for test images using a trained model.
No ground truth required - only generates and saves predictions.
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
            num_labels=1
        )
    elif model_type == 'unet':
        model = UNet(
            in_channels=3,
            out_channels_mask=1,
            out_channels_contour=1
        )
    elif model_type == 'segformer':
        model = DualHeadSegFormer(
            pretrained_model_name='nvidia/mit-b0',  # Default backbone
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
    original_size = image.size  # (width, height)
    
    # Preprocess based on model type
    if model_type == 'segformer':
        # Use HuggingFace processor for SegFormer (do_resize=False to keep original size)
        processor = SegformerImageProcessor.from_pretrained('nvidia/mit-b0')
        inputs = processor(images=image, return_tensors='pt', do_resize=False)
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
        elif model_type == 'unet':
            mask_logits, _ = model(image_tensor)
        elif model_type == 'segformer':
            # SegFormer wrapper (DualHeadSegFormer) - use positional argument
            mask_logits, _ = model(image_tensor)
            # SegFormer output may be at different resolution - interpolate to original size
            output_size = mask_logits.shape[-2:]
            target_size = (original_size[1], original_size[0])  # (height, width)
            if output_size != target_size:
                # Debug: print first time
                import sys
                if not hasattr(sys, '_segformer_resize_printed'):
                    print(f"SegFormer output size {output_size} != target {target_size}, interpolating...")
                    sys._segformer_resize_printed = True
                mask_logits = torch.nn.functional.interpolate(
                    mask_logits, 
                    size=target_size,
                    mode='bilinear', 
                    align_corners=False
                )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        prediction = torch.sigmoid(mask_logits)
    
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
    parser = argparse.ArgumentParser(description='Generate mask predictions for test images (no ground truth required)')
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--model_type', type=str, required=True, 
                        choices=['buildformer', 'unet', 'segformer'],
                        help='Type of model')
    
    # Data parameters
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Directory containing test images')
    
    # Output parameters
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
    test_img_dir = args.test_dir
    
    # Create output directory in current working directory
    output_dir = os.path.join(os.getcwd(), 'predictions')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nTest images directory: {test_img_dir}")
    print(f"Output directory: {output_dir}\n")
    
    # Get test image paths
    image_extensions = ['png', 'jpg', 'jpeg', 'tif', 'tiff']
    test_img_paths = find_files_with_extensions(test_img_dir, image_extensions)
    
    if len(test_img_paths) == 0:
        print(f"ERROR: No test images found in {test_img_dir}")
        return
    
    print(f"Found {len(test_img_paths)} test images")
    print(f"Using threshold: {args.threshold} ({int(args.threshold * 255)}/255)\n")
    
    # Process each test image
    for img_idx, img_path in enumerate(tqdm(test_img_paths, desc="Generating predictions")):
        # Get image name
        img_name = os.path.basename(img_path)
        img_base = os.path.splitext(img_name)[0]
        
        # Generate prediction (returns float in 0-255 range)
        prediction = predict_image(model, img_path, device, args.model_type)
        
        # Debug: Print prediction statistics for first few images
        if img_idx < 3:
            print(f"\n{img_name} - Prediction stats:")
            print(f"  Min: {prediction.min()}, Max: {prediction.max()}, Mean: {prediction.mean():.2f}")
        
        # Save grayscale prediction (probability map)
        pred_gray_path = os.path.join(output_dir, f"{img_base}_prob.png")
        cv2.imwrite(pred_gray_path, prediction)
        
        # Apply threshold to get binary mask
        threshold_value = int(args.threshold * 255)  # Convert 0-1 to 0-255
        _, prediction_binary = cv2.threshold(prediction, threshold_value, 255, cv2.THRESH_BINARY)
        
        # Debug: Print how many pixels are above threshold
        if img_idx < 3:
            pixels_above = (prediction_binary == 255).sum()
            total_pixels = prediction_binary.size
            print(f"  Pixels above threshold: {pixels_above}/{total_pixels} ({100*pixels_above/total_pixels:.1f}%)")
        
        # Save binary prediction
        pred_binary_path = os.path.join(output_dir, f"{img_base}_mask.png")
        cv2.imwrite(pred_binary_path, prediction_binary)
    
    print(f"\n✓ Processing complete!")
    print(f"  - Predictions saved to: {output_dir}")
    print(f"\nOutput files per image:")
    print(f"  - *_prob.png: Grayscale probability map (0-255)")
    print(f"  - *_mask.png: Binary mask (thresholded at {args.threshold})")


if __name__ == '__main__':
    main()
