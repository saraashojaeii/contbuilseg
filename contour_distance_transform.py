import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import argparse
import os
from pathlib import Path


def create_contour_distance_transform(binary_mask, saturation_value=255, invert=True):
    """
    Generate inverted saturated distance transform of contours from a binary mask.
    
    Args:
        binary_mask (numpy.ndarray): Binary mask image (0s and 255s)
        saturation_value (int): Maximum value for saturation (default: 255)
        invert (bool): Whether to invert the distance transform (default: True)
    
    Returns:
        numpy.ndarray: Inverted saturated distance transform image
    """
    # Ensure binary mask is properly formatted
    if binary_mask.dtype != np.uint8:
        binary_mask = binary_mask.astype(np.uint8)
    
    # Convert to binary (0 and 1)
    binary = (binary_mask > 127).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create contour mask
    contour_mask = np.zeros_like(binary)
    cv2.drawContours(contour_mask, contours, -1, 1, thickness=1)
    
    # Compute distance transform from contours
    # Distance transform gives distance to nearest non-zero pixel
    distance_transform = distance_transform_edt(1 - contour_mask)
    
    # Saturate the distance transform
    saturated_dt = np.clip(distance_transform, 0, saturation_value)
    
    # Normalize to 0-255 range
    if saturated_dt.max() > 0:
        saturated_dt = (saturated_dt / saturated_dt.max() * 255).astype(np.uint8)
    else:
        saturated_dt = saturated_dt.astype(np.uint8)
    
    # Invert if requested
    if invert:
        saturated_dt = 255 - saturated_dt
    
    return saturated_dt


def load_and_process_image(image_path, saturation_value=255, show_results=True):
    """
    Load an image and process it to generate contour distance transform.
    
    Args:
        image_path (str): Path to the binary mask image
        saturation_value (int): Maximum value for saturation
        show_results (bool): Whether to display the results
    
    Returns:
        tuple: (original_mask, contour_distance_transform)
    """
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Generate contour distance transform
    result = create_contour_distance_transform(img, saturation_value)
    
    if show_results:
        # Display results
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title('Original Binary Mask')
        axes[0].axis('off')
        
        axes[1].imshow(result, cmap='gray')
        axes[1].set_title('Inverted Saturated Distance Transform')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return img, result


def create_sample_mask():
    """
    Create a sample binary mask for demonstration.
    
    Returns:
        numpy.ndarray: Sample binary mask
    """
    # Create a sample binary mask with some shapes
    mask = np.zeros((200, 200), dtype=np.uint8)
    
    # Add some rectangles and circles
    cv2.rectangle(mask, (50, 50), (100, 100), 255, -1)
    cv2.circle(mask, (150, 80), 30, 255, -1)
    cv2.rectangle(mask, (30, 130), (80, 180), 255, -1)
    cv2.circle(mask, (140, 150), 25, 255, -1)
    
    return mask


def process_directory(input_dir, output_dir, saturation_value=50):
    """
    Process all images in input directory and save results to output directory.
    
    Args:
        input_dir (str): Path to input directory containing binary masks
        output_dir (str): Path to output directory for results
        saturation_value (int): Saturation value for distance transform
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Supported image extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    
    # Find all image files in input directory
    image_files = [f for f in input_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} image files to process...")
    
    processed_count = 0
    for image_file in image_files:
        try:
            # Load image
            img = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Could not load {image_file.name}")
                continue
            
            # Generate contour distance transform
            result = create_contour_distance_transform(img, saturation_value)
            
            # Save result with same filename
            output_file = output_path / image_file.name
            cv2.imwrite(str(output_file), result)
            
            processed_count += 1
            print(f"Processed: {image_file.name} -> {output_file.name}")
            
        except Exception as e:
            print(f"Error processing {image_file.name}: {e}")
    
    print(f"\nCompleted! Processed {processed_count}/{len(image_files)} images.")


def main():
    parser = argparse.ArgumentParser(description='Generate inverted saturated distance transform of contours')
    parser.add_argument('--input', '-i', type=str, help='Input binary mask image path or directory')
    parser.add_argument('--output', '-o', type=str, help='Output image path or directory')
    parser.add_argument('--input_dir', type=str, help='Input directory containing binary masks')
    parser.add_argument('--output_dir', type=str, help='Output directory for results')
    parser.add_argument('--saturation', '-s', type=int, default=50, 
                       help='Saturation value for distance transform (default: 50)')
    parser.add_argument('--demo', action='store_true', 
                       help='Run with demo sample mask')
    
    args = parser.parse_args()
    
    # Check for directory processing
    if args.input_dir and args.output_dir:
        process_directory(args.input_dir, args.output_dir, args.saturation)
        return
    
    # Check if input is a directory (for backward compatibility)
    if args.input and os.path.isdir(args.input):
        if not args.output:
            print("Error: Output directory must be specified when input is a directory")
            return
        process_directory(args.input, args.output, args.saturation)
        return
    
    if args.demo or not args.input:
        print("Running with demo sample mask...")
        # Create and use sample mask
        sample_mask = create_sample_mask()
        result = create_contour_distance_transform(sample_mask, args.saturation)
        
        # Display results
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].imshow(sample_mask, cmap='gray')
        axes[0].set_title('Sample Binary Mask')
        axes[0].axis('off')
        
        axes[1].imshow(result, cmap='gray')
        axes[1].set_title('Inverted Saturated Distance Transform')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        if args.output:
            cv2.imwrite(args.output, result)
            print(f"Result saved to {args.output}")
    
    else:
        # Process single input image
        try:
            original, result = load_and_process_image(args.input, args.saturation)
            
            if args.output:
                cv2.imwrite(args.output, result)
                print(f"Result saved to {args.output}")
                
        except Exception as e:
            print(f"Error processing image: {e}")


if __name__ == "__main__":
    main()
