#!/usr/bin/env python3
"""
Simple script to remove the top N rows from an image.
"""

import argparse
from PIL import Image
import numpy as np
from pathlib import Path


def crop_top_rows(image_path, rows_to_remove=120, output_path=None):
    """
    Remove the top N rows from an image.
    
    Args:
        image_path: Path to input image
        rows_to_remove: Number of rows to remove from top (default: 120)
        output_path: Path to save cropped image (default: adds '_cropped' to filename)
    
    Returns:
        Path to saved image
    """
    # Load image
    img = Image.open(image_path)
    width, height = img.size
    
    print(f"Original image size: {width}×{height}")
    
    # Check if we're removing more rows than the image has
    if rows_to_remove >= height:
        raise ValueError(f"Cannot remove {rows_to_remove} rows from image with height {height}")
    
    # Crop image: (left, top, right, bottom)
    # We want to keep from row 'rows_to_remove' to the bottom
    cropped_img = img.crop((0, rows_to_remove, width, height))
    
    new_width, new_height = cropped_img.size
    print(f"Cropped image size: {new_width}×{new_height}")
    print(f"Removed {rows_to_remove} rows from top")
    
    # Determine output path
    if output_path is None:
        input_path = Path(image_path)
        output_path = input_path.parent / f"{input_path.stem}_cropped{input_path.suffix}"
    
    # Save cropped image
    cropped_img.save(output_path)
    print(f"Saved to: {output_path}")
    
    return output_path


def crop_top_rows_batch(input_dir, rows_to_remove=120, output_dir=None):
    """
    Crop top rows from all images in a directory.
    
    Args:
        input_dir: Directory containing images
        rows_to_remove: Number of rows to remove from top
        output_dir: Directory to save cropped images (default: creates 'cropped' subdirectory)
    """
    input_path = Path(input_dir)
    
    # Create output directory
    if output_dir is None:
        output_path = input_path / "cropped"
    else:
        output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Supported image formats
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
    
    # Find all images
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f'*{ext}'))
        image_files.extend(input_path.glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    print(f"Output directory: {output_path}")
    print("-" * 60)
    
    # Process each image
    for img_file in image_files:
        print(f"\nProcessing: {img_file.name}")
        output_file = output_path / img_file.name
        try:
            crop_top_rows(img_file, rows_to_remove, output_file)
        except Exception as e:
            print(f"Error processing {img_file.name}: {e}")
    
    print("\n" + "=" * 60)
    print("Batch processing complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Remove top rows from image(s)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Remove top 120 rows from a single image
  python crop_top_rows.py --image photo.jpg
  
  # Remove top 200 rows from a single image
  python crop_top_rows.py --image photo.jpg --rows 200
  
  # Specify output filename
  python crop_top_rows.py --image photo.jpg --output result.jpg
  
  # Process all images in a directory
  python crop_top_rows.py --dir ./images --rows 120
  
  # Process directory with custom output location
  python crop_top_rows.py --dir ./images --output_dir ./cropped_images
        """
    )
    
    # Input options
    parser.add_argument('--image', type=str, help='Path to single image file')
    parser.add_argument('--dir', type=str, help='Path to directory containing images')
    
    # Crop options
    parser.add_argument('--rows', type=int, default=120, 
                        help='Number of rows to remove from top (default: 120)')
    
    # Output options
    parser.add_argument('--output', type=str, 
                        help='Output path for single image (default: adds _cropped to filename)')
    parser.add_argument('--output_dir', type=str,
                        help='Output directory for batch processing (default: creates "cropped" subdirectory)')
    
    args = parser.parse_args()
    
    # Validate input
    if not args.image and not args.dir:
        parser.error("Must specify either --image or --dir")
    
    if args.image and args.dir:
        parser.error("Cannot specify both --image and --dir")
    
    # Process single image
    if args.image:
        try:
            crop_top_rows(args.image, args.rows, args.output)
        except Exception as e:
            print(f"Error: {e}")
            return 1
    
    # Process directory
    if args.dir:
        try:
            crop_top_rows_batch(args.dir, args.rows, args.output_dir)
        except Exception as e:
            print(f"Error: {e}")
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main())