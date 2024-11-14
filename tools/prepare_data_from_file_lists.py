from datetime import datetime
import argparse
import json
import os

from PIL import Image


def find_file_with_extensions(base_dir, filename_prefix, base_name, extensions):
    """Helper function to find a file trying different extensions"""
    for ext in extensions:
        ext = ext if ext.startswith('.') else '.' + ext
        test_filename = filename_prefix + base_name + ext
        test_path = os.path.join(base_dir, test_filename)
        if os.path.exists(test_path):
            return test_path, test_filename
    return None, None


def create_coco_annotations(data_root, file_list, output_path, extensions, image_prefix="", gt_prefix=""):
    """
    Create COCO format annotations from image files listed in a text file
    
    Args:
        data_root: Root directory containing img/ and gt-dots/ folders
        file_list: Text file containing image filenames
        output_path: Path to save the COCO json file
        extensions: List of valid image extensions to look for
        image_prefix: Prefix for image filenames
        gt_prefix: Prefix for ground truth filenames
    """
    # Initialize COCO format structure
    coco_format = {
        "info": {
            "description": "Point Detection Dataset",
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "point",
                "supercategory": "none"
            }
        ]
    }

    annotation_id = 1
    
    # Read file list
    with open(file_list, 'r') as f:
        image_files = f.read().splitlines()
    
    # Process each image
    for base_name in image_files:
        image_id = len(coco_format["images"]) + 1
        
        # Find image file
        image_path, image_filename = find_file_with_extensions(
            os.path.join(data_root, "img"),
            image_prefix,
            base_name,
            extensions
        )
        
        # Find ground truth file
        gt_path, _ = find_file_with_extensions(
            os.path.join(data_root, "gt-dots"),
            gt_prefix,
            base_name,
            extensions
        )
                
        if image_path is None or gt_path is None:
            print(f"Warning: Missing files for {base_name}, skipping...")
            continue
            
        # Read image dimensions
        with Image.open(image_path) as img:
            width, height = img.size
            
        # Add image info
        coco_format["images"].append({
            "id": image_id,
            "file_name": image_filename,
            "width": width,
            "height": height,
            "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "license": 1,
            "coco_url": os.path.abspath(image_path),
            "flickr_url": ""
        })
        
        # Get point coordinates from ground truth image
        with Image.open(gt_path) as gt_img:
            gt_img = gt_img.convert('RGB')
            for x in range(width):
                for y in range(height):
                    if gt_img.getpixel((x, y)) == (255, 255, 255):
                        # Add annotation
                        coco_format["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": 1,
                            "bbox": [x, y, 0, 0],  # point
                            "area": 0,
                            "segmentation": [],
                            "iscrowd": 0
                        })
                        annotation_id += 1
    
    # Save COCO format json
    with open(output_path, 'w') as f:
        json.dump(coco_format, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Create COCO annotations from file list')
    parser.add_argument('--data_root', required=True, help='Root directory containing img/ and gt-dots/ folders')
    parser.add_argument('--image_prefix', default='beeType1_', help='Prefix to add to image filenames')
    parser.add_argument('--gt_prefix', default='dots', help='Prefix to add to ground truth filenames')
    parser.add_argument('--extensions', nargs='+', default=['png', 'jpg', 'jpeg', 'bmp'],
                        help='List of image extensions to look for (default: png jpg jpeg bmp)')
    
    args = parser.parse_args()
    
    # Process train, val and test splits
    splits = ['train', 'val', 'test']
    for split in splits:
        file_list = os.path.join(args.data_root, f"{split}_imgs.txt")
        output_path = os.path.join(args.data_root, f"{split}_coco.json")
        
        if os.path.exists(file_list):
            create_coco_annotations(args.data_root, file_list, output_path, args.extensions,
                                 args.image_prefix, args.gt_prefix)
            print(f"Conversion complete for {split}. COCO format annotations saved to {output_path}")
        else:
            print(f"Warning: {file_list} not found, skipping {split} split")

if __name__ == "__main__":
    main()