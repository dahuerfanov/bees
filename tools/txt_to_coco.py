"""
A script to convert txt files containing point annotations to COCO format.
The txt files contain x,y coordinates of points, one coordinate pair per line.
"""

from datetime import datetime
import argparse
import json
import os


def create_coco_annotations(image_dir, label_dir, output_path, extensions, image_prefix="", label_prefix=""):
    """
    Convert point annotations from txt files to COCO format
    
    Args:
        image_dir: Directory containing the images
        label_dir: Directory containing the txt files with point coordinates
        output_path: Path to save the COCO json file
        extensions: List of valid image extensions to look for
        image_prefix: Prefix to add to image filenames when searching
        label_prefix: Prefix to strip from label filenames when searching for images
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
    
    # Process each txt file
    for txt_file in os.listdir(label_dir):
        if not txt_file.endswith('.txt'):
            continue
            
        image_id = len(coco_format["images"]) + 1
        # Strip label prefix and get base name
        if txt_file.startswith(label_prefix):
            txt_file = txt_file[len(label_prefix):]
        base_name = os.path.splitext(txt_file)[0]
        
        # Try each extension
        image_path = None
        image_filename = None
        for ext in extensions:
            ext = ext if ext.startswith('.') else '.' + ext
            test_filename = image_prefix + base_name + ext
            test_path = os.path.join(image_dir, test_filename)
            if os.path.exists(test_path):
                image_path = test_path
                image_filename = test_filename
                break
                
        # Skip if no matching image found
        if image_path is None:
            print(f"Warning: No image found for {txt_file} with extensions {extensions}, skipping...")
            continue
            
        # Read image dimensions
        from PIL import Image
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
        
        # Read point coordinates
        txt_path = os.path.join(label_dir, label_prefix + txt_file)
        with open(txt_path, 'r') as f:
            for line in f:
                x, y = map(int, line.strip().split(','))
                
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
    parser = argparse.ArgumentParser(description='Convert point annotations to COCO format')
    parser.add_argument('--image_dir', required=True, help='Directory containing the images')
    parser.add_argument('--label_dir', required=True, help='Directory containing the txt files with point coordinates')
    parser.add_argument('--output', required=True, help='Path to save the COCO format json file')
    parser.add_argument('--extensions', nargs='+', default=['png', 'jpeg', 'jpg', 'bmp'],
                        help='List of image extensions to look for (default: png jpeg jpg bmp)')
    parser.add_argument('--image_prefix', default='beeType1_', help='Prefix to add to image filenames when searching')
    parser.add_argument('--label_prefix', default='dots', help='Prefix to strip from label filenames when searching for images')
    
    args = parser.parse_args()
    
    create_coco_annotations(args.image_dir, args.label_dir, args.output, args.extensions, 
                          args.image_prefix, args.label_prefix)
    print(f"Conversion complete. COCO format annotations saved to {args.output}")


if __name__ == "__main__":
    main()

