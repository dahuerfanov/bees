"""
A script to extract coordinates of white pixels from images and save them to text files.
This is useful for processing ground truth data where white pixels represent points of interest.
Each output text file contains x,y coordinates of white pixels, one coordinate pair per line.
"""

import argparse
import os

from PIL import Image


def image_to_white_coords(image_path, output_path):
    """
    Reads a PNG image, finds all white pixels, and saves their coordinates to a TXT file.
    
    :param image_path: Path to the input PNG image.
    :param output_path: Path to save the TXT file.
    """
    with Image.open(image_path) as img:
        # Ensure the image is in RGB mode (not necessary for PNG but good practice)
        img = img.convert('RGB')
        width, height = img.size
        
        # List to hold coordinates of white pixels
        white_pixels = []
        
        # Iterate over each pixel in the image
        for x in range(width):
            for y in range(height):
                # Check if the pixel is white (RGB: 255, 255, 255)
                if img.getpixel((x, y)) == (255, 255, 255):
                    white_pixels.append(f"{x},{y}")  # Store as comma-separated string
                    
        # Save the coordinates to a TXT file
        with open(output_path, 'w') as txt_file:
            txt_file.write('\n'.join(white_pixels))


def main(args):
    # Ensure the output folder exists
    os.makedirs(args.output_folder, exist_ok=True)

    for filename in os.listdir(args.input_folder):
        if filename.lower().endswith(tuple(args.extensions)):
            image_path = os.path.join(args.input_folder, filename)
            output_filename = f"{os.path.splitext(filename)[0]}.txt"  # Same name, different extension
            output_path = os.path.join(args.output_folder, output_filename)
            print(f"Processing {filename} -> {output_filename}")
            image_to_white_coords(image_path, output_path)
    print("All images processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract white pixel coordinates from PNG images.')
    parser.add_argument('-i', '--input_folder', required=True, help='Path to the folder containing PNG images.')
    parser.add_argument('-o', '--output_folder', required=True, help='Path to the folder where TXT files will be saved.')
    parser.add_argument('-e', '--extensions', nargs='+', default=['.png', '.jpeg', '.jpg', '.bmp'],
                        help='List of image extensions to process (default: .png .jpeg .jpg .bmp)')
    args = parser.parse_args()
    main(args)