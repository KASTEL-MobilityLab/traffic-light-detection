import os
import shutil
import argparse
from tqdm import tqdm

def oversample_images(image_folder, label_folder, to_oversample, oversample_factor):
    """
    Oversamples images for specific classes.

    Args:
        image_folder (str): Path to the folder containing images.
        label_folder (str): Path to the folder containing labels.
        to_oversample (list): List of classes (as integers) to oversample.
        oversample_factor (int): Factor by which to oversample the specified classes.

    The function checks each label file in the label_folder to see if it contains any
    classes specified in to_oversample. If so, it copies the corresponding image and
    label files 'oversample_factor' times, creating new image and label files with
    suffixes '_new0', '_new1', '_new2', ..., '_new(oversample_factor - 1)' appended to their names.
    """

    for label_file in tqdm(os.listdir(label_folder), desc='Processing label files'):
        label_path = os.path.join(label_folder, label_file)
        
        # Check if any line in the label file contains an item to oversample
        with open(label_path, 'r') as file:
            lines = file.readlines()
            oversample_needed = any(int(line.split(" ")[0]) in to_oversample for line in lines)
        
        if oversample_needed:
            image_name = label_file.split('.')[0] + '.jpg'
            image_path = os.path.join(image_folder, image_name)
            
            for j in range(oversample_factor):
                new_name = f"{label_file.split('.')[0]}_new{j}"
                new_image_path = os.path.join(image_folder, f"{new_name}.jpg")
                new_label_path = os.path.join(label_folder, f"{new_name}.txt")
                
                # Copy image and label file
                shutil.copy(image_path, new_image_path)
                shutil.copy(label_path, new_label_path)

if __name__ == "__main__":
    default_to_oversample = [2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    parser = argparse.ArgumentParser(description='Oversample images for specific classes.')
    parser.add_argument('--image_folder', type=str, help='Path to the folder containing images')
    parser.add_argument('--label_folder', type=str, help='Path to the folder containing labels')
    parser.add_argument('--to_oversample', nargs='+', type=int, default=default_to_oversample, help='List of classes to oversample (default: {})'.format(default_to_oversample))
    parser.add_argument('--oversample_factor', type=int, default=4, help='Oversampling factor for specified classes (default: 4)')
    args = parser.parse_args()

    if args.image_folder is None or args.label_folder is None:
        parser.error("Please provide both image_folder and label_folder.")

    oversample_images(args.image_folder, args.label_folder, args.to_oversample, args.oversample_factor)
