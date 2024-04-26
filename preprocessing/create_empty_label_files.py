import os
import argparse

def create_empty_txt_files(image_folder, label_folder):
    """
    Create empty label files for images without corresponding labels.

    Args:
        image_folder (str): Path to the folder containing images.
        label_folder (str): Path to the folder containing label files.
    """
    # If there is an image and no corresponding label file, create an empty one
    images = os.listdir(image_folder)
    labels = os.listdir(label_folder)

    for i in images:
        label_file = os.path.join(label_folder, f"{i.split('.')[0]}.txt")
        if label_file not in labels:
            open(label_file, 'a').close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create empty label files for images without corresponding labels.')
    parser.add_argument('--image_folder', type=str, help='Path to the folder containing images')
    parser.add_argument('--label_folder', type=str, help='Path to the folder containing label files')
    args = parser.parse_args()

    if args.image_folder is None or args.label_folder is None:
        parser.error("Please provide both image_folder and label_folder.")

    create_empty_txt_files(args.image_folder, args.label_folder)
