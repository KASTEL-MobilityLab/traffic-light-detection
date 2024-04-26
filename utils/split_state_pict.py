import os
from PIL import Image
import argparse
from tqdm import tqdm

def create_dataset(input_dataset_path, build_path):
    """
    Create a dataset by processing label files and corresponding images.
    
    This function takes input_dataset_path containing label files and images, 
    and builds a new dataset at build_path. It parses the label files, 
    reformats them, and copies corresponding images to the new dataset, 
    creating necessary directories as required.
    
    Args:
        input_dataset_path (str): Path to the input dataset containing label files and images.
        build_path (str): Path where the new dataset will be created.
    """
    # Define label mappings
    l = ['circle+green', 'circle+red', 'off+off', 'circle+red_yellow', 'arrow_left+green', 'circle+yellow', 'arrow_right+red', 'arrow_left+red', 'arrow_straight+red', 'arrow_left+red_yellow', 'arrow_left+yellow', 'arrow_straight+yellow', 'arrow_right+red_yellow', 'arrow_right+green', 'arrow_right+yellow', 'arrow_straight+green', 'arrow_straight_left+green', 'arrow_straight+red_yellow', 'arrow_straight_left+red', 'arrow_straight_left+yellow']
    mapping = {ind: i for ind, i in enumerate(l)}

    l2 = ["green", "red", "yellow", "red_yellow", "off"]
    mapping2 = {i: ind for ind, i in enumerate(l2)}

    l3 = ["circle", "arrow_left", "arrow_right", "arrow_straight", "arrow_straight_left", "off"]
    mapping3 = {i: ind for ind, i in enumerate(l3)}

    # Create necessary directories
    for split in ["train", "test"]:
        folder = os.path.join(input_dataset_path, split, 'labels')
        folder_imgs = os.path.join(input_dataset_path, split, 'images')

        new_folder_labels = os.path.join(build_path, split, 'labels')
        new_folder_images = os.path.join(build_path, split, 'images')

        os.makedirs(new_folder_labels, exist_ok=True)
        os.makedirs(new_folder_images, exist_ok=True)

        # Process each label file with tqdm
        for file in tqdm(os.listdir(folder), desc=f'Processing {split} set'):
            img_name = f"{file[:-4]}.jpg"
            label_file = os.path.join(folder, file)

            print_buffer = []

            with open(label_file, "r") as f:
                # Parse each line in the label file
                for line in f:
                    # Parse label information
                    m = mapping[int(line.split(" ")[0])]
                    m = m.split("+")
                    color = m[1]
                    pict = m[0]

                    # Append formatted line to print buffer
                    print_buffer.append(f"{mapping2[color]} {mapping3[pict]} {' '.join(line.split(' ')[1:])}")

            # Write formatted lines to new label file
            with open(os.path.join(new_folder_labels, file), 'w') as filehandle:
                for line in print_buffer:
                    filehandle.write(line)

            # Copy corresponding image to new folder
            if os.path.exists(os.path.join(folder_imgs, img_name)):
                img = Image.open(os.path.join(folder_imgs, img_name)).convert("RGB")
                img.save(os.path.join(new_folder_images, img_name))
            else:
                # Handle case where image file extension is '.png'
                img_name = f"{file[:-4]}.png"
                img = Image.open(os.path.join(folder_imgs, img_name)).convert("RGB")
                img.save(os.path.join(new_folder_images, f"{file[:-4]}.jpg"))

def main():
    parser = argparse.ArgumentParser(description='Process input and target paths for dataset creation')
    parser.add_argument('--input_path', type=str, help='Path to input dataset')
    parser.add_argument('--target_path', type=str, help='Path where to store the splitted dataset')
    args = parser.parse_args()

    create_dataset(args.input_path, args.target_path)

if __name__ == "__main__":
    main()
