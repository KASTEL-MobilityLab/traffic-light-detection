from __future__ import print_function

import json
import argparse
import logging
import sys
import numpy as np
import os
import shutil
import glob
from tqdm import tqdm

# Set numpy print options to suppress scientific notation
np.set_printoptions(suppress=True)

# Configure logging settings
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: " "%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_folder", help="Folder where to store the final dataset. Placed inside /workspace/traffic-light-detection/datasets Should be an empty or non-existing folder.", type=str, required=True)
    parser.add_argument("--data_path", help="Path to the jpg files. The specified folder should contain a train and a test subfolder, created from 'convert_tif.py'.", type=str, required=True)
    parser.add_argument("--label_path", help="Path to the label data", type=str, required=False, default="/data/DTLD/v2.0")
    parser.add_argument('-at', "--arrow_threshold", help="A bbox for an arrow traffic light is removed from the labels, if there is < threshold percentage of it inside of the image.", type=float, required=False, default=0.9)
    parser.add_argument('-ct', "--circle_threshold", help="A bbox for a circle traffic light is removed from the labels, if there is < threshold percentage of it inside of the image.", type=float, required=False, default=0.8)
    parser.add_argument('--small_tl_threshold', help="Set a threshold for the width in pixels. Traffic lights with bboxes up to this width are declared as small, and only the color is detected, not the pictogram", required=False, type=int, default=11)
    parser.add_argument('--extra_label_small_tl', help="Whether to use a separate label for small traffic lights.", required=False, type=bool, default=False)
    return parser.parse_args()

def convert_labels(build_path:str, label_path:str, arrow_threshold:int, circle_threshold:int, small_tl_threshold:int, extra_label_small_tl:bool, data_path: str):
    """
    Convert all the .json files to .txt files containing the label of the images in the correct YOLO format.
    """
    # Define classes for traffic light labels
    all_classes = ['circle_green', 'circle_red', 'off', 'circle_red_yellow', 'arrow_left_green', 'circle_yellow', 'arrow_right_red', 'arrow_left_red', 'arrow_straight_red', 'arrow_left_red_yellow', 'arrow_left_yellow', 'arrow_straight_yellow', 'arrow_right_red_yellow', 'arrow_right_green', 'arrow_right_yellow', 'arrow_straight_green', 'arrow_straight_left_green', 'arrow_straight_red_yellow', 'arrow_straight_left_red', 'arrow_straight_left_yellow']

    # Iterate through train and test splits
    for split in ["train", "test"]:
        save_path = f"{build_path}/{split}/labels" # Define paths to store the labels
        f = open(os.path.join(label_path, f"DTLD_{split}.json"))
        label_file = json.load(f)
        f.close()
        label_list = label_file["images"]

        # Initialize counters
        n_corrected_labels = 0
        n_valid_labels = 0
        n_irrelevant_labels = 0
        n_unknown_labels = 0
        n_removed_labels = 0
        n_labels = 0
        n_removed_images = 0

        # Load the list of poorly labeled images from the JSON file
        json_file = open("/workspace/traffic-light-detection/preprocessing/poor_labeled_images.json")
        poor_labeled_images = json.load(json_file)
        all_images = os.listdir(f"{data_path}/{split}")

        # Iterate through each image in the label list
        for label_entry in tqdm(label_list, desc=f"converting labels for {split}"):
            filename = label_entry["image_path"].split("/")[-1].split(".")[0]
            if filename not in poor_labeled_images and f"{filename}.jpg" in all_images:
                print_buffer = [] # Store the final labels and bbox here
                remove_image = False # Flag to remove the complete labels for the image

                # Iterate through each label in the image
                for label in label_entry["labels"]:
                    remove_label = False
                    n_labels += 1

                    # Flag for filtering out labels labeled as unknown
                    if label["attributes"]["state"] == "unknown" or label["attributes"]["pictogram"] == "unknown":
                        n_unknown_labels += 1
                    else:
                        if label["attributes"]["aspects"] != "one_aspects" and label["attributes"]["direction"] == "front" and label["attributes"]["occlusion"] == "not_occluded":
                            if extra_label_small_tl:
                                if label["w"] <= small_tl_threshold:
                                    label_class = f'small_{label["attributes"]["state"]}'
                                else:
                                    label_class = f'normal_{label["attributes"]["pictogram"]}_{label["attributes"]["state"]}'
                            else:
                                if label["w"] <= small_tl_threshold:
                                    label_class = f'circle_{label["attributes"]["state"]}'
                                else:
                                    label_class = f'{label["attributes"]["pictogram"]}_{label["attributes"]["state"]}'

                            if label["attributes"]["pictogram"] in ["tram", "pedestrian", "bicycle", "pedestrian_bicycle"]:
                                n_irrelevant_labels += 1
                            else:
                                if "off" in label_class:
                                    label_class = "off"
                                image_width, image_height = 2048, 1024
                                h = label["h"]
                                w = label["w"]
                                x = label["x"]
                                y = label["y"]
                                x2 = x + w
                                y2 = y + h

                                # Check if the bounding box is inside of the picture boundaries
                                if (y < 0 or y2 > (image_height - 1) or x < 0 or x2 > (image_width - 1)):
                                    bbox_area = h * w
                                    bbox_area_picture = (min(y2,image_height-1) - max(y,0)) * (min(x2,image_width-1) - max(x,0))
                                    percentage = bbox_area_picture / bbox_area

                                    # Correct bbox if it lays in tolerance area
                                    if ("circle" in label_class and percentage > circle_threshold) or ("arrow" in label_class and percentage > arrow_threshold):
                                        x = max(x,0)
                                        y2 = min(y2,image_height-1) 
                                        y = max(y,0)
                                        x2 = min(x2,image_width-1)
                                        x = max(x,0)
                                        h = y2 - y
                                        w = x2 - x
                                        n_corrected_labels += 1
                                    else:  
                                        remove_label = True
                                else:
                                    n_valid_labels += 1
                                if remove_label:
                                    n_removed_labels += 1
                                    remove_image = True
                               
                                else:
                                    mapping = {"not_relevant": 0, "relevant": 1}
                                    rel = label["attributes"]["relevance"]
                                    rel = mapping[rel]
                                    class_name_to_id_mapping = {i: j for j, i in enumerate(all_classes)}
                                    class_id = class_name_to_id_mapping[label_class]
                                    b_center_x = (x + x2) / 2 
                                    b_center_y = (y + y2) / 2
                                    b_width    = w
                                    b_height   = h

                                    # Normalize the co-ordinates by the dimensions of the image
                                    b_center_x /= image_width
                                    b_center_y /= image_height
                                    b_width    /= image_width
                                    b_height   /= image_height

                                    if w > 4:
                                        print_buffer.append(f"{class_id} {b_center_x} {b_center_y} {b_width} {b_height}")

                if remove_image:
                    n_removed_images += 1

                # Save label file if there is at least one label
                if len (print_buffer) > 0:
                    filename = label_entry["image_path"].split("/")[-1].split(".")[0]
                    with open(f'{save_path}/{filename}.txt', 'w') as filehandle:
                        for line in print_buffer:
                            filehandle.write(("".join(line) + "\n"))

        # Logging at split level
        logging.info(f"""
                    Execution information for {split} folder :\n 
                    Originally contained labels : {n_labels} \n
                    Amount of unknown labels: {n_unknown_labels} \n
                    Amount of irrelevant labels: {n_irrelevant_labels} \n
                    Amount of removed labels: {n_removed_labels} \n
                    Amount of valid labels: {n_valid_labels} \n
                    Amount of corrected labels: {n_corrected_labels} \n
                    Amount of removed images because of removed label: {n_removed_images} \n
                    """)

    logging.info(f"There are {len(all_classes)} classes: {all_classes}")

def copy_images(build_path:str, data_path:str):
    """
    Copy the images for which a label file exists to the train / test folder in the build_path directory.
    """
    for split in ["train", "test"]:
        for file in os.listdir(f"{build_path}/{split}/labels"):
            id = file.split('.')[0]
            src = f"{data_path}/{split}/{id}.jpg"
            dst = f"{build_path}/{split}/images/{id}.jpg"
            shutil.copyfile(src, dst)
        n_original_files = len(glob.glob(f"{data_path}/{split}/*.jpg"))
        n_copies = len(glob.glob(f"{build_path}/{split}/images/*.jpg"))
        logging.info(f"Copied {n_copies} of {n_original_files} images from {split} set.")

def main(args):
    """
    Main function to execute the conversion process.
    """
    data_path = args.data_path
    if not (os.path.exists(f"{data_path}/train") and os.path.exists(f"{data_path}/test")):
        raise Exception("Invalid data path. Make sure to provide the data as described at the --data_path")

    label_path = args.label_path
    if not os.path.exists(label_path):
        raise Exception("Invalid label path.")
    
    build_path = os.path.join('/workspace/traffic-light-detection/datasets', args.target_folder)
    if os.path.exists(build_path):
        if input("Folder already exists. Are you sure to overwrite the data? Please type yes. \n")!='yes':
            raise Exception("Stopping Execution! Please Change build_name !")
            
    if os.path.exists(build_path):
        shutil.rmtree(build_path)

    arrow_threshold = args.arrow_threshold
    circle_threshold = args.circle_threshold
    small_tl_threshold = args.small_tl_threshold
    extra_label_small_tl = args.extra_label_small_tl

    os.makedirs(f"{build_path}/train/labels", exist_ok=True)
    os.makedirs(f"{build_path}/test/labels", exist_ok=True)
    convert_labels(build_path, label_path, arrow_threshold, circle_threshold, small_tl_threshold, extra_label_small_tl, data_path)

    os.makedirs(f"{build_path}/train/images", exist_ok=True)
    os.makedirs(f"{build_path}/test/images", exist_ok=True)
    copy_images(build_path, data_path)

if __name__ == "__main__":
    main(parse_args())
