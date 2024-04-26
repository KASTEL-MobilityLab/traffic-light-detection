## imports
from __future__ import print_function
import argparse
import logging
import sys
import numpy as np
import os
import glob
from natsort import natsorted
from tqdm import tqdm
import cv2
import math
import pandas as pd
import pickle
import warnings
from ultralytics import YOLO
import json

# Suppress warnings
warnings.filterwarnings("ignore")

# Set print options for numpy
np.set_printoptions(suppress=True)

# Logging configuration
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: " "%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Define label names for traffic lights and road markings
tl_label_names = ['circle_green', 'circle_red', 'off', 'circle_red_yellow', 'arrow_left_green', 'circle_yellow',
                  'arrow_right_red', 'arrow_left_red', 'arrow_straight_red', 'arrow_left_red_yellow', 'arrow_left_yellow',
                  'arrow_straight_yellow', 'arrow_right_red_yellow', 'arrow_right_green', 'arrow_right_yellow',
                  'arrow_straight_green', 'arrow_straight_left_green', 'arrow_straight_red_yellow', 'arrow_straight_left_red',
                  'arrow_straight_left_yellow']
rm_label_names = ['straight', 'left', 'right', 'str_left', 'str_right']

# Function to find key by value containing a specific string in a dictionary
def find_key_by_value_containing_string(dictionary, search_string):
    for key, value in dictionary.items():
        if any(search_string in item for item in value):
            return key
    return None  # Return None if the string is not found in any list

# Unpack column from the yolo format to a more handy and flexible format.
def unpack_data(idx, raw_data, img)->dict:

    data = {}
    name = 'tl' + str(idx)
    label, x, y, w, h = map(float, raw_data.split(' '))
    label_name = tl_label_names[int(label)]
    
    data['name'] = name
    data['label_name'] = label_name
    data['label'] = int(label)

    imgHeight, imgWidth, _ = img.shape
    w = int(np.around(w * imgWidth))
    h = int(np.around(h * imgHeight))
    x = int(np.around(x * imgWidth - w / 2))
    y = int(np.around(y * imgHeight - h / 2))

    data['w'], data['h'], data['x'], data['y'] = w, h, x, y

    return data


# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rel_model", type=str, help="Path to the relevance model file", required=True)
    parser.add_argument("--tl_model", type=str, help="Path to the tl_detection_model file", required=True)
    parser.add_argument("--rm_model", type=str, help="Path to the rm_detection_model file", required=True)
    parser.add_argument("--imgHeight", type=int, default=1024, help="Image height (default: 1024)")
    parser.add_argument("--imgWidth", type=int, default=2048, help="Image width (default: 2048)")
    parser.add_argument("--data_path", help="Path with a sequence of labeled images", type=str, required=True)
    parser.add_argument("--use_dtld_label", help="Uses DLTL labels for tl if true, uses model for tl_detection if false", type=bool, default=False)
    return parser.parse_args()


# Function to draw bounding boxes on an image
def draw(img, df):
    """
    Drawing a cv2 image from the information contained in a dataframe.
    """

    imgHeight, imgWidth, _ = img.shape
    thick = int((imgHeight + imgWidth) // 1200)

    for _, row in df.iterrows():
        x, y, w, h = int(row['x']), int(row['y']), int(row['w']), int(row['h'])
        relevance = '1' if row['relevant_pred'] == 1 else '0'
        label = f'{row["name"]}; {row["label_name"]}; pred: {relevance}'

        if relevance == "1":
            color = (0, 255, 3)  # Green color for relevant predictions
        else:
            color = (255, 255, 255)  # White color for irrelevant predictions

        cv2.rectangle(img, (int(x), int(y)), (x + w, y + h), color, thick)

        if x > 1500:
            x -= 11 * len(label)
        cv2.putText(img, label, (x, y - 10), 0, 0.00075 * imgHeight, color, thick)


# Function to calculate distance from the center of the image to the bounding box center
def calculate_distance(row, imgWidth, imgHeight):
    center = [imgWidth / 2, imgHeight]
    bbox_center = [row['x'] + 0.5 * row['w'], row['y'] + 0.5 * row['h']]
    return math.dist(center, bbox_center)

# Function to calculate x-offset from the center of the image to the bounding box center
def calculate_x_offset(row, imgWidth, imgHeight):
    center = [imgWidth / 2, imgHeight]
    bbox_center = [row['x'] + 0.5 * row['w'], row['y'] + 0.5 * row['h']]
    return center[0] - bbox_center[0]

# Function to calculate height-width ratio of the bounding box
def calculate_hw_ratio(row):
    return row['h'] / row['w']

# Main function
def main(args):
    # Check input path and structure
    data_path = args.data_path
    if not (os.path.exists(data_path)):
        raise Exception("Invalid data path. Make sure to provide the data as described at the --data_path")
    
    # Check if model paths exist
    if not os.path.isfile(args.rel_model):
        print(f"Error: Model file '{args.rel_model}' does not exist.")
        return
    if not os.path.isfile(args.tl_model):
        print(f"Error: Model file '{args.tl_model}' does not exist.")
        return
    if not os.path.isfile(args.rm_model):
        print(f"Error: Model file '{args.rm_model}' does not exist.")
        return
    
    use_dtld_label = args.use_dtld_label
    
    # Load relevant model files
    rel_model = pickle.load(open(args.rel_model, 'rb'))
    tl_model = YOLO(args.tl_model)
    rm_model = YOLO(args.rm_model)

    # Set image dimensions (use provided values or defaults)
    imgHeight = args.imgHeight
    imgWidth = args.imgWidth
    
    test_folders = glob.glob(f'{data_path}/*')

    rel_direction_mapping = {0: ["straight"], 1: ["left", "str_left"], 2: ["right"]}

    # Iterate over all test folders
    for test_folder in tqdm(test_folders, desc="Overall progress", position=0):
        if os.path.isdir(test_folder):

            image_files = glob.glob(f'{test_folder}/images/*.jpg')
            image_files = natsorted(image_files)

            last_state = None

            for image_file in tqdm(image_files, desc="Test folder progress", position=1, leave=False):
                img = cv2.imread(image_file)
                entries = []

                filename = image_file.split('/')[-1].split('.')[0]

                # Load label data from dtld if activated
                if use_dtld_label:
                    labelfile = f'{filename}.txt'

                    # convert label data and add to data frame for traffic lights
                    tl_label_file = open(f'{test_folder}/dtld/{labelfile}')
                    tl_labels = tl_label_file.readlines()
                    tl_label_file.close()

                    for idx, tl_label in enumerate(tl_labels):

                        # Split string to float
                        entry = unpack_data(idx, tl_label, img)
                        if entry:
                            entries.append(entry)

                else:
                
                    # Convert label data predictions from model and add to data frame for traffic lights
                    tl_preds = tl_model.predict(image_file, verbose=False, imgsz=(1920, 1920), conf=0.3, iou=0.6, device=0,
                                                agnostic_nms=True)
                    for idx, box in enumerate(json.loads(tl_preds[0].tojson())):
                        box_dims = box['box']

                        data = {}
                        data['name'] = f"tl_{idx}"
                        data['label_name'] = box['name']
                        data["label"] = tl_label_names.index(box['name'])
                        data['w'] = int(box_dims['x2']) - int(box_dims['x1'])
                        data['h'] = int(box_dims['y2']) - int(box_dims['y1'])
                        data['x'] = int(box_dims['x1'])
                        data['y'] = int(box_dims['y1'])
                        entries.append(data)

                # Convert label data and add to data frame for road marking
                rm_preds = rm_model.predict(image_file, verbose=False, imgsz=(1920, 1920), conf=0.5, iou=0.6, device=0,
                                             agnostic_nms=True)
                for idx, box in enumerate(json.loads(rm_preds[0].tojson())):
                    box_dims = box['box']

                    data = {}
                    data['name'] = f"rm_{idx}"
                    data['label_name'] = box['name']
                    data["label"] = int(box['name'])
                    data['w'] = int(box_dims['x2']) - int(box_dims['x1'])
                    data['h'] = int(box_dims['y2']) - int(box_dims['y1'])
                    data['x'] = int(box_dims['x1'])
                    data['y'] = int(box_dims['y1'])
                    entries.append(data)

                if entries:
                    # Building a dataframe that contains all labels for one image
                    df = pd.DataFrame(entries)

                    # Remove tl with a width smaller than 5 px
                    df = df[df['w'] >= 5]

                    # Remove 'tl' entries that are far away compared to closer ones
                    df_tl = df[df['name'].str.contains('tl')]
                    max_width = df_tl['w'].max()
                    ratio = 2
                    df = df[~((df['name'].str.contains('tl')) & (df['w'] * ratio <= max_width))]

                    # Calculate features for the labels
                    df['distance'] = df.apply(lambda row: calculate_distance(row, imgWidth, imgHeight), axis=1)
                    df['x_offset'] = df.apply(lambda row: calculate_x_offset(row, imgWidth, imgHeight), axis=1)
                    df['hw_ratio'] = df.apply(lambda row: calculate_hw_ratio(row), axis=1)

                    # Do evaluation
                    df['relevant_pred'] = 0

                    # When all traffic lights have the same color
                    df_tl = df[df['name'].str.contains('tl')].copy()
                    df_rm = df[df['name'].str.contains('rm')]
                    df_tl = df_tl[df_tl["label_name"] != "off"]

                    if len(df_tl) != 0:
                        df_tl["color"] = ""
                        df_tl.loc[df_tl["label_name"].str.contains("green"), "color"] = "green"
                        df_tl.loc[df_tl["label_name"].str.contains("red"), "color"] = "red"
                        df_tl.loc[df_tl["label_name"].str.contains("yellow"), "color"] = "yellow"
                        df_tl.loc[df_tl["label_name"].str.contains("red_yellow"), "color"] = "red_yellow"
                        df_tl.loc[df_tl["label_name"].str.contains("off"), "color"] = "off"

                        df_tl["pict"] = df_tl.apply(lambda x: x['label_name'].split(x['color'])[0][:-1], axis=1)
                        df_tl["filtered_pict"] = df_tl.apply(lambda x: x['pict'].split('arrow_')[-1], axis=1)
                        if df_tl["pict"].nunique() == 1:
                            df_tl["relevant_pred"] = 1
                            if len(df_rm) > 0:
                                df_rm["relevant_pred"] = rel_model.predict(df_rm.iloc[:, 2:-1])

                        # When traffic lights have different colors but no markings were detected and no last state is available
                        elif df_tl["pict"].nunique() > 1 and len(df_rm) == 0 and last_state == None:
                            closest_tl_x_offset = df_tl["x_offset"].abs().min()
                            rel_pikt = df_tl.loc[df_tl["x_offset"].abs() == closest_tl_x_offset, "pict"].values[0]
                            df_tl.loc[df_tl["pict"] == rel_pikt, "relevant_pred"] = 1

                        # When traffic lights have different colors but no markings were detected but a last state is available
                        elif df_tl["pict"].nunique() > 1 and len(df_rm) == 0 and last_state != None:
                            rel_direction = last_state

                            if (rel_direction == 1 and df_tl['filtered_pict'].isin(rel_direction_mapping[1]).any()) or (
                                    rel_direction == 2 and df_tl['filtered_pict'].isin(rel_direction_mapping[2]).any()):
                                df_tl.loc[df_tl["filtered_pict"].isin(rel_direction_mapping[rel_direction]), "relevant_pred"] = 1
                            else:
                                df_tl.loc[df_tl["filtered_pict"] == "circle", "relevant_pred"] = 1

                        # When traffic lights have different colors and markings were detected
                        elif (len(df_tl) == 0 or df_tl["pict"].nunique() > 1) and len(df_rm) > 0:
                            x = df_rm.iloc[:, 2:-1]
                            preds = []
                            for ind, row in x.iterrows():
                                preds.append(rel_model.predict(np.array(row).reshape(1, -1))[0])

                            inds = np.where(np.array(preds) == 1)[0]
                            rel_markings = df_rm.iloc[inds, :]
                            rel_directions = (rel_markings.iloc[:, 2]).to_list()
                            if len(set(rel_directions)) > 1:
                                # Multiple different markings found
                                # Possible scenarios: curve, car driving between 2 lanes
                                if last_state != None and len(
                                        list(set(rel_direction_mapping[last_state]) & set((rel_markings.iloc[:, 2]).to_list()))) > 0:
                                    rel_direction = last_state
                                else:
                                    closest_rel_marking = rel_markings.loc[df["distance"].idxmin()]
                                    rel_direction = find_key_by_value_containing_string(rel_direction_mapping,
                                                                                          closest_rel_marking["label_name"])
                                    df_rm.loc[df_rm["label"] == rel_direction, "relevant_pred"] = 1
                            elif len(set(rel_directions)) == 1:
                                # Only one marking = that counts
                                rel_direction = find_key_by_value_containing_string(rel_direction_mapping,
                                                                                      str(rel_directions[0]))

                                df_rm.iloc[inds, -1] = 1

                            elif len(set(rel_directions)) == 0:
                                # No marking = straight ahead
                                if last_state != None:
                                    rel_direction = last_state
                                else:
                                    rel_direction = 0

                            if (rel_direction == 1 and df_tl['filtered_pict'].isin(rel_direction_mapping[1]).any()) or (
                                    rel_direction == 2 and df_tl['filtered_pict'].isin(rel_direction_mapping[2]).any()):
                                df_tl.loc[df_tl["filtered_pict"].isin(rel_direction_mapping[rel_direction]), "relevant_pred"] = 1
                            else:
                                df_tl.loc[df_tl["filtered_pict"] == "circle", "relevant_pred"] = 1

                            df_tl.loc[df_tl['relevant_pred'] != 1, 'relevant_pred'] = 0

                            last_state = rel_direction
                        df = pd.concat([df_tl, df_rm], ignore_index=True)


                    draw(img, df)  # Draw final image
                    folder_name = test_folder.split("/")[-1]
                    savedir = os.path.join(data_path, folder_name, 'results')
                    os.makedirs(savedir, exist_ok=True)
                    cv2.imwrite(f'{savedir}/{filename}.jpg', img)  # Save image from df

if __name__ == "__main__":
    main(parse_args())
