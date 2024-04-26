## imports
from __future__ import print_function

import argparse
import logging
import sys
import numpy as np
import os
import cv2
import shutil
import multiprocessing
import glob
from tqdm import tqdm

from dtld_parsing.driveu_dataset import DriveuDatabase
from PIL import Image


np.set_printoptions(suppress=True)

# Logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: " "%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--target_path", help="Path to the directory where to store the data", type=str, required=True)
    parser.add_argument("--data_path", help="Path to the raw data", type=str, required=False, default="/data/DTLD")
    parser.add_argument("--label_path", help="Path to the label data", type=str, required=False, default="/data/DTLD/v2.0")
    return parser.parse_args()


def convert_images(target_path: str, label_path: str, data_path: str, split: str):
    """
    Convert images depending on the labels. If there is no label for an image, the image is not converted.
    """

    database = DriveuDatabase(os.path.join(label_path, f"DTLD_{split}.json"))
    if not database.open(data_path):
        logging.warn(f"Database could not be opened for {split}.")
        return False

    # Convert image by image
    for _, img in tqdm(enumerate(database.images), desc=f"Converting images for {split}", total=len(database.images)):
        # name of the image
        img_name = f"{img.file_path.split('/')[-1].split('.')[0]}"

        # Get color images
        img_color = img.get_labeled_image()
        img_color = cv2.resize(img_color, (1024, 440))
        img_concat_rgb = img_color[..., ::-1]
        final_img = Image.fromarray(img_concat_rgb)
        final_img = final_img.resize((2048, 1024))
        final_img.save(os.path.join(target_path, split, f"{img_name}.jpg"))

    return True


def main(args):
        
    data_path = args.data_path
    if not os.path.exists(data_path):
        raise Exception("Invalid data path.")

    label_path = args.label_path
    if not os.path.exists(label_path):
        raise Exception("Invalid label path.")
    
    target_path = args.target_path
    if os.path.exists(target_path):
        if input("Target folder already exists. Current files in this folder are deleted. Are you sure to overwrite the data? Please type yes. \n") != 'yes':
            raise Exception("Stopping Execution! Please Change --target_path !")
        shutil.rmtree(target_path)
    
    os.makedirs(f"{target_path}/train", exist_ok=True)
    os.makedirs(f"{target_path}/test", exist_ok=True)

    ## using multi processing for convertion of the TIF data
    processlist = []

    for split in ["train", "test"]:
        p = multiprocessing.Process(target=convert_images, args=(target_path, label_path, data_path, split))
        processlist.append(p)
        p.start()

    for pr in processlist:
        pr.join()

    train_images = {len(glob.glob(target_path + '/train/*.jpg'))}
    test_images = {len(glob.glob(target_path + '/test/*.jpg'))}
    logging.info(f"Finished conversion of images.\n Final amount in train folder: {train_images}.\n Final amount in test folder: {test_images}")

    
if __name__ == "__main__":
    main(parse_args())
