import os
import shutil
from datetime import datetime
import argparse
from tqdm import tqdm

def main(dtld_image_folder, rm_label_folder, dtld_label_folder, output_folder):
    # List files in the directories
    labels = os.listdir(dtld_label_folder)

    last_date = False
    current_seq = []
    seq_num = 0

    with tqdm(total=len(labels), desc="Processing images") as pbar:
        # Iterate over files in the first directory
        for i in labels:
            # Extract datetime from the image name
            img_name = "-".join(i.split("_")[2:-1]).split("-")[:-1]
            datetime_object = datetime.strptime("-".join(img_name), '%Y-%m-%d-%H-%M-%S')

            if last_date:
                # Check if the difference between current datetime and last datetime is greater than 21 seconds
                if (datetime_object - last_date).seconds > 21:
                    # Create new sequence directories
                    os.makedirs(f"{output_folder}/{seq_num}/images", exist_ok=True)
                    os.makedirs(f"{output_folder}/{seq_num}/dtld", exist_ok=True)
                    os.makedirs(f"{output_folder}/{seq_num}/markings", exist_ok=True)

                    # Copy relevant files to the new sequence directories
                    for j in current_seq:
                        shutil.copy(os.path.join(rm_label_folder, j), os.path.join(f"{output_folder}/{seq_num}/markings", j)) # copy marking labels
                        shutil.copy(os.path.join(dtld_label_folder, j), os.path.join(f"{output_folder}/{seq_num}/dtld", j)) # copy traffic light labels
                        shutil.copy(os.path.join(dtld_image_folder, f"{j.split('.')[0]}.jpg"), os.path.join(f"{output_folder}/{seq_num}/images", f"{j.split('.')[0]}.jpg")) # copy images
                    seq_num += 1
                    current_seq = [i]  # Reset current sequence
                else:
                    current_seq.append(i)  # Add file to current sequence
            else:
                current_seq.append(i)  # Add file to current sequence

            last_date = datetime_object  # Update last datetime
            pbar.update(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images and labels.')
    parser.add_argument('--dtld_image_folder', type=str, help='Path to the folder containing images')
    parser.add_argument('--rm_label_folder', type=str, help='Path to the folder containing marking labels')
    parser.add_argument('--dtld_label_folder', type=str, help='Path to the folder containing traffic light labels')
    parser.add_argument('--output_folder', type=str, help='Output folder path')

    args = parser.parse_args()
    main(args.dtld_image_folder, args.rm_label_folder, args.dtld_label_folder, args.output_folder)
