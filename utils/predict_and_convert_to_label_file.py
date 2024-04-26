import os
from ultralytics import YOLO
import json
from PIL import Image
import argparse
from tqdm import tqdm

def main(model_path, input_path):
    """
    Perform inference using YOLO model on images in the specified input path
    and save the predicted bounding boxes as label files.

    Args:
        model_path (str): Path to the YOLO model weights.
        input_path (str): Path to the folder containing images for inference.
    """

    # Load YOLO model
    model = YOLO(model_path)

    # Iterate over train and test splits
    for split in ["train", "test"]:
        image_folder = os.path.join(input_path, split, "images")
        label_folder = os.path.join(input_path, split, "labels")
        os.makedirs(label_folder, exist_ok=True)

        # Iterate over images in the current split
        for image_name in tqdm(os.listdir(image_folder), desc=f'Processing {split} images'):
            print_buffer = []
            image_path = os.path.join(image_folder, image_name)

            # Perform inference on the current image
            pil_img = Image.open(image_path)
            image_width, image_height = pil_img.size

            # Get predictions
            result = model.predict(image_path, verbose=False, conf=0.5, iou=0.4, device=0)
            for box in json.loads(result[0].tojson()):
                if box["confidence"] >= 0.5:
                    box_dims = box['box']
                    x1 = box_dims['x1']
                    x2 = box_dims['x2']
                    y1 = box_dims['y1']
                    y2 = box_dims['y2']
                    label = box["name"]

                    # Clip coordinates to image boundaries
                    x1 = max(x1, 0)
                    y2 = min(y2, image_height - 1)
                    y1 = max(y1, 0)
                    x2 = min(x2, image_width - 1)
                    h = y2 - y1
                    w = x2 - x1
                    b_center_x = (x1 + x2) / 2
                    b_center_y = (y1 + y2) / 2
                    b_width = w
                    b_height = h

                    # Normalize the coordinates by the dimensions of the image
                    b_center_x /= image_width
                    b_center_y /= image_height
                    b_width /= image_width
                    b_height /= image_height

                    # Write the bbox details to the file
                    print_buffer.append(f"{int(label)} {b_center_x} {b_center_y} {b_width} {b_height}")

            # Save label file
            filename = image_name.split(".")[0]
            with open(os.path.join(label_folder, f"{filename}.txt"), 'a') as filehandle:
                for line in print_buffer:
                    filehandle.write(("".join(line) + "\n"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform inference using YOLO model.")
    parser.add_argument("--model_path", type=str, help="Path to YOLO model weights.")
    parser.add_argument("--input_path", type=str, help="Path to folder containing images for inference. Should contain train and test folder each containing an images folder.")
    args = parser.parse_args()

    main(args.model_path, args.input_path)
