import argparse
import os
import glob
import cv2
import json
from ultralytics import YOLO
from tqdm import tqdm
import moviepy.video.io.ImageSequenceClip
import torch
from natsort import natsorted



def draw_border(img, left,top,right,bot, color, thickness, r=5, d=10):
    x1,y1 = left,top
    x2,y2 = right,bot
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)




# adapted from https://gist.github.com/kodekracker/1777f35e8462b2b21846c14c0677f611
def drawBoundingBoxes(imageData, imageOutputPath, inferenceResults, rounded_edges = False):
    """
    Draw bounding boxes on an image.
    
    Args:
        imageData: Image data in numpy array format.
        imageOutputPath: Output image file path.
        inferenceResults: Inference results array of objects containing bounding box coordinates and labels.
    """
    
    # Calculate the thickness of the bounding box

    thick = int(sum(imageData.shape[:2]) / 1200)
    for res in inferenceResults:
        left = int(res['left'])
        top = int(res['top'])
        right = int(res['right'])
        bottom = int(res['bottom'])
        label = res['label']
        imgHeight, imgWidth, _ = imageData.shape
                
        # Determine color based on label
        if 'yellow' in label:
            color = (3, 111, 252)

        elif 'red' in label:
            color = (0, 3, 252)
        elif 'green' in label:
            color = (28, 252, 3)
        else:
            color = (92, 91, 91)
        
        # Draw bounding box
        if rounded_edges: # nicer visualisation
            draw_border(imageData, left, top, right, bottom, color, thick)       
        else: # usual 2d bbox visualization
            cv2.rectangle(imageData,(left, top), (right, bottom), color, thick)
        # Adjust text position to avoid going out of the image
        if left > 1500:
            left -= 11 * len(label)
        
        # Add label text
        cv2.putText(imageData, label, (left, top - 10), 0, 0.00075 * imgHeight, color, thick)

    # Save the image with bounding boxes and labels

    cv2.imwrite(imageOutputPath, imageData)

def load_model(model_type, model_path):
    """
    Load YOLO model based on the specified model type.

    Args:
        model_type: Type of YOLO model (v7, v8, or v9).

    Returns:
        Loaded YOLO model.
    """
    if model_type == 'v7':
        return torch.hub.load('WongKinYiu/yolov7', 'custom', model_path,
                        force_reload=True, trust_repo=True).to("cuda")
    elif model_type == 'v8':
        return YOLO(model_path)
    elif model_type == 'v9':
        return torch.hub.load('WongKinYiu/yolov9', 'custom', model_path,
                        force_reload=True, trust_repo=True).to("cuda")
    else:
        raise ValueError("Invalid YOLO model type. Choose 'v7', 'v8', or 'v9'.")

def main(args):

    # load the model depending on the type
    model_type = args.model_type
    model = load_model(args.model_type, args.model_path)

    source_dir = args.source_path
    target_dir = args.target_path

    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(os.path.join(target_dir, "images"), exist_ok=True) 
    # list all images in the folder
    image_files = glob.glob(os.path.join(source_dir, '*.jpg'))
    for image_file in tqdm(image_files):
        
        try:
            image_file_name = os.path.split(image_file)[-1]
            imcv = cv2.imread(image_file)

            to_draw = []
            # distinguish the inference of v7/v9 and v8
            if model_type in ['v7', 'v9']:
                result = model(image_file)
                for _, row in result.pandas().xyxy[0].iterrows():
                    if row["confidence"] >= 0.3:
                        result_args = {
                            "left": row['xmin'],
                            "top": row['ymin'],
                            "right": row['xmax'],
                            'bottom': row['ymax'],
                            'label': row['name']
                        }
                        to_draw.append(result_args)

            else:
                result = model.predict(image_file, verbose=False, imgsz=(1920, 1216), conf=0.5, iou=0.7)
                for box in json.loads(result[0].tojson()):
                    box_dims = box['box']
                    result_args = {
                        "left": box_dims['x1'],
                        "top": box_dims['y1'],
                        "right": box_dims['x2'],
                        'bottom': box_dims['y2'],
                        'label': box['name']
                    }
                    to_draw.append(result_args)

            target_file = os.path.join(target_dir,"images", image_file_name)
            drawBoundingBoxes(imcv, target_file, to_draw)

        except Exception as e:
            print(f"Error occurred during :{image_file_name}: {e}")

    if args.video:
        os.makedirs(os.path.join(target_dir, "videos"), exist_ok=True)
        image_files = sorted(glob.glob(os.path.join(target_dir,"images", '*.jpg')))
        image_files = natsorted(image_files)
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=args.fps)
        clip.write_videofile(f'{target_dir}/videos/inference_video.mp4')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic Light Detection Script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the YOLO model")
    parser.add_argument("--model_type", type=str, required=True, help="Version of the Yolo model. 'v7','v8', or 'v9'")
    parser.add_argument("--source_path", type=str, required=True, help="Directory which holds folder of images.")
    parser.add_argument("--target_path", type=str, required=True, help="Directory where to store the resulting images and videos.")
    parser.add_argument("--fps", type=int, default=45, help="Frames per second for output video")
    parser.add_argument("--video", type=bool, default=False, help="Frames per second for output video")
    args = parser.parse_args()
    main(args)
