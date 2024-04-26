import cv2
import numpy as np
from PIL import Image
import pandas as pd
import os
from shapely import box, Point
import pickle

# Global variables to store mouse coordinates
mouseX = 0
mouseY = 0

# Function to draw a circle on left mouse button click
def draw_circle(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x, y
        point = Point(mouseX, mouseY)
        for ind, bbox in enumerate(markings):
            if point.within(bbox):
                if df_tl.iloc[ind, -1] == 0:
                    cv2.circle(img, (mouseX, mouseY), 20, (0, 255, 0), -1)
                    df_tl.iloc[ind, -1] = 1

# List of already labeled images
already_labeled_images = os.listdir("difficult_rel_seqs/labels")

# Load the trained model
loaded_model = pickle.load(open("finalized_model2.sav", 'rb'))

# Iterate over each image in the images folder
for i in os.listdir("difficult_rel_seqs/images"):
    # Check if the corresponding CSV file is not already labeled
    if not f"{i.split('.')[0]}.csv" in already_labeled_images:
        # Read the image
        img = np.array(Image.open(os.path.join("difficult_rel_seqs/images", i)))
        cv2.namedWindow('image')

        # Read the CSV file and extract relevant columns
        df = pd.read_csv(f"difficult_rel_seqs/frames/{i.split('.')[0]}.csv")
        df = df.iloc[:, 2:]

        # Set mouse callback function
        cv2.setMouseCallback('image', draw_circle)

        # Initialize 'rel' column in DataFrame to 0
        df["rel"] = 0
        markings = []

        # Extract traffic light bounding boxes
        df_tl = df[df['name'].str.contains('rm')]
        for ind, row in df_tl.iterrows():
            x_data = row[2:-1]
            x1, y1 = row["x"], row["y"]
            x2, y2 = int(row["x"] + row["w"]), int(row["y"] + row["h"])
            markings.append(box(x1, y1, x2, y2))
            pred = loaded_model.predict(np.array(x_data).reshape(1, -1))[0]
            if pred == 0:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 1)
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Main loop for displaying the image and processing user input
        while True:
            cv2.imshow('image', img)
            k = cv2.waitKey(20) & 0xFF
            if k == 32:  # Space key: save labels and move to the next image
                df = df[df['name'].str.contains('tl')]
                df = pd.concat([df, df_tl], ignore_index=True)
                cv2.destroyAllWindows()
                break
            if k == 27:  # Esc key: reset the image
                img = np.array(Image.open(os.path.join("difficult_rel_seqs/images", i)))
