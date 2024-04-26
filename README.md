# Traffic-Light-Detection


https://github.com/KASTEL-MobilityLab/traffic-light-detection/assets/38588856/3a88631f-59b1-4169-b63b-c6d0e136554e


## Introduction to Our Work

This repository encompasses our approaches and results for traffic light detection and relevance assignment.

It is currently work in progress and will be finalized for a camera ready version. 

## Installation Guide
- To ease installation, Docker is used. An existing Docker installation is therefore required.
- Clone this repository.
- Move into the project's root directory.
- Execute *./docker/build_docker.sh* script to build the docker.
- Execute *./start_docker.sh* to run the container in interactive mode
- Change to the project's root directory using *cd traffic-light-detection*.
- You are now ready to proceed with further steps.

## Usage

### Initialization
- Download model weights
    - *cd model_weights*
    - Execute script to download the model weights from cloud *./download_model_weights.sh*
        - If Error: Permission denied: *chmod +x download_model_weights.sh*


- Run inference with downloaded model weights on a provided image
	- Execute *./inference/inference_provided_image.sh*. This will use the YOLOv8x model to predict traffic lights for the provided image and stores the result in the newly created *output/images* folder


We have structured our various approaches into different folders:


- **Ultralytics Modified Folder**: Holds the modified git project from Ultralytics for YOLOv8. It is adapted for using an additional class input in the label files and outputs that class as well during predictions.

- **YOLOv7 and YOLOv9**: Both can be used to train a model and have their respective folders.

- **ROS Folder**: Contains prediction files which use published images by cameras from a ROS system.

- **Inference Folder**: Contains all files related to inference.

- **Relevance Folder**: Contains all files related to the relevance prediction of road markings and traffic lights. The relevance of traffic lights is divided into steps: train road marking model, train road marking relevance classifier, custom logic to classify traffic light relevance. This part is primarily a proof of concept that it is possible to infer relevance from road markings. The relevance logic will work for single images and sequences of images, but it may perform significantly worse if multiple images are used which are not in a sequence, as we did not implement a reset for the last state variable.

## Project Folder Structure Overview

- **configs/**: Contains different configurations to run Ultralytics YOLO training.
- **datasets/**: The folder where to put your datasets.
- **docker/**: Dockerfile as well as a script to build the container.
- **inference/**: Holds scripts to evaluate models on images and draw the results. Also allows creating inference videos.
- **preprocessing/**: Creates a dataset from multiple sources with modification of labels.
- **region_proposal/**: Train a model using the region proposal approach.
- **relevance/**: Train a model to detect the relevance of a traffic light.- **ultralytics_modified/**: Ultralytics YOLOv8 repo but with modifications.
- **utils/**: Some helper methods as well as replacement files for broken YOLO code.
- **yolov7/**: Ultralytics YOLO v7 repo. Has to be cloned on your own at the first time. See YOLOv7 section.
- **yolov9/**: Ultralytics YOLO v9 repo. Has to be cloned on your own at the first time. See YOLOv9 section.
- **start_docker.sh**: Use this script to run the Docker file with mounted folders and GPU support.
- **train_yolo.py**: Main Script to train Ultralytics models. Adjust config file as well as selected model and training arguments there.


### General Hints

- When running the *train_yolo.py* script, make sure that your dataset you want to use is in a subfolder of */datasets*.
- When using one dataset for different YOLO architectures, delete the cache files within the train and test folders of the dataset between the training runs.

### Preparing Dataset
- The published models are trained with the DriveU Traffic Light Dataset (DTLD) dataset. https://www.uni-ulm.de/in/iui-drive-u/projekte/driveu-traffic-light-dataset/
    - Request access here: https://www.uni-ulm.de/in/iui-drive-u/projekte/driveu-traffic-light-dataset/registrierungsformular-dtld/
- Make sure that dataset is mounted in Docker. Edit the start_docker.sh file or download into ./datasets/ folder.
- Run *preprocessing/convert_tif.py* to convert all DTLD TIFF images to JPEG images into a folder of your choice.
- Run *preprocessing/prepare_dataset.py* and specify the parameters to create the dataset in the YOLO format. There are various options for adjustments that can be made in this step. Refer to the documentation of the command-line argument options for more information.
- *Optional*: If you want to oversample images containing at least one object of a minority class, use *oversample.py* on your dataset.
- *Optional*: If you have background images (non-labeled images with no traffic lights) you want to use, add them manually to your dataset and run *create_empty_label_files.py* on your dataset.

### Train YOLOv7

- **First Start Only:**
    - Clone the YOLOv7 repo into your *traffic-light-detection* folder:
        ```
        git clone https://github.com/WongKinYiu/yolov7.git
        ```
    - Copy the template from *configs/custom_tl.yaml* into the YOLOv7 folder.
    - Replace *utils/yolov7/loss.py* with *yolov7_fix/utils/loss.py* to fix a bug and make YOLOv7 usable.
- Move into */yolov7* folder.
- Adjust the paths in the copy of the *custom_tl.yaml* file to the paths of your dataset.
- Run *train_aux.py* from the command line and use the following arguments:
    ```
    python train_aux.py --save_period=1 --device 0 --batch-size 2 --data custom_tl.yaml --img 1920 1920 --cfg cfg/training/yolov7-e6e.yaml --weights yolov7-e6e.pt --name v7_dtld --hyp data/hyp.scratch.p6.yaml
    ```

### Train YOLOv9

- **First Start Only:**
    - Clone the YOLOv9 repo into your *traffic-light-detection* folder:
        ```
        git clone https://github.com/WongKinYiu/yolov9.git
        ```
    - Download model weights from [YOLOv9 Releases](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-e.pt) and copy *gelan-e.pt* to the YOLOv9 folder.
    - Copy the template from *configs/custom_tl.yaml* into the YOLOv9 folder.
- Move into */yolov9* folder.
- Adjust the paths in the copy of the *custom_tl.yaml* file to the paths of your dataset.
- Run *train.py* from the command line and use the following arguments:
    ```
    python train.py --data custom_tl.yaml --img 1920 --epochs 150 --hyp data/hyps/hyp.scratch-high.yaml --device 0 --image-weights --save-period 1 --batch-size 2 --cfg models/detect/gelan-e.yaml --weights gelan-e.pt
    ```

### Train YOLOv8

- Copy the template from *configs/custom_tl.yaml* into the root directory.
- Adjust the paths in the copy of the *custom_tl.yaml* file to the paths of your dataset (use the relative path from the */datasets* folder, e.g., 'name_of_set'/train/images).
- Run *train_yolo.py* from the root directory and specify the modified *custom_tl.yaml* inside the script.

### Train YOLOv8 Split

- Run utils/split_state_pict.py* and adjust the paths accordingly to create a dataset with labels split into two labels.
- Uninstall standard Ultralytics:
    ```
    pip uninstall ultralytics
    ```
- Install Ultralytics_modified:
    ```
    cd ultralytics_modified
    python setup.py install
    ```
- Copy the template from *configs/custom_split.yaml* into the root directory.
- Adjust the paths in the copy of the *custom_split.yaml* file to the paths of your created dataset (use the relative path from the */datasets* folder, e.g., 'name_of_set'/train/images).
- Run *train_yolo.py* from the root directory and specify the modified *custom_split.yaml* inside the script.
  - The output *cls tensor* represents the loss for state classification.
  - The output *cls2 tensor* represents the loss for pictogram classification.
  - However, all other evaluations (precision, recall, etc.) only show the results for the state class; the pictogram class is ignored.
- *Please note that you should uninstall Ultralytics and install it anew via pip to work with the standard YOLOv8 again.*

### Validation

- Command line:
    ```console
    yolo val model='insert_path_to_model' data='insert_path_to_yaml' batch=4 imgsz=2048 conf='insert_conf_threshold'
    ```
    - *Make sure to pass the right YAML file you used to train the model!*

### Inference

- Move into the inference folder.
- Perform inference with the model of your choice, draw bounding boxes, or create whole inference videos.
    - *inference_images.py* for YOLOv7, YOLOv8, YOLOv9.
    - For command-line arguments, refer to the file documentation.

### Road Markings

- First, create a dataset to train a model on detecting road markings:
    - Run *utils/predict_convert_to_label_file.py*.
    - Specify the model path.
    - Use a dataset consisting of two folders: train, test, each containing an images folder as input.
- Train a YOLO model of your choice to predict road markings on your dataset:
    - Follow the instructions for training YOLOv7, YOLOv8, or YOLOv9.
    - Use a copy of *configs/custom_rm.yaml* as the config file.
    - Specify a dataset in the YAML file that contains labels for road markings.
    - *Optional*: You can also use a copy of *config/custom_tl_rm.yaml* to train one model for traffic lights and road markings together.
        - To train traffic lights and road markings together, merge the label files.
        - In *utils/predict_convert_to_label_file.py*, edit what the label integer mapping should be like.
        - Adapt line 66 in the script with the following, which sets the road markings labels (20-24) after the traffic lights (0-19):
            ```python
            print_buffer.append(f"{int(label)+20} {b_center_x} {b_center_y} {b_width} {b_height}")
            ```

### Relevance of Road Markings

- To train a gradient boosting classifier for the relevance of the road markings, use the *relevance/train_rm_relevance.py* script.
    - Specify a path to a labeled training dataset (expected to be .csv files, provided and labeled by us).
    - Specify a path to store the model weights.
    - If you have data you want to label data on your own, you can use the labeling tool under *relevance/labeling_tool_rm_relevance.py* (quick and dirty solution).

### Relevance of Traffic Lights

- Run *relevance/split_dtld_frames_into_seqs.py* with an existing DTLD dataset to create a dataset for relevance evaluation of traffic lights.
    - It will create one subfolder for each sequence in the DTLD dataset in your specified path.
    - The sequences are determined by the time between images; if two images are less than 21 seconds apart, they are considered to be in a sequence. This does not work all the time since the frames were not captured at a static frequency, but it works most of the time.
    - The resulting directory structure you will get is:
        - Target Folder
            - Sequence_id
                - images (images from the DTLD dataset within the sequence)
                - dtld (labels for the traffic lights)
                - markings (labels for the road markings)

- To perform inference with the relevance logic (contained inside the file), use the *relevance/predict_traffic_light_relevance.py* script.
    - Specify models for tl_detection, rm_detection, as well as rm_relevance classification.
    - Don't worry; it'll take some time until all models are loaded and the file execution starts.
    - Use a dataset created from the step before.
    - The resulting images will be added to the folder of each sequence under *'dataset_folder'/'sequence_number'/results*.
    - *OPTIONAL*: You can set --use_dtld_label to True to use the DTLD labels of the traffic lights instead of detecting them with your model. This allows for an analysis of your relevance logic that is independent of your tl_detection performance.

## Results
on filtered data
| Type     | Model   | Precision | Recall | mAP50 |
|----------|---------|-----------|--------|-------|
| Traffic Lights   | Yolov8 XL | 0.87 | 0.74 | 0.82 |
| Road Markings   | Yolov8 m | 0.92      | 0.90   | 0.94  |
| Road Marking Relevance   | Gradient Boosting | 0.96      | 0.96   | -  |

## Troubleshooting

| Error/Warning         | Scenario                                      | Recommended Actions                                |
|-----------------------|-----------------------------------------------|----------------------------------------------------|
| 'ConnectionResetError: [Errno 104] Connection reset by peer' -> Means that VRAM is full | Training a yolo model using ultralytics | Use a smaller batch size/ smaller model/ lower image resolution |
| '_pickle.UnpicklingError: STACK_GLOBAL requires str' -> Means that cache of dataset has to be deleted | Training a yolo v7 model after using same dataset with another model | Delete caches in the dataset folder |
