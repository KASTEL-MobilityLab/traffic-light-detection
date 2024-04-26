#!/bin/bash
cd /workspace/traffic-light-detection/inference/
python inference_images.py --model_path "/workspace/traffic-light-detection/model_weights/traffic_lights_yolov8x.pt" --model_type v8 --source_path "/workspace/traffic-light-detection/inference/inference_images" --target_path "/workspace/traffic-light-detection/output/"