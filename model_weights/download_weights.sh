#!/bin/bash
# Specify the URL Model weights are stored here

# yolov8 split
wget -nc --content-disposition --trust-server-names "https://bwsyncandshare.kit.edu/s/mAWFNrfJboLbAq6/download/traffic_lights_split_yolov8.pt"

# yolov9
wget -nc --content-disposition --trust-server-names "https://bwsyncandshare.kit.edu/s/Pc9jKBp6ncNbpf7/download/traffic_lights_yolov9.pt"

#yolov8xl 
wget -nc --content-disposition --trust-server-names "https://bwsyncandshare.kit.edu/s/RrbwrQ7iLktd87W/download/traffic_lights_yolov8xl.pt"

# yolov8x
wget -nc --content-disposition --trust-server-names "https://bwsyncandshare.kit.edu/s/ZLtoyf4q8nTLL7Y/download/traffic_lights_yolov8x.pt"

#yolov8m road markings
wget -nc --content-disposition --trust-server-names  "https://bwsyncandshare.kit.edu/s/5YQPPoz4mPSe3j9/download/road_markingsyolov8m.pt"

#yolov7 
wget -nc --content-disposition --trust-server-names  "https://bwsyncandshare.kit.edu/s/yBrYefxSMm2wkdo/download/traffic_lights_yolov7.pt"