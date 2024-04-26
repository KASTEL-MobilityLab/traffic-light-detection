docker run -ti --gpus all --network=host --shm-size 20G\
 --mount type=bind,source=$(pwd),target=/workspace/traffic-light-detection/ \
 --name traffic-light-detection traffic-light-detection:latest /bin/bash


