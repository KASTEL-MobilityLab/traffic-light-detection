docker run --gpus all --network=host --shm-size 20G\
 --mount type=bind,source=$(pwd),target=/workspace/traffic-light-detection/ \
  -it tld:latest --name traffic_light_detection /bin/bash


