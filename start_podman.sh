podman run -ti --device nvidia.com/gpu=all --network=host --shm-size 20G\
 --name traffic-light-detection \
 --mount type=bind,source=$(pwd),target=/workspace/traffic-light-detection/ \
  localhost/traffic-light-detection /bin/bash


