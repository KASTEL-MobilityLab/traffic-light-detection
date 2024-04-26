FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

#ENV PATH=/opt/cmake-3.14.6-Linux-x86_64/bin/:/usr/local/mpi/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/ucx/bin:/opt/tensorrt/bin

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN --mount=type=cache,target=/var/cache/apt apt-get update && apt-get install -y \
    #apt-utils \
    git \
    ssh \ 
    #python-is-python3 \ 
    python3-pip \
   libcppunit-dev \
   clang \
   clang-format \
   vim \
   && apt-get clean \
   && rm -rf /var/lib/apt/lists/*

# opencv
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


# ros
RUN python3 -m pip install pycocotools==2.0.7 PyYaml==6.0.1

# processing
RUN python3 -m pip install scikit-learn==1.3.2 pyarrow==15.0.2 shapely==2.0.3

# modelling
RUN python3 -m pip install ultralytics==8.1.31

# inference
RUN python3 -m pip install opencv-python moviepy==1.0.3 natsort==8.4.0

#region proposal
RUN python3 -m pip install transformers[torch]
RUN python3 -m pip install datasets

# Clone yolov7
WORKDIR /workspace/traffic-light-detection
RUN git clone https://github.com/WongKinYiu/yolov7.git
RUN python3 -m pip install -r yolov7/requirements.txt


# Clone yolov9
WORKDIR /workspace/traffic-light-detection
RUN git clone https://github.com/WongKinYiu/yolov9.git
RUN python3 -m pip install -r yolov9/requirements.txt


WORKDIR /workspace
ENTRYPOINT [ "/bin/bash", "-l", "-c" ]
