from ultralytics import YOLO, RTDETR


# Load a model
model = YOLO('yolov8x.yaml')

# Train the model: Here training code for Yolov8x on DTLD dataset
results = model.train(data='/workspace/traffic-light-detection/custom.yaml',
                      epochs=100, # train for 100 epochs
                      imgsz=2048, # full image size of DTLD
                      workers=24, # adapt to available CPU cores
                      device="0", # add more GPUs if available
                      batch=6, # relative to how much GPU memory you have, this was used for H100 80GB
                      nbs=64,
                      patience=0, # disable early stopping
                      project="yolo8x-traffic-light-detection",
                      name="yolo8x-traffic-light-detection",
                      pretrained=True, # use MS-Coco weights for initialization
                      optimizer="AdamW",
                      lr0=0.0001,
                      warmup_epochs=1, # one epoch warmup
                      warmup_bias_lr = 1e-9,
                      warmup_momentum = 1e-9,
                      cos_lr=True,
                      lrf=0, # decay learning rate to zero with cosine function
                      label_smoothing=0.001, # small regularization
                      mosaic=0.6, # refer all following augmentations to here: https://docs.ultralytics.com/reference/data/augment/
                      scale=0.4,
                      degrees= 5,
                      translate=0.1,
                      save_period=1,
                      fliplr=0.5,
                      mixup=0.2,
                      shear=0.3,
                      copy_paste=0.4,
                      close_mosaic=5,
                      cls=0.7,
                      plots=True
                      )
