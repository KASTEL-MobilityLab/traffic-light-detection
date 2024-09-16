from ultralytics import YOLO


# Load a model pretrained on Open Images V7
model = YOLO("yolov8x-oiv7.pt")

# Train the model: Here training code for Yolov8x on DTLD dataset
results = model.train(data='/workspace/traffic-light-detection/configs/custom_tl.yaml',

                    name="yolo8x-traffic-light-detection",
                    plots=True,

                    ## Data
                    epochs=100, # train for 100 epochs
                    imgsz=2048, # full image size of DTLD
                    workers=24, # adapt to available CPU cores
                    patience=0, # disable early stopping

                    ## GPU
                    device=[0], # add more GPUs if available
                    batch=3, # 3 is maximum for 24 GB VRAM (RTX 4090). If more VRAM is available, increase this value.
                    nbs=30, # gradient accumulation steps.
                    amp = True, # automatic mixed precision training

                    ## Optimizer
                    optimizer="SGD", # SGD is more stable for fp16 training than AdamW
                    lr0=0.005, # initial learningrate for SGD
                    lrf=0.001, # decay to 5e-6
                    cos_lr=True, # cosine decay

                    ## Warmup
                    warmup_bias_lr = 0.0, # also warmup bias learning rate not just weights learning rate
                    warmup_epochs=3, # three warmup epochs until lr0 is reached

                    ## Augmentation and Regularization refer here: https://docs.ultralytics.com/reference/data/augment/
                    label_smoothing=0.001,
                    mosaic=0.7,
                    close_mosaic=15,
                    scale=0.7,
                    degrees= 5,
                    translate=0.1,
                    save_period=1,
                    fliplr=0, # do not flip images as it would change arrow directions
                    mixup=0.1,
                    shear=0.1,
                    copy_paste=0.1,
                    )
