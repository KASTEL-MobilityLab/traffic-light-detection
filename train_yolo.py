from ultralytics import YOLO, RTDETR

# Load a model
#model = RTDETR('rtdetr-l.pt')  # build a new model from YAML
#model = YOLO("yolov8x.yaml")
model = YOLO("ultralytics_modified/ultralytics/cfg/models/v8/yolov8.yaml") #xl
# Train the model
results = model.train(data='custom_split.yaml', name='insert_name', epochs=100, imgsz=1920, device=0, batch=4, patience=0, mosaic=0.3, scale=0.7, save_period=1, cos_lr=True, fliplr=0, shear=0.1, copy_paste=0.1, perspective=0.0002,amp=False, mixup=0.1, degrees=5, cache="ram", workers=24,optimizer="AdamW", lr0=0.001, lrf=0.00005, translate=0, hsv_v=0.7, hsv_h=0.05)
