from ultralytics import YOLO

model = YOLO("./ultralytics/cfg/models/v8/yolov8n-ca.yaml") 

model.train(
    data="./path-to-dataset.yaml", # path to dataset
    epochs=300,
    device=5,
    batch=32,
    patience=0,
    project="yolov8/runs/train",
    name="yolov8n_focal_ca",
    cache="disk",
    optimizer="Adam",
    lr0=0.001,
    lrf=0.001,
    hsv_h=0.7,
    hsv_s=0.15,
    hsv_v=0.15,
    flipud=0.5,
    box=7.5, # weight for bbox loss
    cls=200.0, # weight for class loss
    dfl=0.5 # weight for dfl loss
)