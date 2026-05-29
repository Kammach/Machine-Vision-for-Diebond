from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")

# Train
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16
)