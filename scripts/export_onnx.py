from ultralytics import YOLO

model = YOLO("yolov8n.pt")
path = model.export(format="onnx", imgsz=640, dynamic=False, simplify=True, opset=13, nms=False)
print("Exported to:", path)
