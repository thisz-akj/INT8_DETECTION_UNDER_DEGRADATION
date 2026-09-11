import os
from ultralytics import YOLO

# Point at models/yolov8n.pt (committed to the repo) rather than a bare "yolov8n.pt" --
# ultralytics saves the exported ONNX alongside the input weights, so this keeps
# both files in models/ instead of dropping copies at the repo root.
os.makedirs("models", exist_ok=True)
model = YOLO("models/yolov8n.pt")
path = model.export(format="onnx", imgsz=640, dynamic=False, simplify=True, opset=13, nms=False)
print("Exported to:", path)
