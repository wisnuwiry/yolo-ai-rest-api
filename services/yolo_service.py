from ultralytics import YOLO
import os

class YOLOService:
    def __init__(self, models_dir="models/"):
        self.models_dir = models_dir

    def load_model(self, plant_type):
        model_path = os.path.join(self.models_dir, f"{plant_type}.pt")
        if not os.path.exists(model_path):
            raise ValueError(f"Model for plant type '{plant_type}' not found.")
        return YOLO(model_path)

    def predict(self, model, img):
        results = model(img, conf=0.4)
        predictions = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = model.names[cls]
                predictions.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class_name": class_name
                })
        return predictions
