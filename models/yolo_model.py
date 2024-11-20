from ultralytics import YOLO

class YOLOModel:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def predict(self, image):
        """
        Run inference on the input image.
        """
        results = self.model(image, conf=0.5)
        return results
