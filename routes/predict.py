from fastapi import APIRouter, File, UploadFile
from services.yolo import preprocess_image, postprocess_output
from models.yolo_model import YOLOModel
import numpy as np
import cv2

router = APIRouter()

# Initialize model
yolo_model = YOLOModel("models/yolo11n.onnx")  # Adjust path as needed

@router.post("/")
async def predict(image: UploadFile = File(...)):
    image_data = await image.read()
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    input_size = (416, 416)
    input_tensor = preprocess_image(img, input_size)
    outputs = yolo_model.predict(input_tensor)

    boxes, scores, classes = postprocess_output(outputs, img)
    labels = ["class_1", "class_2", "class_3"]

    results = [
        {
            "box": box,
            "score": score,
            "class": labels[class_id],
        }
        for box, score, class_id in zip(boxes, scores, classes)
    ]
    return {"predictions": results}
