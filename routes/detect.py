from fastapi import APIRouter, File, UploadFile
from fastapi.responses import StreamingResponse
from services.yolo import preprocess_image, postprocess_output, draw_boxes
from models.yolo_model import YOLOModel
import numpy as np
import cv2
import io

router = APIRouter()

# Initialize model
yolo_model = YOLOModel("models/yolo11n.onnx")  # Adjust path as needed

@router.post("/")
async def detect(image: UploadFile = File(...)):
    image_data = await image.read()
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    input_size = (416, 416)
    input_tensor = preprocess_image(img, input_size)
    outputs = yolo_model.predict(input_tensor)

    boxes, scores, classes = postprocess_output(outputs, img)
    labels = ["class_1", "class_2", "class_3"]

    annotated_image = draw_boxes(img, boxes, scores, classes, labels)
    _, encoded_img = cv2.imencode(".jpg", annotated_image)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/jpeg")
