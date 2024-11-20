from fastapi import APIRouter, File, UploadFile
from models.yolo_model import YOLOModel
import numpy as np
import cv2

router = APIRouter()

# Initialize model
yolo_model = YOLOModel("models/tomato.pt")

@router.post("/")
async def predict(image: UploadFile = File(...)):
    """
    Predict metadata of detected objects.
    """
    image_data = await image.read()

    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image. The image may be corrupted or invalid.")
    
    results = yolo_model.predict(img)

    # Parse results
    predictions = []
    for result in results:
        for box in result.boxes.xyxy:  # Bounding box coordinates
            print(box)
            x1, y1, x2, y2 = map(int, box[:4])
            # TODO(wisnu): recheck this confidence & class 
            conf = float(box[2])  # Confidence score
            cls = int(box[3])     # Class ID
            predictions.append({"box": [x1, y1, x2, y2], "confidence": conf, "class_id": cls})
    
    return {"predictions": predictions}
