from fastapi import APIRouter, File, UploadFile
from fastapi.responses import StreamingResponse
from models.yolo_model import YOLOModel
import cv2
import numpy as np
import io

router = APIRouter()

# Initialize model
yolo_model = YOLOModel("models/tomato.pt")

@router.post("/")
async def detect(image: UploadFile = File(...)):
    """
    Detect objects and return an image with bounding boxes.
    """
    image_data = await image.read()

    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image. The image may be corrupted or invalid.")

    results = yolo_model.predict(img)

    # Annotate image
    annotated_img = img.copy()
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            print(box)
            # TODO(wisnu): recheck this confidence & class 
            conf = float(box[2])
            cls = int(box[3])
            label = f"{cls} {conf:.2f}"
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Encode and return the annotated image
    _, encoded_img = cv2.imencode(".jpg", annotated_img)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/jpeg")
