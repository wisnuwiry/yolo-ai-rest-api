import io
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
from services.solution_service import SolutionService
from services.yolo_service import YOLOService

router = APIRouter()

yolo_service = YOLOService()
solution_service = SolutionService()

@router.post("/")
async def detect(plant_type: str = Form(...), image: UploadFile = File(...)):
    """
    Detect objects and return an image with bounding boxes.
    """
    # Validate model
    try:
        model = yolo_service.load_model(plant_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    image_data = await image.read()

    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image. The image may be corrupted or invalid.")

    # Run the predictions
    predictions = yolo_service.predict(model, img)

    # Annotate image
    annotated_img = img.copy()
    for result in predictions:
        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, result['box'])
        
        # Prepare label with class name and confidence
        label = f"{result['class_name']} {result['confidence']:.2f}"
        
        # Draw bounding box and label
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Encode and return the annotated image
    _, encoded_img = cv2.imencode(".jpg", annotated_img)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/jpeg")
