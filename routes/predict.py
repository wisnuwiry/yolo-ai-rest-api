from fastapi import APIRouter, File, Form, HTTPException, UploadFile
import numpy as np
import cv2
from services.solution_service import SolutionService
from services.yolo_service import YOLOService

router = APIRouter()

yolo_service = YOLOService()
solution_service = SolutionService()

@router.post("/")
async def predict(plant_type: str = Form(...), image: UploadFile = File(...)):
    """
    Predict metadata of detected objects.
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

    # Add solutions to predictions
    for prediction in predictions:
        disease = prediction["class_name"]
        prediction["solution"] = solution_service.get_solution(plant_type, disease)
     
    return {"plant_type": plant_type, "predictions": predictions}
