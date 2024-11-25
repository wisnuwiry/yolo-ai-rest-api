from fastapi import APIRouter, File, Form, HTTPException, UploadFile
import numpy as np
import cv2
from services.solution_service import SolutionService
from services.yolo_service import YOLOService
from services.model_service import ModelService

router = APIRouter()

yolo_service = YOLOService()
solution_service = SolutionService()

@router.post("/")
async def predict(plant_type: str = Form(...), image: UploadFile = File(...)):
    """
    Predict metadata of detected objects.
    """
    # Validate model with plant type
    validation = ModelService.validate_plant_type(plant_type)

    if(validation != None):
        raise HTTPException(400, validation)

    # Validate model
    try:
        model = yolo_service.load_model(plant_type)
    except ValueError as e:
        error = {
            "status": "error",
            "message": str(e),
        }
        raise HTTPException(400, error)
    
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
        solution_data = solution_service.get_solution_data(plant_type, disease)
        prediction["solution"] = solution_data["solution"]
        prediction["class_label"] = solution_data["disease_label"]
     
    return {
        "detail": {
            "status": "success",
            "plant_type": plant_type, 
            "predictions": predictions
        }
    }
