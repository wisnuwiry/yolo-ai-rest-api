import base64
from io import BytesIO
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
async def predict(
    plant_type: str = Form(...),
    image: UploadFile = File(...),
    show_image: bool = Form(False)
):
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
        error = {
            "status": "error",
            "message": "Failed to decode image. The image may be corrupted or invalid."
        }

        raise HTTPException(400, error)
    
    # Run the predictions
    predictions = yolo_service.predict(model, img)

    # Add solutions to predictions
    for prediction in predictions:
        disease = prediction["class_name"]
        solution_data = solution_service.get_solution_data(plant_type, disease)
        prediction["solution"] = solution_data["solution"]
        prediction["class_label"] = solution_data["disease_label"]
    
    # Resize image and encode to base64 if `show_image` is True
    base64_image = None
    if show_image:
        resized_img = cv2.resize(img, (300, 300))  # Resize to smaller size
        _, buffer = cv2.imencode(".jpg", resized_img)
        base64_image = base64.b64encode(buffer).decode("utf-8")
     
    # Construct response
    response = {
        "detail": {
            "status": "success",
            "plant_type": plant_type,
            "predictions": predictions,
        }
    }

    if show_image:
        response["detail"]["image"] = base64_image

    return response
