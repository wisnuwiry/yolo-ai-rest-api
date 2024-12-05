import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.predict import router as predict_router
from routes.detect import router as detect_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router, prefix="/api/v1/predict", tags=["Prediction"])
app.include_router(detect_router, prefix="/api/v1/detect", tags=["Detection"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Plant Disease Prediction API!"}
