import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.predict import router as predict_router
from routes.detect import router as detect_router

app = FastAPI()

origins_env = os.getenv("ALLOWED_ORIGINS", "")
if origins_env:
    origins = origins_env.split(",")  # Split comma-separated list
else:
    origins = [
        "http://localhost",
        "http://localhost:8080",
        "http://localhost:3000",
        "http://localhost:5173",
    ]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router, prefix="/api/v1/predict", tags=["Prediction"])
app.include_router(detect_router, prefix="/api/v1/detect", tags=["Detection"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Plant Disease Prediction API!"}
