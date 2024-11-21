from fastapi import FastAPI
from routes.predict import router as predict_router
from routes.detect import router as detect_router

app = FastAPI()

app.include_router(predict_router, prefix="/api/v1/predict", tags=["Prediction"])
app.include_router(detect_router, prefix="/api/v1/detect", tags=["Detection"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Plant Disease Prediction API!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)