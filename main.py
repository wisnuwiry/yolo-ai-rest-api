from fastapi import FastAPI
from routes import predict, detect

app = FastAPI()

# Include routes
app.include_router(predict.router, prefix="/predict", tags=["Predict"])
app.include_router(detect.router, prefix="/detect", tags=["Detect"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
