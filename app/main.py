import sys
import os

# Add project root to sys.path to allow running script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
from fastapi import FastAPI
from app.routers.predict_router import router as predict_router
from app.routers.nutrition_router import router as nutrition_router

app = FastAPI()

app.include_router(predict_router)
app.include_router(nutrition_router)


@app.get("/")
def connection():
    return {"message": "Watson 모듈 LLM 서버 구동"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
