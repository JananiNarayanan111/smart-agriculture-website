from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os

router = APIRouter()

MODEL_PATH = "models/yield_prediction.pkl"

class YieldPredictionRequest(BaseModel):
    crop: str
    area: float
    rainfall: float
    temperature: float
    humidity: float
    nitrogen: float
    phosphorus: float
    potassium: float
    region: str

class YieldPredictionResponse(BaseModel):
    predicted_yield: float
    unit: str
    confidence_interval: dict
    recommendations: list

CROP_TO_INDEX = {
    "rice": 0, "maize": 1, "jute": 2, "cotton": 3, "coconut": 4,
    "sugarcane": 5, "groundnut": 6, "soyabean": 7, "sorghum": 8, "wheat": 9,
    "barley": 10, "mungbean": 11, "masoor": 12, "mothbean": 13, "uradbean": 14,
    "pigeonpea": 15, "kidneybeans": 16, "chickpea": 17, "lentil": 18,
    "pomegranate": 19, "banana": 20, "mango": 21, "grapes": 22, "watermelon": 23,
    "muskmelon": 24, "apple": 25, "orange": 26, "papaya": 27
}

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=500, detail="Model not found. Please train the model first.")
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

@router.post("/predict", response_model=YieldPredictionResponse)
def predict_yield(request: YieldPredictionRequest):
    try:
        model = load_model()
        
        # Get crop index
        crop_index = CROP_TO_INDEX.get(request.crop.lower(), 0)
        
        # Prepare features
        features = np.array([[
            crop_index,
            request.area,
            request.rainfall,
            request.temperature,
            request.humidity,
            request.nitrogen,
            request.phosphorus,
            request.potassium
        ]])
        
        # Prediction
        yield_prediction = model.predict(features)[0]
        
        # Confidence interval (±10%)
        lower_bound = yield_prediction * 0.9
        upper_bound = yield_prediction * 1.1
        
        recommendations = generate_yield_recommendations(request.crop, yield_prediction)
        
        return YieldPredictionResponse(
            predicted_yield=round(yield_prediction, 2),
            unit="tons/hectare",
            confidence_interval={
                "lower": round(lower_bound, 2),
                "upper": round(upper_bound, 2)
            },
            recommendations=recommendations
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def generate_yield_recommendations(crop: str, yield_value: float) -> list:
    recommendations = []
    
    if yield_value < 2:
        recommendations.append("Yield is below average. Consider improving soil quality and irrigation.")
    elif yield_value < 4:
        recommendations.append("Moderate yield. Optimize fertilizer usage for better results.")
    else:
        recommendations.append("Good yield predicted. Maintain current practices.")
    
    recommendations.append(f"Monitor {crop} growth regularly for diseases.")
    recommendations.append("Plan crop rotation for soil health.")
    
    return recommendations