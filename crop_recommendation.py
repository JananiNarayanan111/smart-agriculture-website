from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os

router = APIRouter()

# Load pre-trained Random Forest Classifier
MODEL_PATH = "models/crop_recommendation.pkl"

class CropRecommendationRequest(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float
    region: str

class CropRecommendationResponse(BaseModel):
    recommended_crop: str
    confidence: float
    top_3_crops: list

# Crop names mapping
CROP_NAMES = {
    0: "rice", 1: "maize", 2: "jute", 3: "cotton", 4: "coconut", 5: "sugarcane",
    6: "groundnut", 7: "soyabean", 8: "sorghum", 9: "wheat", 10: "barley",
    11: "mungbean", 12: "masoor", 13: "mothbean", 14: "uradbean", 15: "pigeonpea",
    16: "kidneybeans", 17: "chickpea", 18: "lentil", 19: "pomegranate", 20: "banana",
    21: "mango", 22: "grapes", 23: "watermelon", 24: "muskmelon", 25: "apple",
    26: "orange", 27: "papaya", 28: "coconut", 29: "cotton"
}

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=500, detail="Model not found. Please train the model first.")
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

@router.post("/recommend", response_model=CropRecommendationResponse)
def recommend_crop(request: CropRecommendationRequest):
    try:
        model = load_model()
        
        # Prepare input features
        features = np.array([[
            request.nitrogen,
            request.phosphorus,
            request.potassium,
            request.temperature,
            request.humidity,
            request.ph,
            request.rainfall
        ]])
        
        # Get prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Get top 3 crops
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        top_3_crops = [
            {
                "crop": CROP_NAMES.get(idx, "unknown"),
                "confidence": float(probabilities[idx] * 100)
            }
            for idx in top_3_indices
        ]
        
        return CropRecommendationResponse(
            recommended_crop=CROP_NAMES.get(prediction, "unknown"),
            confidence=float(probabilities[prediction] * 100),
            top_3_crops=top_3_crops
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/crops-list")
def get_crops_list():
    return {"crops": list(set(CROP_NAMES.values()))}