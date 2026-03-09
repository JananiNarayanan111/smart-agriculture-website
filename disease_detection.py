from fastapi import APIRouter, File, UploadFile, HTTPException
from PIL import Image
import numpy as np
import tensorflow as tf
from io import BytesIO
import os

router = APIRouter()

# Load pre-trained disease detection model
MODEL_PATH = "models/disease_detection_model.h5"

class DiseaseDetectionResponse:
    def __init__(self, disease_name: str, confidence: float, treatment: str):
        self.disease_name = disease_name
        self.confidence = confidence
        self.treatment = treatment

# Disease information mapping
DISEASE_INFO = {
    0: {"name": "Healthy", "treatment": "No treatment needed. Continue regular maintenance."},
    1: {"name": "Apple Scab", "treatment": "Apply fungicide. Remove infected leaves. Improve air circulation."},
    2: {"name": "Apple Black rot", "treatment": "Prune affected branches. Apply fungicide. Sanitize tools."},
    3: {"name": "Cedar Apple rust", "treatment": "Remove galls from cedar trees. Apply sulfur fungicide."},
    4: {"name": "Tomato Early blight", "treatment": "Remove lower leaves. Apply copper fungicide. Improve spacing."},
    5: {"name": "Tomato Late blight", "treatment": "Apply mancozeb. Remove infected leaves. Improve ventilation."},
    6: {"name": "Tomato Septoria leaf spot", "treatment": "Remove affected leaves. Apply chlorothalonil. Water at soil level."},
    7: {"name": "Powdery mildew", "treatment": "Apply sulfur or neem oil. Increase air circulation. Reduce humidity."},
    8: {"name": "Rice Blast", "treatment": "Apply tricyclazole or azoxystrobin. Improve drainage. Use resistant varieties."},
    9: {"name": "Wheat Rust", "treatment": "Apply propiconazole. Use resistant varieties. Remove volunteer wheat."},
}

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=500, detail="Model not found. Please train the model first.")
    return tf.keras.models.load_model(MODEL_PATH)

@router.post("/detect")
async def detect_disease(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        
        # Preprocess image
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        # Load model and predict
        model = load_model()
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]) * 100)
        
        # Get disease info
        disease_info = DISEASE_INFO.get(predicted_class, {
            "name": "Unknown Disease",
            "treatment": "Consult with an agricultural expert."
        })
        
        return {
            "disease": disease_info["name"],
            "confidence": confidence,
            "treatment": disease_info["treatment"],
            "recommendation": "Upload a clear image of the affected leaf for accurate diagnosis."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@router.get("/diseases-list")
def get_diseases_list():
    return {
        "diseases": [
            {"id": k, "name": v["name"]}
            for k, v in DISEASE_INFO.items()
        ]
    }