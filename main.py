from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
from dotenv import load_dotenv
import traceback

# Import routes
from routes import crop_recommendation, disease_detection, weather, yield_prediction, market_prices

load_dotenv()

app = FastAPI(
    title="Smart Agriculture API 🌾",
    description="AI-Powered Agriculture Solution for Indian Farmers",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routes
app.include_router(crop_recommendation.router, prefix="/api/crop", tags=["Crop Recommendation"])
app.include_router(disease_detection.router, prefix="/api/disease", tags=["Disease Detection"])
app.include_router(weather.router, prefix="/api/weather", tags=["Weather"])
app.include_router(yield_prediction.router, prefix="/api/yield", tags=["Yield Prediction"])
app.include_router(market_prices.router, prefix="/api/market", tags=["Market Prices"])

@app.get("/")
def read_root():
    return {
        "message": "🌾 Welcome to Smart Agriculture Website API",
        "status": "✅ Running",
        "version": "1.0.0",
        "endpoints": {
            "crop_recommendation": "/api/crop/recommend",
            "disease_detection": "/api/disease/detect",
            "weather": "/api/weather/current/{city}",
            "yield_prediction": "/api/yield/predict",
            "market_prices": "/api/market/current-price"
        },
        "documentation": "/docs"
    }

@app.get("/health")
def health_check():
    return {"status": "🟢 Healthy", "service": "Smart Agriculture API"}

@app.get("/api/status")
def api_status():
    return {
        "status": "✅ All systems operational",
        "timestamp": str(__import__('datetime').datetime.now()),
        "services": {
            "crop_recommendation": "✅ Ready",
            "disease_detection": "✅ Ready",
            "weather": "✅ Ready",
            "yield_prediction": "✅ Ready",
            "market_prices": "✅ Ready"
        }
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": type(exc).__name__}
    )

if __name__ == "__main__":
    print("🌾 Starting Smart Agriculture Website Backend...")
    print("📡 Server will be available at http://localhost:8000")
    print("📚 API Documentation at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)