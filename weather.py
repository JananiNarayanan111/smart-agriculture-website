from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

WEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "your_api_key_here")

class WeatherRequest(BaseModel):
    latitude: float
    longitude: float
    city_name: str = ""

class WeatherResponse(BaseModel):
    city: str
    current_temp: float
    humidity: float
    rainfall: float
    wind_speed: float
    forecast: list

@router.post("/forecast")
def get_weather_forecast(request: WeatherRequest):
    try:
        # Using OpenWeatherMap API
        current_url = f"https://api.openweathermap.org/data/2.5/weather?lat={request.latitude}&lon={request.longitude}&appid={WEATHER_API_KEY}&units=metric"
        forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={request.latitude}&lon={request.longitude}&appid={WEATHER_API_KEY}&units=metric"
        
        current_response = requests.get(current_url)
        forecast_response = requests.get(forecast_url)
        
        if current_response.status_code != 200:
            raise HTTPException(status_code=400, detail="Unable to fetch weather data. Invalid coordinates.")
        
        current_data = current_response.json()
        forecast_data = forecast_response.json()
        
        # Extract current weather
        current = current_data['main']
        weather_desc = current_data['weather'][0]['description']
        
        # Extract forecast
        forecast_list = []
        for item in forecast_data['list'][::8]:  # Every 24 hours
            forecast_list.append({
                "date": item['dt_txt'],
                "temp": item['main']['temp'],
                "description": item['weather'][0]['description'],
                "rainfall": item.get('rain', {}).get('3h', 0),
                "humidity": item['main']['humidity']
            })
        
        return {
            "city": current_data.get('name', request.city_name),
            "current_temp": current['temp'],
            "humidity": current['humidity'],
            "rainfall": current_data.get('rain', {}).get('1h', 0),
            "wind_speed": current_data['wind']['speed'],
            "description": weather_desc,
            "forecast": forecast_list[:5]  # 5-day forecast
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Weather API Error: {str(e)}")

@router.get("/current/{city_name}")
def get_current_weather(city_name: str):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url)
        
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"City '{city_name}' not found.")
        
        data = response.json()
        return {
            "city": data['name'],
            "country": data['sys']['country'],
            "temperature": data['main']['temp'],
            "humidity": data['main']['humidity'],
            "pressure": data['main']['pressure'],
            "wind_speed": data['wind']['speed'],
            "description": data['weather'][0]['description'],
            "clouds": data['clouds']['all']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))