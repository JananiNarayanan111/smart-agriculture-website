from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta
import random

router = APIRouter()

class MarketPriceRequest(BaseModel):
    crop: str
    market: str = "Delhi"

class MarketPriceResponse(BaseModel):
    crop: str
    current_price: float
    unit: str
    market: str
    last_updated: str
    price_trend: list
    avg_price_week: float
    price_change_percent: float

# Sample price data (in real scenario, this would come from AGMARKNET API)
MARKET_DATA = {
    "rice": {"delhi": 2500, "bangalore": 2600, "mumbai": 2550, "kolkata": 2450},
    "wheat": {"delhi": 2200, "bangalore": 2300, "mumbai": 2250, "kolkata": 2150},
    "cotton": {"delhi": 5500, "bangalore": 5600, "mumbai": 5400, "kolkata": 5300},
    "sugarcane": {"delhi": 300, "bangalore": 310, "mumbai": 305, "kolkata": 290},
    "maize": {"delhi": 1800, "bangalore": 1850, "mumbai": 1820, "kolkata": 1750},
    "groundnut": {"delhi": 5000, "bangalore": 5100, "mumbai": 4950, "kolkata": 4900},
    "soyabean": {"delhi": 4500, "bangalore": 4600, "mumbai": 4550, "kolkata": 4400},
    "coconut": {"delhi": 8000, "bangalore": 7900, "mumbai": 8100, "kolkata": 8200},
}

@router.post("/current-price", response_model=MarketPriceResponse)
def get_current_price(request: MarketPriceRequest):
    try:
        crop = request.crop.lower()
        market = request.market.lower()
        
        if crop not in MARKET_DATA:
            raise HTTPException(status_code=404, detail=f"Crop '{crop}' not found in database.")
        
        if market not in MARKET_DATA[crop]:
            # Use average price if specific market not found
            current_price = sum(MARKET_DATA[crop].values()) / len(MARKET_DATA[crop])
        else:
            current_price = MARKET_DATA[crop][market]
        
        # Generate price trend (last 7 days)
        trend = [current_price + random.uniform(-100, 100) for _ in range(7)]
        
        # Calculate average and change
        avg_price = sum(trend) / len(trend)
        price_change = ((current_price - trend[0]) / trend[0]) * 100
        
        return MarketPriceResponse(
            crop=crop,
            current_price=round(current_price, 2),
            unit="₹/quintal",
            market=request.market,
            last_updated=datetime.now().isoformat(),
            price_trend=[round(p, 2) for p in trend],
            avg_price_week=round(avg_price, 2),
            price_change_percent=round(price_change, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/all-crops")
def get_all_crops():
    return {"crops": list(MARKET_DATA.keys())}

@router.get("/markets")
def get_markets():
    markets = set()
    for crop_data in MARKET_DATA.values():
        markets.update(crop_data.keys())
    return {"markets": list(markets)}

@router.get("/crop-prices/{crop}")
def get_crop_prices_all_markets(crop: str):
    crop = crop.lower()
    if crop not in MARKET_DATA:
        raise HTTPException(status_code=404, detail=f"Crop '{crop}' not found.")
    
    return {
        "crop": crop,
        "markets": [
            {"market": market, "price": price}
            for market, price in MARKET_DATA[crop].items()
        ]
    }