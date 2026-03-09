import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import os

def train_yield_prediction_model():
    """
    Train Random Forest Regressor for Yield Prediction
    Dataset: https://www.kaggle.com/datasets/rajyellow46/crop-yield-in-indian-districts
    """
    
    print("🌾 Training Yield Prediction Model (Random Forest Regressor)...")
    
    # Create sample dataset
    crops = ['rice', 'wheat', 'cotton', 'sugarcane', 'maize', 'groundnut', 'soyabean']
    
    data = {
        'crop': np.random.choice(crops, 200),
        'area': np.random.uniform(10, 1000, 200),
        'rainfall': np.random.uniform(50, 250, 200),
        'temperature': np.random.uniform(15, 35, 200),
        'humidity': np.random.uniform(30, 90, 200),
        'nitrogen': np.random.uniform(20, 150, 200),
        'phosphorus': np.random.uniform(10, 100, 200),
        'potassium': np.random.uniform(10, 100, 200),
        'yield': np.random.uniform(1, 10, 200)  # tons/hectare
    }
    
    df = pd.DataFrame(data)
    
    print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Encode crop names
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['crop_encoded'] = le.fit_transform(df['crop'])
    
    # Prepare features and target
    X = df[['crop_encoded', 'area', 'rainfall', 'temperature', 'humidity', 'nitrogen', 'phosphorus', 'potassium']]
    y = df['yield']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"✓ Training set: {X_train.shape[0]} samples")
    print(f"✓ Testing set: {X_test.shape[0]} samples")
    
    # Train Random Forest Regressor
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("✓ Model training completed!")
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📊 Model Performance:")
    print(f"   MSE: {mse:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   R² Score: {r2:.4f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    with open('models/yield_prediction.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("\n✅ Model saved to 'models/yield_prediction.pkl'")
    
    return model

if __name__ == "__main__":
    train_yield_prediction_model()