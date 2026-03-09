import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

def train_crop_recommendation_model():
    """
    Train Random Forest Classifier for Crop Recommendation
    Dataset: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
    """
    
    print("🌱 Training Crop Recommendation Model (Random Forest Classifier)...")
    
    # Create sample dataset (in production, load from CSV)
    # Dataset columns: N, P, K, temperature, humidity, ph, rainfall, label
    
    data = {
        'N': np.random.uniform(0, 140, 100),
        'P': np.random.uniform(5, 145, 100),
        'K': np.random.uniform(5, 205, 100),
        'temperature': np.random.uniform(8, 43, 100),
        'humidity': np.random.uniform(14, 99, 100),
        'ph': np.random.uniform(3.5, 9.5, 100),
        'rainfall': np.random.uniform(20, 225, 100),
        'label': np.random.choice([
            'rice', 'maize', 'jute', 'cotton', 'coconut', 'sugarcane',
            'groundnut', 'soyabean', 'sorghum', 'wheat', 'barley',
            'mungbean', 'masoor', 'mothbean', 'uradbean', 'pigeonpea',
            'kidneybeans', 'chickpea', 'lentil', 'pomegranate', 'banana',
            'mango', 'grapes', 'watermelon', 'muskmelon', 'apple',
            'orange', 'papaya', 'coconut', 'cotton'
        ], 100)
    }
    
    df = pd.DataFrame(data)
    
    print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"✓ Features: {df.columns.tolist()}")
    
    # Prepare features and labels
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']
    
    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    
    print(f"✓ Training set: {X_train.shape[0]} samples")
    print(f"✓ Testing set: {X_test.shape[0]} samples")
    
    # Train Random Forest Classifier
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("✓ Model training completed!")
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n📊 Model Performance:")
    print(f"   Accuracy: {accuracy * 100:.2f}%")
    print(f"\n   Classification Report:\n{classification_report(y_test, y_pred, target_names=le.classes_)}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    with open('models/crop_recommendation.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("\n✅ Model saved to 'models/crop_recommendation.pkl'")
    
    return model

if __name__ == "__main__":
    train_crop_recommendation_model()