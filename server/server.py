import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
import joblib
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = FastAPI()

# Model paths
class PredictionInput(BaseModel):
    month: str
    district: str
    quarter: str
    food_category: str
    week_day: int
    date: str
    num_samples: int = 3

# Load models and resources
regression_model_path = "model/regression/quantity_predictor_model.pkl"
regression_label_encoders_path = "model/regression/quantity_label_encoders.pkl"
regression_scaler_path = "model/regression/quantity_scaler.pkl"
regression_model_info_path = "model/regression/quantity_model_info.pkl"

classification_model_path = "model/classification/food_classifier_model.pkl"
classification_label_encoders_path = "model/classification/label_encoders.pkl"
classification_scaler_path = "model/classification/scaler.pkl"
classification_model_info_path = "model/classification/model_info.pkl"

# Load model info and create DataFrame templates
regression_model_info = joblib.load(regression_model_info_path)
classification_model_info = joblib.load(classification_model_info_path)
X_reg = pd.DataFrame(columns=regression_model_info['feature_columns'])
X_clf = pd.DataFrame(columns=classification_model_info['feature_columns'])

def predict_food_items(input_data: dict, n: int) -> List[str]:
    model = joblib.load(classification_model_path)
    label_encoders = joblib.load(classification_label_encoders_path)
    scaler = joblib.load(classification_scaler_path)
    
    processed_data = input_data.copy()
    for column in processed_data:
        if isinstance(processed_data[column], str):
            processed_data[column] = processed_data[column].lower()
        if column in label_encoders:
            processed_data[column] = label_encoders[column].transform([processed_data[column]])[0]
    
    input_df = pd.DataFrame([processed_data])
    input_df = input_df[X_clf.columns]
    input_scaled = scaler.transform(input_df)
    probabilities = model.predict_proba(input_scaled)[0]
    top_n_indices = probabilities.argsort()[-n:][::-1]
    return label_encoders['food_item'].inverse_transform(top_n_indices)

def predict_quantity_for_item(input_data: dict) -> float:
    model = joblib.load(regression_model_path)
    label_encoders = joblib.load(regression_label_encoders_path)
    scaler = joblib.load(regression_scaler_path)
    
    processed_data = input_data.copy()
    for column in processed_data:
        if isinstance(processed_data[column], str):
            processed_data[column] = processed_data[column].lower()
        if column in label_encoders:
            processed_data[column] = label_encoders[column].transform([processed_data[column]])[0]
    
    input_df = pd.DataFrame([processed_data])
    input_df = input_df[X_reg.columns]
    input_scaled = scaler.transform(input_df)
    return round(float(model.predict(input_scaled)[0]), 2)

@app.post("/predict/food-and-quantity")
async def predict_food_and_quantity(input_data: PredictionInput) -> Dict:
    try:
        # Log incoming request
        logging.info(f"Received request with data: {input_data.dict()}")
        
        # Convert input data to dict and process date
        data = input_data.dict()
        data['date'] = pd.to_datetime(data['date']).timestamp()
        
        # Get food predictions
        food_items = predict_food_items(data, data['num_samples'])
        
        # Get quantity predictions for each food item
        results = []
        for food_item in food_items:
            item_data = data.copy()
            item_data['food_item'] = food_item
            quantity = predict_quantity_for_item(item_data)
            results.append({
                "food_item": food_item,
                "predicted_quantity": quantity
            })
        
        # Log response
        logging.info(f"Sending response: {results}")
        
        return {
            "status": "success",
            "predictions": results
        }
    
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)