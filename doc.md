# Food Prediction System Documentation

## Project Overview
This project implements a machine learning-based food prediction system that provides two main functionalities:
1. Predicting the most likely food items based on given parameters
2. Predicting the quantity required for specific food items

## System Architecture

### Components
1. **Data Generation Module** (`data.py`)
2. **Model Training Modules**
   - Classification Model (`model.py`)
   - Regression Model (`quantity_predictor.py`)
3. **API Server** (`server.py`)

### Technology Stack
- FastAPI for REST API
- Scikit-learn for ML models
- Pandas for data processing
- Joblib for model serialization

## Detailed Component Description

### 1. Data Generation Module
Location: `data.py`

This module generates synthetic training data with the following features:
- Date-related: date, month, quarter, week_day
- Location: district
- Food-related: food_category, food_item
- Quantity information

The dataset includes:
- 13 different food categories
- 64 districts of Bangladesh
- Dates ranging from 2020 to present
- 100,000 synthetic records

### 2. Model Training Modules

#### Classification Model (`model.py`)
- **Purpose**: Predicts the most likely food items based on input parameters
- **Algorithm**: Random Forest Classifier
- **Features**: month, district, quarter, food_category, week_day, date
- **Target**: food_item
- **Performance Metrics**:
  - Uses train-test split (80-20)
  - Evaluates accuracy and provides detailed classification report
- **Model Storage**: Saves model and preprocessing components in `model/classification/`

#### Quantity Prediction Model (`quantity_predictor.py`)
- **Purpose**: Predicts the required quantity for a specific food item
- **Algorithm**: Random Forest Regressor
- **Features**: All features including food_item
- **Target**: quantity
- **Performance Metrics**:
  - R² Score
  - RMSE (Root Mean Square Error)
  - MAE (Mean Absolute Error)
- **Model Storage**: Saves model and preprocessing components in `model/regression/`

### 3. API Server
Location: `server.py`

#### Endpoints
1. `/predict/food-and-quantity`
   - **Method**: POST
   - **Input**: JSON with date, month, district, quarter, food_category, week_day
   - **Output**: List of predicted food items with their estimated quantities

#### Request Format
```json
{
    "month": "january",
    "district": "dhaka",
    "quarter": "q1",
    "food_category": "beverages",
    "week_day": 1,
    "date": "2023-06-14",
    "num_samples": 3
}
```

#### Response Format
```json
{
    "status": "success",
    "predictions": [
        {
            "food_item": "item1",
            "predicted_quantity": 10.5
        },
        {
            "food_item": "item2",
            "predicted_quantity": 8.2
        }
    ]
}
```

## Model Training and Performance

### Classification Model
- Uses RandomForestClassifier with optimized parameters:
  - n_estimators: 200
  - max_depth: 20
  - min_samples_split: 5
  - min_samples_leaf: 2

### Regression Model
- Uses RandomForestRegressor with similar hyperparameters
- Features standardized using StandardScaler
- Categorical variables encoded using LabelEncoder

## Deployment and Usage

### Prerequisites
- Python 3.7+
- Required packages: fastapi, scikit-learn, pandas, joblib, uvicorn

### Setup Instructions
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Generate dataset: `python data.py`
4. Train models:
   ```bash
   python model.py
   python quantity_predictor.py
   ```
5. Start the server: `python server.py`

### API Usage
The server runs on `http://localhost:8000` by default.

Example curl request:
```bash
curl -X POST "http://localhost:8000/predict/food-and-quantity" \
     -H "Content-Type: application/json" \
     -d '{"month":"january","district":"dhaka","quarter":"q1","food_category":"beverages","week_day":1,"date":"2023-06-14","num_samples":3}'
```

## Error Handling and Logging
- Comprehensive error handling with HTTP status codes
- Logging configuration for debugging and monitoring
- Input validation using Pydantic models

## Future Improvements
1. Add more sophisticated feature engineering
2. Implement model retraining pipeline
3. Add model version control
4. Implement caching for frequent predictions
5. Add authentication and rate limiting
