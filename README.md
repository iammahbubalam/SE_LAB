# Food Item Prediction System

This project implements a machine learning system for predicting food items and their quantities.

## Setup Instructions

### 1. Create and Activate Virtual Environment

For Linux/Mac:
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate
```

For Windows:
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate
```

### 2. Install Dependencies

Create requirements.txt and install dependencies:

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Project

Execute the following commands in order:

1. Train the classification model:
```bash
python model.py
```

2. Train the regression model:
```bash
python quantity_predictor.py
```

3. Start the API server:
```bash
python server/server.py
```

4. In a new terminal (with venv activated), test the API:
```bash
python server/test_api.py
```

## Project Structure

```
se/
├── .venv/                  # Virtual environment
├── model/                  # Saved models
│   ├── classification/     # Classification model files
│   └── regression/        # Regression model files
├── server/                 # API server files
├── model.py               # Food item classifier
├── quantity_predictor.py  # Quantity predictor
├── dataset.csv           # Training data
└── README.md             # This file
```

## API Endpoints

### POST /predict/food-and-quantity

Example request:
```json
{
    "month": "january",
    "district": "dhaka",
    "quarter": "q1",
    "food_category": "rice",
    "week_day": 1,
    "date": "2023-06-14",
    "num_samples": 5
}
```

## Notes

- Ensure all categorical inputs are lowercase
- The server runs on http://localhost:8000
- API documentation available at http://localhost:8000/docs
