import os
from datetime import date
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import joblib

# Create models directory for regression
dir = "model/regression"
os.makedirs(dir, exist_ok=True)

# Read the dataset
df = pd.read_csv('dataset.csv')

# Convert categorical columns to lowercase
categorical_columns = ['month', 'district', 'quarter', 'food_category', 'food_item']
for column in categorical_columns:
    df[column] = df[column].str.lower()

# Convert date column
df['date'] = pd.to_datetime(df['date']).apply(lambda x: x.timestamp())

# Encode categorical variables
label_encoders = {}
categorical_columns = ['month', 'district', 'quarter', 'food_category', 'food_item']
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Define features and target for regression
X = df.drop(columns=['quantity'])  # Now quantity is our target
y = df['quantity']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define regression models with tuned parameters
models = {
    'Random Forest': RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
}

# Compare models
best_score = float('-inf')
best_model = None
best_model_name = None

print("\nModel Comparison:")
print("-" * 50)

for name, model in models.items():
    # Train and evaluate
    model.fit(X_train_scaled, y_train)
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_mae = mean_absolute_error(y_test, test_pred)
    
    print(f"\n{name}:")
    print(f"Train R² Score: {train_r2:.4f}")
    print(f"Test R² Score: {test_r2:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    if test_r2 > best_score:
        best_score = test_r2
        best_model = model
        best_model_name = name

print(f"\nBest Model: {best_model_name}")
print(f"Best R² Score: {best_score:.4f}")

# Save models and preprocessors
model_path = os.path.join(dir, 'quantity_predictor_model.pkl')
label_encoders_path = os.path.join(dir, 'quantity_label_encoders.pkl')
scaler_path = os.path.join(dir, 'quantity_scaler.pkl')

joblib.dump(best_model, model_path)
joblib.dump(label_encoders, label_encoders_path)
joblib.dump(scaler, scaler_path)

# Save model metadata
model_info = {
    'model_name': best_model_name,
    'r2_score': best_score,
    'feature_columns': X.columns.tolist(),
    'training_date': str(date.today())
}

joblib.dump(model_info, os.path.join(dir, 'quantity_model_info.pkl'))

def predict_quantity(model, label_encoders, scaler, input_data):
    """Predict quantity for given input data"""
    # Convert input categorical values to lowercase
    for column in input_data:
        if column in label_encoders:
            input_data[column] = label_encoders[column].transform([input_data[column]])[0]
    input_df = pd.DataFrame([input_data])
    input_df = input_df[X.columns]  # Ensure correct column order
    input_scaled = scaler.transform(input_df)
    quantity = model.predict(input_scaled)[0]
    return round(quantity, 2)

def load_model_and_predict_quantity(input_data):
    """Load model and predict quantity"""
    model = joblib.load(model_path)
    label_encoders = joblib.load(label_encoders_path)
    scaler = joblib.load(scaler_path)
    return predict_quantity(model, label_encoders, scaler, input_data)

# Example usage
if __name__ == "__main__":
    input_data = {
        "month": "january",
        "district": "dhaka",
        "quarter": "q1",
        "food_category": "beverages",
        "food_item": "cold coffee",
        "week_day": 1,
        "date": "2023-06-14"
    }
    input_data['date'] = pd.to_datetime(input_data['date']).timestamp()
    predicted_quantity = load_model_and_predict_quantity(input_data)
    print(f"\nPredicted quantity: {predicted_quantity}")
