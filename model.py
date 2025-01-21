import os
from datetime import date
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import joblib
dir = "model/classification"
# Create models directory if it doesn't exist
os.makedirs(dir, exist_ok=True)

# Read the dataset
df = pd.read_csv('dataset.csv')

# Convert categorical columns to lowercase
categorical_columns = ['month', 'district', 'quarter', 'food_category', 'food_item']
for column in categorical_columns:
    df[column] = df[column].str.lower()

# Convert 'date' column to numerical format
df['date'] = pd.to_datetime(df['date']).apply(lambda x: x.timestamp())

# Encode categorical variables
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Define features and target (removed quantity)
X = df.drop(columns=['food_item', 'quantity'])  # Remove both food_item and quantity
y = df['food_item']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features for better performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Try different models with tuned parameters
models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
}

# Compare models
best_score = 0
best_model = None
best_model_name = None

print("\nModel Comparison:")
print("-" * 50)

for name, model in models.items():
    # Train and evaluate
    model.fit(X_train_scaled, y_train)
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print(f"\n{name}:")
    print(f"Train Accuracy: {train_score:.4f}")
    print(f"Test Accuracy: {test_score:.4f}")
    
    # Detailed classification report
    y_pred = model.predict(X_test_scaled)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    
    if test_score > best_score:
        best_score = test_score
        best_model = model
        best_model_name = name

print(f"\nBest Model: {best_model_name}")
print(f"Best Test Accuracy: {best_score:.4f}")

# Use the best model
model = best_model

# Save models and preprocessors
model_path = os.path.join(dir, 'food_classifier_model.pkl')
label_encoders_path = os.path.join(dir, 'label_encoders.pkl')
scaler_path = os.path.join(dir, 'scaler.pkl')

joblib.dump(model, model_path)
joblib.dump(label_encoders, label_encoders_path)
joblib.dump(scaler, scaler_path)

# Save model metadata
model_info = {
    'model_name': best_model_name,
    'train_accuracy': train_score,
    'test_accuracy': test_score,
    'feature_columns': X.columns.tolist(),
    'num_classes': len(label_encoders['food_item'].classes_),
    'training_date': str(date.today())
}

joblib.dump(model_info, os.path.join(dir, 'model_info.pkl'))

# Function to predict top n food items
def predict_top_n_food(model, label_encoders, scaler, input_data, n=3):
    for column in input_data:
        if column in label_encoders:
            input_data[column] = label_encoders[column].transform([input_data[column]])[0]
    input_df = pd.DataFrame([input_data])
    input_df = input_df[X.columns]  # Ensure the input data has the same columns as the training data
    input_scaled = scaler.transform(input_df)
    probabilities = model.predict_proba(input_scaled)[0]
    top_n_indices = probabilities.argsort()[-n:][::-1]
    top_n_foods = label_encoders['food_item'].inverse_transform(top_n_indices)
    return top_n_foods

# Example usage with loaded model
def load_model_and_predict(input_data,model_path, n=3):
    model = joblib.load(model_path)
    label_encoders = joblib.load(label_encoders_path)
    scaler = joblib.load(scaler_path)
    return predict_top_n_food(model, label_encoders, scaler, input_data, n)

# Example usage
if __name__ == "__main__":
    input_data = {
        "month": "january",  # lowercase input
        "district": "dhaka",  # lowercase input
        "quarter": "q1",     # lowercase input
        "food_category": "rice",  # lowercase input
        "week_day": 1,
        "date": "2023-06-14"
    }
    input_data['date'] = pd.to_datetime(input_data['date']).timestamp()  # Convert 'date' to Unix timestamp
    predictions = load_model_and_predict(input_data,model_path, n=5)
    print("\nPredicted food items:", predictions)