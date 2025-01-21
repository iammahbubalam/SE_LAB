import requests
import json
from datetime import datetime

def test_api():
    # API endpoint
    url = "http://localhost:8000/predict/food-and-quantity"
    
    # Test cases
    test_cases = [
        {
            "month": "january",
            "district": "dhaka",
            "quarter": "q1",
            "food_category": "rice",
            "week_day": 1,
            "date": "2023-06-14",
            "num_samples": 5
        },
        {
            "month": "june",
            "district": "dhaka",
            "quarter": "q2",
            "food_category": "beverages",
            "week_day": 3,
            "date": "2023-06-15",
            "num_samples": 3
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print("Request:")
        print(json.dumps(test_case, indent=2))
        
        try:
            # Make POST request
            response = requests.post(url, json=test_case)
            
            print("\nResponse Status:", response.status_code)
            print("Response Headers:", dict(response.headers))
            print("\nResponse Body:")
            print(json.dumps(response.json(), indent=2))
            
            if response.status_code == 200:
                # Access specific predictions
                predictions = response.json()["predictions"]
                print("\nPredicted Items:")
                for pred in predictions:
                    print(f"Food Item: {pred['food_item']}, Quantity: {pred['predicted_quantity']}")
                    
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
        
        print("\n" + "="*50)

if __name__ == "__main__":
    test_api()
