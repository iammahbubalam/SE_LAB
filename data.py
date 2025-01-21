from datetime import date
import pandas as pd
import random

months = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

quarters = [
    "Q1", "Q2", "Q3"
]
districts = [
    "Barguna", "Barisal", "Bhola", "Jhalokati", "Patuakhali", "Pirojpur",
    "Bandarban", "Brahmanbaria", "Chandpur", "Chattogram", "Cumilla", "Cox's Bazar", "Feni", "Khagrachari", "Lakshmipur", "Noakhali", "Rangamati",
    "Dhaka", "Faridpur", "Gazipur", "Gopalganj", "Kishoreganj", "Madaripur", "Manikganj", "Munshiganj", "Narayanganj", "Narsingdi", "Rajbari", "Shariatpur", "Tangail",
    "Bagerhat", "Chuadanga", "Jashore", "Jhenaidah", "Khulna", "Kushtia", "Magura", "Meherpur", "Narail", "Satkhira",
    "Jamalpur", "Mymensingh", "Netrokona", "Sherpur",
    "Bogura", "Joypurhat", "Naogaon", "Natore", "Chapainawabganj", "Pabna", "Rajshahi", "Sirajganj",
    "Dinajpur", "Gaibandha", "Kurigram", "Lalmonirhat", "Nilphamari", "Panchagarh", "Rangpur", "Thakurgaon",
    "Habiganj", "Moulvibazar", "Sunamganj", "Sylhet"
]

foods = {
    "rice": ["Biryani", "Kachchi Biryani", "Khichuri", "Panta Bhat", "Fried Rice", "Jeera Rice", "Pulao"],
    "beef": ["Beef Bhuna", "Beef Dish", "Beef Curry", "Beef Stir-Fry", "Beef Burger", "Beef Kala Bhuna", "Roasted beef grill", "Beef Stew", "Beef Stroganoff"],
    "chicken": ["Chicken Curry", "BBQ Chicken", "Chicken Wings", "Chicken Soup", "Chicken Tikka", "Chicken Biryani", "Grilled Chicken", "Chicken Alfredo"],
    "fish": ["Fish Bharta", "Fish Pasta", "Crispy Fish Fingers", "Fish Dopiaza", "Fish Grill", "Fish Rezala", "Fish Curry", "Fish Tacos", "Fish Stew"],
    "mutton": ["Mutton Rogan Josh", "Mutton Keema", "Mutton Korma", "Mutton Rezala", "Mutton Curry", "Mutton Biryani", "Mutton Chops", "Mutton Stew"],
    "dessert": ["Falooda", "Tiramisu", "New York Cheesecake", "Apple Pie", "Chocolate Lava Cake", "Brownie", "Ice Cream", "Pudding", "Gulab Jamun"],
    "street food": ["Haleem", "Pani Puri", "Chaat", "Vada Pav", "Kathi Roll", "Dabeli", "Pav Bhaji"],
    "Italian": ["Lasagna", "Pasta", "Pizza", "Risotto", "Gnocchi", "Bruschetta", "Focaccia"],
    "vegetarian": ["Vegetable Curry", "Paneer Butter Masala", "Aloo Gobi", "Palak Paneer", "Chana Masala", "Baingan Bharta", "Vegetable Biryani"],
    "vegan": ["Vegan Burger", "Vegan Salad", "Vegan Curry", "Vegan Pasta", "Vegan Tacos", "Vegan Stir-Fry", "Vegan Soup"],
    "seafood": ["Shrimp Scampi", "Grilled Salmon", "Lobster Bisque", "Crab Cakes", "Seafood Paella", "Fish and Chips", "Seafood Chowder"],
    "snacks": ["Samosa", "Pakora", "Spring Rolls", "Nachos", "Chips", "Popcorn", "Pretzels"],
    "beverages": ["Mango Lassi", "Masala Chai", "Cold Coffee", "Lemonade", "Smoothie", "Iced Tea", "Hot Chocolate"]
}
week_days = [1, 2, 3, 4, 5, 6, 7]
today = date.today()
specific_date = date(2020, 1, 1)

def random_date(start, end):
    return start + (end - start) * random.random()

n = 100000  # Number of rows
data = []
random.seed(42)
for _ in range(n):
    rand_date = random_date(specific_date, today)
    month = months[rand_date.month - 1]
    quarter = quarters[(rand_date.month - 1) // 4]
    district = random.choice(districts)
    food_category = random.choice(list(foods.keys()))
    food_item = random.choice(foods[food_category])
    week_day = rand_date.isoweekday()
    quantity = random.randint(1, 20)

    data.append({
        "date": rand_date,
        "month": month,
        "district": district,
        "quarter": quarter,
        "food_category": food_category,
        "food_item": food_item,
        "week_day": week_day,
        "quantity": quantity
    })

df = pd.DataFrame(data)
df.to_csv('dataset.csv', index=False)

