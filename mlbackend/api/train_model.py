import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Sample dataset
data = {
    'area': [1000, 1200, 900, 1500, 1100],
    'bedrooms': [2, 3, 2, 4, 3],
    'price': [150000, 180000, 135000, 210000, 175000]
}
df = pd.DataFrame(data)

X = df[['area', 'bedrooms']]
y = df['price']

model = LinearRegression()
model.fit(X, y)

# Save model in the same env
joblib.dump(model, 'api/house_model.pkl')
print("✅ Model trained and saved successfully.")