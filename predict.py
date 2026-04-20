import joblib
import numpy as np

model = joblib.load("model/model.pkl")

# Example input: [square footage, bedrooms, bathrooms]
sample = np.array([[2000, 3, 2]])

prediction = model.predict(sample)

print(f"Predicted House Price: {prediction[0]}")