import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

os.makedirs("model", exist_ok=True)

# Load dataset
df = pd.read_csv("data/electricity.csv")

# Features & target
X = df.drop("bill", axis=1)
y = df["bill"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Accuracy
score = model.score(X_test, y_test)
print("Model Accuracy:", score)

# Save model
joblib.dump(model, "model/bill_model.pkl")

print("Model saved successfully!")