import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Sample dummy dataset (for 2nd year level)
data = {
    "age": [25, 45, 52, 23, 40, 60, 48, 33],
    "bp": [120, 140, 150, 110, 130, 160, 145, 125],
    "cholesterol": [180, 220, 250, 170, 200, 270, 230, 190],
    "risk": [0, 1, 1, 0, 1, 1, 1, 0]
}

df = pd.DataFrame(data)

X = df[["age", "bp", "cholesterol"]]
y = df["risk"]

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "heart_model.pkl")
