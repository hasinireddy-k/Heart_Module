from sklearn.ensemble import RandomForestClassifier
import joblib

def train_progression_model(X, y):
    model = RandomForestClassifier()
    model.fit(X, y)
    joblib.dump(model, "progression_model.pkl")
    return model
