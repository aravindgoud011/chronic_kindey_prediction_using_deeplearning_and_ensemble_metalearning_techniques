import os
import joblib

def ensure_dirs():
    os.makedirs("artifacts/scalers", exist_ok=True)
    os.makedirs("artifacts/models", exist_ok=True)
    os.makedirs("artifacts/reports", exist_ok=True)

def save_joblib(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)

def load_joblib(path):
    return joblib.load(path)
