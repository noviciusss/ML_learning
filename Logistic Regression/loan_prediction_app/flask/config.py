import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your_default_secret_key'
    MODEL_PATH = os.path.join('models', 'loan_prediction_model.pkl')
    SCALER_PATH = os.path.join('models', 'scaler.pkl')