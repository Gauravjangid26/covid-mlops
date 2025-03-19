import sys
import os
import pytest
import joblib
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Dynamically add the src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the trained model from train.py
from src.train import model, BASE_DIR

MODEL_PATH = os.path.join(BASE_DIR, "api", "model.pkl")

def test_model_training():
    """Test if the model is trained properly and saved."""
    assert model is not None, "❌ Model training failed, model is None!"
    assert isinstance(model, (RandomForestClassifier, LogisticRegression)), "❌ Model is not a valid classifier!"

def test_model_prediction():
    """Test if the trained model makes predictions correctly."""
    test_data = np.array([[30,0,30,0,1]])  # Adjust according to dataset
    prediction = model.predict(test_data)
    assert prediction is not None, "❌ Model failed to make a prediction"
    assert isinstance(prediction, (list, tuple, np.ndarray)), "❌ Prediction format incorrect"

def test_model_saving():
    """Test if the model is saved and can be loaded correctly."""
    joblib.dump(model, MODEL_PATH)
    assert os.path.exists(MODEL_PATH), "❌ Model file was not saved"
    
    loaded_model = joblib.load(MODEL_PATH)
    assert loaded_model is not None, "❌ Model saving or loading failed!"
    assert isinstance(loaded_model, (RandomForestClassifier, LogisticRegression)), "❌ Loaded model type mismatch!"

    print("✅ All tests passed successfully!")

if __name__ == "__main__":
    pytest.main()
