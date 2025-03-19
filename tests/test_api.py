import joblib
import pandas as pd

def test_model_loading():
    model = joblib.load("api/model.pkl")
    assert model is not None  # Ensure model loads successfully

def test_model_prediction():
    model = joblib.load("api/model.pkl")
    test_data = pd.DataFrame([[30,0,30,0,1]])
    prediction = model.predict(test_data)
    assert prediction is not None  # Ensure prediction works
