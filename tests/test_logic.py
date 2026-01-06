import joblib
import pandas as pd
import numpy as np
import os

def test_model_loading():
    """Check if the joblib file exists and can be loaded"""
    model_path = "heart_disease_pipeline.joblib"
    assert os.path.exists(model_path), "Model file is missing!"
    model = joblib.load(model_path)
    assert hasattr(model, 'predict'), "Loaded object is not a valid model"

def test_data_cleaning_logic():
    """Test the null-handling logic used in your notebook"""
    # Simulate the '?' handling from your notebook
    raw_data = pd.DataFrame({'val': ['1.0', '2.0', '?', '4.0']})
    raw_data.replace('?', np.nan, inplace=True)
    raw_data = raw_data.apply(pd.to_numeric)
    
    # Fill with median as done in your code
    raw_data.fillna(raw_data.median(), inplace=True)
    
    assert raw_data.isnull().sum().sum() == 0
    assert raw_data.iloc[2, 0] == 2.0 # Median of 1, 2, 4 is 2