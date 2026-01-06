import pandas as pd
import numpy as np
import urllib.request
import os

def download_data(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    urllib.request.urlretrieve(url, path)

def load_and_clean(path):
    columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
               "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
    df = pd.read_csv(path, names=columns)
    df.replace("?", np.nan, inplace=True)
    df = df.apply(pd.to_numeric)
    df.fillna(df.median(), inplace=True)
    # Binary classification as per your notebook: target > 0 is 1
    df['target'] = (df['target'] > 0).astype(int)
    return df