import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

def train_pipeline(df):
    X = df.drop("target", axis=1)
    y = df["target"]
    
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), X.columns.tolist())]
    )
    
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", RandomForestClassifier(n_estimators=200, random_state=42))
    ])
    
    pipeline.fit(X, y)
    joblib.dump(pipeline, "heart_disease_pipeline.joblib")
    return pipeline