import os
import json
from datetime import datetime

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


DATA_PATH = os.path.join("data", "processed.cleveland.data")
MODEL_OUT = os.path.join("artifacts", "heart_disease_pipeline.joblib")
LOG_OUT = os.path.join("logs", "train_run.json")


def load_data(path: str) -> pd.DataFrame:
    cols = [
        "age","sex","cp","trestbps","chol","fbs","restecg","thalach",
        "exang","oldpeak","slope","ca","thal","target"
    ]
    df = pd.read_csv(path, header=None, names=cols)
    df = df.replace("?", pd.NA)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.fillna(df.median(numeric_only=True))
    return df


def train_pipeline(df: pd.DataFrame):
    X = df.drop("target", axis=1)
    y = (df["target"] > 0).astype(int)   # binary for AUC

    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), X.columns.tolist())]
    )

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", RandomForestClassifier(n_estimators=200, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)

    return pipeline, float(roc_auc)


def main():
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    df = load_data(DATA_PATH)
    model, auc = train_pipeline(df)

    joblib.dump(model, MODEL_OUT)

    run_log = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "data_path": DATA_PATH,
        "model_path": MODEL_OUT,
        "model_params": model.named_steps["model"].get_params(),
        "metrics": {"roc_auc": auc},
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
    }
    with open(LOG_OUT, "w", encoding="utf-8") as f:
        json.dump(run_log, f, indent=2)

    print(f"✅ Training complete | ROC-AUC={auc:.4f}")
    print(f"✅ Model saved to: {MODEL_OUT}")
    print(f"✅ Logs saved to:  {LOG_OUT}")


if __name__ == "__main__":
    main()