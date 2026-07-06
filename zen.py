from fastapi import FastAPI
from pydantic import create_model
import pandas as pd
import sqlite3
import joblib
from datetime import datetime


model = joblib.load("credit_model.pkl")
feature_names = joblib.load("features.pkl")


app = FastAPI(
    title="Credit Prediction API"
)


fields = {}

for feature in feature_names:
    fields[feature] = (float, ...)

PredictionInput = create_model(
    "PredictionInput",
    **fields
)


conn = sqlite3.connect(
    "predictions.db",
    check_same_thread=False
)

cursor = conn.cursor()

columns_sql = ", ".join(
    [f'"{col}" REAL' for col in feature_names]
)

cursor.execute(f"""
CREATE TABLE IF NOT EXISTS predictions(

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    {columns_sql},

    prediction INTEGER,

    prediction_time TEXT

)
""")

conn.commit()


@app.get("/")
def home():

    return {
        "message": "Credit Prediction API Running"
    }


@app.post("/predict")
def predict(data: PredictionInput):

    data_dict = data.model_dump()

    df = pd.DataFrame([data_dict])

    df = df[feature_names]

    # Predict
    prediction = int(model.predict(df)[0])

    # Save prediction
    values = [data_dict[col] for col in feature_names]
    values.append(prediction)
    values.append(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    columns = ",".join(feature_names)
    columns += ",prediction,prediction_time"

    placeholders = ",".join(
        ["?"] * len(values)
    )

    cursor.execute(
        f"""
        INSERT INTO predictions ({columns})
        VALUES ({placeholders})
        """,
        values
    )

    conn.commit()

    df = pd.read_sql("SELECT * FROM predictions", conn)
    df.to_csv("predictions.csv", index=False)

    return {
    "prediction": prediction,
    "message": "Prediction stored successfully."
}

@app.get("/predictions")
def get_predictions():

    df = pd.read_sql(
        "SELECT * FROM predictions",
        conn
    )

    return df.to_dict(
        orient="records"
    )


@app.get("/health")
def health():

    return {
        "status": "API Running"
    }