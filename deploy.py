from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle

app = FastAPI(title="House Price Prediction API")
with open("scaler_weights.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model_weights.pkl", "rb") as f:
    model = pickle.load(f)

latest_input = None


class HouseFeatures(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int

    sqft_above: int
    sqft_basement: int
    yr_built: int
    yr_renovated: int
    zipcode: int
    lat: float
    long: float
    sqft_living15: int
    sqft_lot15: int
    year: int
    month: int


FEATURE_COLUMNS = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "zipcode",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
    "year",
    "month",
]


@app.post("/input")
def take_input(data: HouseFeatures):
    global latest_input
    latest_input = data.model_dump()
    return {
        "message": "Input received successfully",
        "stored_input": latest_input
    }


@app.get("/predict")
def get_prediction():
    global latest_input

    if latest_input is None:
        raise HTTPException(
            status_code=400,
            detail="No user input found. Send data first to POST /input"
        )

    try:
        input_df = pd.DataFrame([latest_input])
        input_df = input_df[FEATURE_COLUMNS]

        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)

        predicted_price = float(prediction[0][0])

        return {
            "input_data": latest_input,
            "predicted_price": predicted_price
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
