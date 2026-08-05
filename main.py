from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd
import numpy as np
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# Load files
# ==========================

model = joblib.load("final_model.pkl")

encoder = joblib.load("encoder.pkl")

columns = joblib.load("columns.pkl")

df = pd.read_csv("Clean_Dataset.csv")
df = df.drop(
    columns=["Unnamed: 0", "flight"],
    errors="ignore")

# ==========================
# mappings
# ==========================

stops_mapping = {
    "zero":0,
    "one":1,
    "two_or_more":2
}

class_mapping = {
    "Economy":0,
    "Business":1
}

time_mapping = {
    "Early_Morning":0,
    "Morning":1,
    "Afternoon":2,
    "Evening":3,
    "Night":4,
    "Late_Night":5
}

# ==========================

class FlightInput(BaseModel):

    airline:str
    source_city:str
    destination_city:str

    departure_time:str
    arrival_time:str

    stops:str

    flight_class:str

    duration:float

    days_left:int


@app.get("/")
def home():

    return {
        "status": "running",
        "message": "Flight Price Prediction API is running"
    }


@app.post("/predict")
def predict(data:FlightInput):

    try:

        row = pd.DataFrame([{

            "stops":stops_mapping[data.stops],

            "class":class_mapping[data.flight_class],

            "departure_time":time_mapping[data.departure_time],

            "arrival_time":time_mapping[data.arrival_time],

            "duration":data.duration,

            "days_left":data.days_left,

            "airline":data.airline,

            "source_city":data.source_city,

            "destination_city":data.destination_city

        }])

        # OneHotEncoder
        nominal = ["airline","source_city","destination_city"]

        encoded = encoder.transform(row[nominal])

        encoded = pd.DataFrame(
            encoded,
            columns=encoder.get_feature_names_out(nominal)
        )

        row = row.drop(columns=nominal)

        row = pd.concat(
            [row.reset_index(drop=True), encoded],
            axis=1
        )

    

        row = row.reindex(columns=columns, fill_value=0)

        pred_log = model.predict(row)[0]
        
        pred = np.expm1(pred_log)

        return {
            "predicted_price":round(float(pred),2)
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

@app.get("/correlations")
def correlations():

    try:

        corr_df = df.corr(
            numeric_only=True
        )

        return {

            "columns":
                corr_df.columns.tolist(),

            "matrix":
                corr_df.values.tolist()

        }


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Correlation error: {str(e)}"
        )

@app.get("/feature-importance")
def feature_importance():

    try:

        importance = pd.DataFrame({

            "Feature": columns,

            "Importance":
                model.feature_importances_

        })


        # ==========================
        # Group One-Hot Encoded Features
        # ==========================

        importance["Feature"] = (
            importance["Feature"].replace(
                {
                    r"^airline_.*":
                        "Airline",

                    r"^source_city_.*":
                        "Source City",

                    r"^destination_city_.*":
                        "Destination City"
                },
                regex=True
            )
        )


        grouped_importance = (

            importance

            .groupby(
                "Feature",
                as_index=False
            )["Importance"]

            .sum()

            .sort_values(
                by="Importance",
                ascending=False
            )

        )


        # Convert to Percentage

        grouped_importance["Importance"] = (

            grouped_importance["Importance"]

            * 100

        )


        return {

            "features":
                grouped_importance[
                    "Feature"
                ].tolist(),

            "importance":
                grouped_importance[
                    "Importance"
                ]
                .round(2)
                .tolist()

        }


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Feature importance error: {str(e)}")