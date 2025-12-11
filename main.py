from fastapi import FastAPI
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime, timedelta
import math
import argparse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware

app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # This allows all origins, change to specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods like GET, POST, etc.
    allow_headers=["*"],  # Allows all headers
)

# Load dataset
df = pd.read_csv("sales_history.csv")
df['sale_date'] = pd.to_datetime(df['sale_date'])

@app.get("/")
def home():
    return {"message": "Hello! Welcome to Retail Demand Forecasting API"}

@app.get("/forecast")
def forecast(days: int = 7):
    today = datetime.today().date()
    results = []

    # Loop through each product
    for product in df["product_name"].unique():

        product_df = df[df["product_name"] == product].copy()

        # Aggregate by day
        daily = (
            product_df.groupby("sale_date")["quantity"]
            .sum()
            .asfreq("D")
            .fillna(0)
        )

        # Not enough data → skip
        if len(daily) < 10:
            continue

        # Handle GAP between last sale date and today
        last_date = daily.index.max().date()
        gap_days = (today - last_date).days

        total_days_to_predict = gap_days + days

        # Train model
        model = ExponentialSmoothing(daily, trend="add").fit()
        pred = model.forecast(total_days_to_predict)

        # Remove gap predictions
        pred = pred[-days:]

        # Sum predictions for total quantity
        forecast_total_quantity = sum(int(math.ceil(q)) for q in pred)

        # Last day of forecast window
        forecast_end_date = today + timedelta(days=days)

        results.append({
            "product_name": product,
            "forecast_date": forecast_end_date.strftime("%Y-%m-%d"),
            "forecast_quantity": forecast_total_quantity
        })

    return {
        "forecast_horizon": days,
        "generated_from_date": today.strftime("%Y-%m-%d"),
        "products": results
    }


# -------------------------
# MAIN ENTRY POINT
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=2727, help="Port to run the API on")
    args = parser.parse_args()

    uvicorn.run("main:app", host="0.0.0.0", port=args.port, reload=True)