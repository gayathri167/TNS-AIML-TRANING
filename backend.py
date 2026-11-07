
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load Walmart dataset
def load_data():

    train = pd.read_csv("dataset/train.csv")
    features = pd.read_csv("dataset/features.csv")
    stores = pd.read_csv("dataset/stores.csv")

    data = pd.merge(train, features, on=["Store", "Date", "IsHoliday"])
    data = pd.merge(data, stores, on="Store")

    data["Date"] = pd.to_datetime(data["Date"])

    return data


# Prepare data
def prepare_data(data):

    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month
    data["Week"] = data["Date"].dt.isocalendar().week.astype(int)

    data["IsHoliday"] = data["IsHoliday"].astype(int)

    data = data.fillna(0)

    data = pd.get_dummies(
        data,
        columns=["Type"],
        drop_first=True,
        dtype=int
    )

    return data


# Train model
def train_model():

    data = load_data()
    data = prepare_data(data)

    features = [
        "Store",
        "Dept",
        "Year",
        "Month",
        "Week",
        "IsHoliday",
        "Temperature",
        "Fuel_Price",
        "CPI",
        "Unemployment",
        "Size"
    ]

    features += [col for col in data.columns if col.startswith("Type_")]

    X = data[features]
    y = data["Weekly_Sales"]

    # Split data
    split = int(len(data) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Model evaluation
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    # Train using complete data
    model.fit(X, y)

    # Save model
    joblib.dump(model, "model.joblib")
    joblib.dump(features, "last_date.joblib")

    return mae, mse, rmse, r2


# Get sales data
def get_sales_data():

    return load_data()


# Predict sales
def predict_sales(store, department, date, holiday):

    model = joblib.load("model.joblib")
    features = joblib.load("last_date.joblib")

    data = load_data()
    data = prepare_data(data)

    # Get store information
    store_data = data[data["Store"] == store]

    if store_data.empty:
        raise ValueError("Store not found")

    row = store_data.iloc[-1]

    # Prepare input data
    input_data = {
        "Store": store,
        "Dept": department,
        "Year": date.year,
        "Month": date.month,
        "Week": date.isocalendar().week,
        "IsHoliday": int(holiday),
        "Temperature": row["Temperature"],
        "Fuel_Price": row["Fuel_Price"],
        "CPI": row["CPI"],
        "Unemployment": row["Unemployment"],
        "Size": row["Size"]
    }

    for col in features:
        if col.startswith("Type_"):
            input_data[col] = row.get(col, 0)

    input_df = pd.DataFrame([input_data])[features]

    prediction = model.predict(input_df)[0]

    return max(0, prediction)
