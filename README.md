# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset
To develop a machine learning model using Random Forest Regression to predict environmental parameters such as temperature, PM2.5, and solar radiation based on historical weather data.
```
weather-station-eee-block_2024_07_13.csv
```

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
Here is a precise 4-line algorithm matching your aim:
1. Load the environmental sensor dataset and preprocess it by handling missing values and encoding categorical features.
2. Split the data into input features and target variables (temperature, PM2.5, energy), and normalize the features.
3. Divide the dataset into training and testing sets, then train a Random Forest Regressor on the training data.
4. Predict outputs on the test data and evaluate the model using metrics like MAE and R² score.
```
## Program:
```
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: Muruga S
RegisterNumber:  212225040265
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv("weather-station-eee-block_2024_07_13.csv")


df["time"] = pd.to_datetime(df["time"])

df = df.sort_values("time").reset_index(drop=True)


df["hour"] = df["time"].dt.hour
df["dayofyear"] = df["time"].dt.dayofyear
df["month"] = df["time"].dt.month
df["dayofweek"] = df["time"].dt.dayofweek


df = df.dropna(subset=["tem"])


features = [
    "hum",
    "co2",
    "illumination",
    "pressure",
    "pm2_5",
    "pm10",
    "wind_direction",
    "wind_direction_angle",
    "wind_speed",
    "wind_speed_level",
    "tsr",
    "hour",
    "dayofyear",
    "month",
    "dayofweek"
]

X = df[features]
y = df["tem"]


numeric_features = [
    "hum", "co2", "illumination", "pressure",
    "pm2_5", "pm10", "wind_direction_angle",
    "wind_speed", "wind_speed_level", "tsr",
    "hour", "dayofyear", "month", "dayofweek"
]

categorical_features = ["wind_direction"]


preprocessor = ColumnTransformer([
    (
        "num",
        SimpleImputer(strategy="median"),
        numeric_features
    ),
    (
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        categorical_features
    )
])


rf = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("random_forest", rf)
])


split = int(len(X) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]


model.fit(X_train, y_train)

y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Random Forest Results")
print("---------------------")
print("MAE  :", mae)
print("RMSE :", rmse)
print("R²   :", r2)
*/
```

## Output:
<img width="313" height="115" alt="image" src="https://github.com/user-attachments/assets/efed098d-b9fb-4601-be93-e9f99374304b" />


## Result:
Thus,a python program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm has completed successfully.
