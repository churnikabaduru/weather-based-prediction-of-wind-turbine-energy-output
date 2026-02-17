# Import Libraries
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("data/T1.csv")

# Rename columns properly
df.rename(columns={
    'Date/Time': 'Time',
    'LV ActivePower (kW)': 'ActivePower',
    'Wind Speed (m/s)': 'WindSpeed',
    'Theoretical_Power_Curve (KWh)': 'TheoreticalPower'
}, inplace=True)

# Drop unnecessary columns
df.drop(['Wind Direction (°)', 'Time'], axis=1, inplace=True)

# Remove null values
df = df.dropna()

# Define Features and Target
X = df[['TheoreticalPower', 'WindSpeed']]
y = df['ActivePower']

# Split Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train Model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

model = RandomForestRegressor(n_estimators=200, random_state=0)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))

# 📊 Plot Graph (AFTER prediction)
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Power")
plt.ylabel("Predicted Power")
plt.title("Actual vs Predicted Power Output")
plt.show()

# Save Model
joblib.dump(model, "power_prediction.sav")
print("Model Saved Successfully!")
