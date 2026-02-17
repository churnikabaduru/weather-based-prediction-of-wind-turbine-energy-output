import joblib
import pandas as pd

# Load model
model = joblib.load("power_prediction.sav")

# Create input as DataFrame with column names
input_data = pd.DataFrame(
    [[900, 8.5]],
    columns=["TheoreticalPower", "WindSpeed"]
)

prediction = model.predict(input_data)

print("Predicted Power Output:", prediction[0])
