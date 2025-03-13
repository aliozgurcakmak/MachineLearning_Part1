import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

data = {
    "Deneyim Yılı (x)": [5, 7, 3, 3, 2, 7, 3, 10, 6, 4, 8, 1, 1, 9, 1],
    "Maaş (y)": [600, 900, 550, 500, 400, 950, 540, 1200, 900, 550, 1100, 460, 400, 1000, 380]
}
df = pd.DataFrame(data)

b = 275
w = 90

df["Tahmin Edilen Maaş (y')"] = b + w * df["Deneyim Yılı (x)"]

y_true = df["Maaş (y)"]
y_pred = df["Tahmin Edilen Maaş (y')"]

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)

df["MSE"] = (df["Maaş (y)"] - df["Tahmin Edilen Maaş (y')"]) ** 2
df["RMSE"] = np.sqrt(df["MSE"])
df["MAE"] = abs(df["Maaş (y)"] - df["Tahmin Edilen Maaş (y')"])
df

