### DEVELOPED BY : KULASEKARAPANDIAN K
### REGISTER NO : 212222240052
#### Date: 

# Ex.No: 03   COMPUTE THE AUTO FUNCTION(ACF)

## AIM:
To compute the AutoCorrelation Function (ACF) of the ETH15M dataset and determine the model type to fit the data.

## ALGORITHM:
1. Import the necessary libraries.
2. Load and preprocess the dataset.
3. Plot the data to visualize the trend.
4. Compute the AutoCorrelation Function (ACF) for the dataset.
5. Fit an Autoregressive (AR) model to the training data.
6. Predict the values using the AR model.
7. Calculate and display performance metrics (MAE, RMSE, Variance).
8. Plot the actual vs predicted values.
9. 
### PROGRAM:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_absolute_error, mean_squared_error


data = pd.read_csv('ETH15M.csv', parse_dates=['dateTime'], index_col='dateTime')
data.dropna(inplace=True)


plt.figure(figsize=(12, 6))
plt.plot(data['Volume'], label='Data')
plt.xlabel('DateTime')
plt.ylabel('Volume')
plt.legend()
plt.title('ETH Volume Data')
plt.show()

train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

y_train = train_data['Volume']
y_test = test_data['Volume']

series = data['Volume']
plot_acf(series, lags=50)
plt.show()

lag_order = 1  # Adjust based on ACF insights
ar_model = AutoReg(y_train, lags=lag_order)
ar_results = ar_model.fit()

y_pred = ar_results.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1, dynamic=False)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
variance = np.var(y_test)

print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'Variance (testing): {variance:.2f}')

plt.figure(figsize=(12, 6))
plt.plot(test_data.index, y_test, label='Actual Volume')
plt.plot(test_data.index, y_pred, label='Predicted Volume', linestyle='--')
plt.xlabel('DateTime')
plt.ylabel('Volume')
plt.legend()
plt.title('ETH Volume Prediction with Autoregressive Model')
plt.show()

```
## OUTPUT:
#### VISUAL REPRESENTATION OF DATASET:
![image](https://github.com/user-attachments/assets/1413e37b-32e5-43a2-916c-f20d8ea5bf87)

#### AUTO CORRELATION : 
![image](https://github.com/user-attachments/assets/44a4fc31-8a51-4395-be61-c44b44d412b7)

#### VALUES OF MAE,RMSE,VARIANCE:
```
Mean Absolute Error (MAE): 15897.12
Root Mean Squared Error (RMSE): 21119.19
Variance (testing): 394806822.81
```

#### AUTOREGRESSIVE MODEL FOR CONSUMPTION PREDICTION :
![image](https://github.com/user-attachments/assets/a665cf3f-76e9-4177-abfe-1228f97d440b)


### RESULT:
Thus, The python code for implementing auto correlation for infy_stock is successfully executed.
