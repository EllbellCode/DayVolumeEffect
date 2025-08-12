"""
Shows Stationarity in the volume and returns data!
"""


import pandas as pd
from statsmodels.tsa.stattools import adfuller
import glob
import os

folder_path = 'Data/Verified'  
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

# Initialize empty DataFrames to store results
returns_adf = pd.DataFrame(columns=["Coin", "ADF Statistic", "p-value", "Number of Lags"])
vol_adf = pd.DataFrame(columns=["Coin", "ADF Statistic", "p-value", "Number of Lags"])

for file in csv_files:
    data = pd.read_csv(file)

    train = data[(data['Date'] >= '2020-01-01') & (data['Date'] <= '2023-12-31')]

    # Drop NaNs and multiply returns by 1000 to reduce scale warnings
    log_returns = train['Log Returns'].dropna() * 1000
    volume = train['VolLogChange'].dropna()

    returns_result = adfuller(log_returns)
    vol_result = adfuller(volume)

    basename = os.path.basename(file)
    ticker = basename.replace('Verif_', '').replace('.csv', '')

    # Create small DataFrames for this iteration's results
    returns_row = pd.DataFrame([{
        "Coin": ticker,
        "ADF Statistic": returns_result[0],
        "p-value": returns_result[1],
        "Number of Lags": returns_result[2]
    }])

    vol_row = pd.DataFrame([{
        "Coin": ticker,
        "ADF Statistic": vol_result[0],
        "p-value": vol_result[1],
        "Number of Lags": vol_result[2]
    }])

    # Append current results to the overall DataFrames
    returns_adf = pd.concat([returns_adf, returns_row], ignore_index=True)
    vol_adf = pd.concat([vol_adf, vol_row], ignore_index=True)

print('Returns:')
print(returns_adf)

print('Volume:')
print(vol_adf)