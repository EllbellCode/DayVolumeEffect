import numpy as np
import pandas as pd
from arch import arch_model
import statsmodels.api as sm

data = pd.read_csv('Data/Verified/Verif_BTC.csv')
#No Sunday to avoid dummy variable trap!
returns = data[["Date", "Vol_Norm", 'Log Returns', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']] 

train = returns[(returns['Date'] >= '2020-01-01') & (returns['Date'] <= '2023-12-31')]
test = returns[returns['Date'] > '2023-12-31']
train = train.dropna()
regressors = train[['Vol_Norm', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']]


# Assume df has 'returns' and 'log_volume'
max_lag = 20

# Step 1: Fit basic GARCH(1,1) to get conditional variances
base_model = arch_model(train['Log Returns'], vol='GARCH', p=1, q=1, mean='Constant')
res = base_model.fit(disp='off')
# What we are trying to predict
train['cond_var'] = res.conditional_volatility

# Step 2: Test multiple lags of volume as predictors of variance
results = []
for lag in range(0, max_lag + 1):  # start at 0 now
    if lag == 0:
        vol_lagged = train['Vol_Norm']  # no shift
    else:
        vol_lagged = train['Vol_Norm'].shift(lag)
    
    tmp = train.copy()
    tmp[f'vol_lag{lag}'] = vol_lagged
    tmp = tmp.dropna(subset=['cond_var', f'vol_lag{lag}'])
    
    X = sm.add_constant(tmp[f'vol_lag{lag}'])
    reg = sm.OLS(tmp['cond_var'], X).fit()
    
    results.append({
        'lag': lag,
        'AIC': reg.aic,
        'BIC': reg.bic,
        'p_value': reg.pvalues[f'vol_lag{lag}'],
        'coef': reg.params[f'vol_lag{lag}']
    })

results_df = pd.DataFrame(results).set_index('lag')
print(results_df)

sig = results_df[results_df['p_value'] < 0.05]

if sig.empty:
    print("No lags have p-value < 0.05")
else:
    for lag, row in sig.iterrows():
        print(f"Lag {lag}: p-value={row['p_value']:.6f}, coef={row['coef']}, BIC={row['BIC']}")
