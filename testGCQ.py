import pandas as pd
import numpy as np
import pandas as pd
from statsmodels.regression.quantile_regression import QuantReg
from sklearn.utils import resample


btc = pd.read_csv('Data/Verified/Verif_BTC.csv')
usdt = pd.read_csv('Data/Verified/Verif_USDT.csv')

y = btc['Close']
x = usdt['Vol_Norm']

max_lag = 2
quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]

# Create lagged variables for y and x
df = pd.DataFrame({'y': y, 'x': x})
for lag in range(1, max_lag + 1):
    df[f'y_lag{lag}'] = df['y'].shift(lag)
    df[f'x_lag{lag}'] = df['x'].shift(lag)

df = df.dropna()

results = {}

for q in quantiles:
    # Restricted model: y ~ past y
    X_r = df[[f'y_lag{i}' for i in range(1, max_lag + 1)]]
    y_r = df['y']
    model_r = QuantReg(y_r, X_r)
    res_r = model_r.fit(q=q)

    # Unrestricted model: y ~ past y + past x
    X_u = df[[f'y_lag{i}' for i in range(1, max_lag + 1)] + [f'x_lag{i}' for i in range(1, max_lag + 1)]]
    y_u = df['y']
    model_u = QuantReg(y_u, X_u)
    res_u = model_u.fit(q=q)

    # Test statistic: compare sum of absolute residuals or residuals difference
    test_stat = np.sum(np.abs(res_r.resid)) - np.sum(np.abs(res_u.resid))

    # Bootstrap to get p-value
    n_boot = 500
    boot_stats = []
    for _ in range(n_boot):
        sample_df = resample(df)
        try:
            mod_r_boot = QuantReg(sample_df['y'], sample_df[[f'y_lag{i}' for i in range(1, max_lag + 1)]]).fit(q=q)
            mod_u_boot = QuantReg(sample_df['y'], sample_df[[f'y_lag{i}' for i in range(1, max_lag + 1)] +
                                                           [f'x_lag{i}' for i in range(1, max_lag + 1)]]).fit(q=q)
            stat_boot = np.sum(np.abs(mod_r_boot.resid)) - np.sum(np.abs(mod_u_boot.resid))
            boot_stats.append(stat_boot)
        except:
            continue

    boot_stats = np.array(boot_stats)
    p_value = np.mean(boot_stats >= test_stat)

    results[q] = {'test_stat': test_stat, 'p_value': p_value}

print(results)