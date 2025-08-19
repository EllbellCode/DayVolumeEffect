import pandas as pd
from arch import arch_model 
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm


data = pd.read_csv('Data/Verified/Verif_BTC.csv')
#No Sunday to avoid dummy variable trap!
returns = data[["Date", 'Log Returns', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']] 



train = returns[(returns['Date'] >= '2020-01-01') & (returns['Date'] <= '2023-12-31')]
test = returns[returns['Date'] > '2023-12-31']
train = train.dropna()
# # CODE FOR MODELLING VOLATILITY DIRECT

# model = arch_model(train["Log Returns"], vol='GARCH', p=1, q=1, x=train[['Vol_Norm']], mean='ARX')

# residuals = model.fit()

# print(residuals)

# cond_vol = residuals.conditional_volatility

# regressors_c = sm.add_constant(regressors)

# vol_model = sm.OLS(cond_vol, regressors_c).fit()

# # Show summary
# print(vol_model.summary())

# train['Volatility'] = cond_vol
# train['Day'] = pd.to_datetime(train['Date']).dt.day_name()

# train.groupby('Day')['Vol_Norm'].mean().reindex([
#     'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'
# ]).plot(kind='bar', title='Average Conditional Volatility by Day')
# plt.ylabel('Volatility')
# plt.show()

#ORIGINAL CODE

g_model = arch_model(train["Log Returns"], vol='GARCH', p=1, q=1)

g_model_fit = g_model.fit()

eg_model = arch_model(train["Log Returns"], vol='EGARCH', p=1, q=1)
eg_model_fit = eg_model.fit()

#Print the summary of the model
print(g_model_fit.summary())
print(eg_model_fit.summary())

garch_vol = g_model_fit.conditional_volatility

# plt.figure(figsize=(12, 6))
# plt.plot(train['Date'], train['Log Returns'])
# plt.plot(train['Date'], garch_vol)
# plt.legend()
# plt.show()

# predicted = len(test)
# # Generate forecasts
# forecast = g_model_fit.forecast(horizon=predicted)

# # Extract the predicted values and the actual values
# predicted_volatility = forecast.variance.values[-1, :]  # Forecasted variances
# predicted_returns = forecast.mean.values[-1, :]  # Forecasted means (if applicable)

# # Create a DataFrame to compare actual vs predicted
# results = pd.DataFrame({
#     'Actual': test['Change %'].values,
#     'Predicted': predicted_returns,
#     'Predicted Volatility': predicted_volatility
# }, index=test['Date'].values)

# # Display the results
# print(results)

# Optionally, you can visualize the results

# plt.figure(figsize=(12, 6))
# plt.plot(results['Actual'], label='Actual Returns', color='blue')
# plt.plot(results['Predicted'], label='Predicted Returns', color='orange')
# plt.title('Actual vs Predicted Returns')
# plt.figure(figsize=(12, 6))
# plt.plot(train['Date'], train['Log Returns'])
# plt.legend()
# plt.show()