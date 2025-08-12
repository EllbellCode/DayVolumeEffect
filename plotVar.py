import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from arch import arch_model

folder_path = 'Data/Verified'  
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=False, sharey=False)
axes = axes.flatten()  # flatten 2D array for easy iteration

colours = plt.get_cmap('rainbow')

for i, (ax, file) in enumerate(zip(axes, csv_files)):

    print(file)
    data = pd.read_csv(file)
    data['Date'] = pd.to_datetime(data['Date'])

    colour = colours(i / len(csv_files))

    base_model = arch_model(data['Log Returns'].dropna(), vol='GARCH', p=1, q=1, mean='Constant')
    res = base_model.fit(disp='off')
    data['cond_var'] = res.conditional_volatility

    ax.plot(data['Date'], data['cond_var'], color=colour, label='Conditional Volatility')

    # Extract ticker from filename
    basename = os.path.basename(file)          
    ticker = basename.replace('Verif_', '')   
    ticker = ticker.replace('.csv', '')   

    ax.set_title(ticker)
    ax.set_ylabel('Conditional Volatility')

    # Date formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    ax.legend(loc='upper left')

# Hide any unused subplots if less than 9 files
for ax in axes[len(csv_files):]:
    ax.axis('off')

plt.tight_layout()
plt.show()