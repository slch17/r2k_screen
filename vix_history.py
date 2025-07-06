import yfinance as yf
import pandas as pd

# Download daily VIX data
vix = yf.download('^VIX', start='2010-01-01', end='2025-01-01', interval='1d')

# Optional: reset index and save to CSV
vix.reset_index(inplace=True)
vix.to_csv('vix_daily.csv', index=False)

import numpy as np
percentiles = np.arange(0, 110, 10)
decile_values = np.percentile(vix['Close'].dropna(), percentiles)

# Create a DataFrame for the decile table
decile_table = pd.DataFrame({
    'Decile': [f'{p}%' for p in percentiles],
    'Close': decile_values.round(2)
})

# see that 90% of time VIX close < 27
print(decile_table.to_string(index=False))

# chart where > 27
import matplotlib.pyplot as plt

vix_timeseries = pd.Series(vix['Close'].values.ravel(), index=vix['Date'])

plt.figure(figsize=(12, 6))
plt.plot(vix_timeseries, label='VIX', color='blue')
# Highlight area under the curve where VIX > 27 (from 0 to VIX)
plt.fill_between(vix_timeseries.index, 0, vix_timeseries.values, where=(vix_timeseries > 27), color='cyan', alpha=1, label='VIX > 27')

# Add a horizontal line at VIX = 27
plt.axhline(27, color='gray', linestyle='--', linewidth=1, label='VIX = 27')

plt.title('VIX Daily Close (2010-2025) with Area Highlighted Where VIX > 27')
plt.xlabel('Date')
plt.ylabel('VIX Close')
plt.legend()
plt.grid(True)
plt.show()

buy_days = []
for i in range(1, len(vix_timeseries)):
    prev_close = vix_timeseries.iloc[i-1]
    curr_close = vix_timeseries.iloc[i]
    # Ensure both values are scalars and not NaN
    if pd.notna(prev_close) and pd.notna(curr_close):
        if curr_close > 27 and prev_close <= 27:
            buy_days.append(vix['Date'].iloc[i])

buy_days= [d for d in buy_days if d < pd.Timestamp('2025-01-01')] # retain only dates before 2020

buy_days_df = pd.DataFrame(buy_days, columns=['Buy Date'])

# Save buy days to CSV
buy_days_df.to_csv('vix_buy_days.csv', index=False)