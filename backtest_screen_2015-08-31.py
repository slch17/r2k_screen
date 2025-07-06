import numpy as np
import pandas as pd
import requests
import json
import math
filename="8.17.25_IWM.csv"

universe_raw = np.genfromtxt(filename, delimiter=',', dtype=str, skip_header=1)
universe=universe_raw.tolist() 

API_TOKEN = '6827ef52222123.02025464' # Or 'demo' for testing AAPL.US, BTC-USD.CC

start_date = pd.to_datetime('2010-01-01')   # Start date for backtest
end_date = pd.to_datetime('2025-01-01')    # End date   
target_date = pd.to_datetime("2018-08-21") ### THIS IS THE TARGET DATE FOR BACKTEST/ DATA COLLECTION

# Start collecting data here. Inputs: symbol, target_date. Output:  total assets, prior total assets, revenues, gross profit, ebit, ocf, 
# Test with AAPL.US  
# def data_collection(symbol, date):

fundamental_data = {}

for i in universe:
    SYMBOL = i
    url_fundamental = f'https://eodhd.com/api/fundamentals/{SYMBOL}?api_token={API_TOKEN}&fmt=json'
    try:   
        response = requests.get(url_fundamental)
        response.raise_for_status()  # Raise an error for bad responses
        data = response.json()
        # Data validation: check for expected keys
        if not data or 'Financials' not in data or not data['Financials']:
            print(f"Invalid or missing data for {SYMBOL}, skipping.")
            continue
    except (requests.exceptions.RequestException, ValueError) as e:
        print(f"Error fetching data for {SYMBOL}: {e}")
        continue    
    fundamental_data[SYMBOL] = data

# Save fundamental_data to CSV
fundamental_df = pd.DataFrame.from_dict(fundamental_data, orient='index')
fundamental_df.to_csv('fundamental_data.csv')

historical_data = {}
for i in universe:
    SYMBOL_h = i
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    url_historical = f'https://eodhd.com/api/eod/{SYMBOL_h}?from={start_date_str}&to={end_date_str}&period=d&api_token={API_TOKEN}&fmt=json'
    try:   
        response_h = requests.get(url_historical)
        response_h.raise_for_status()
        data_h = response_h.json()  
    except (requests.exceptions.RequestException, ValueError) as e:
        print(f"Error fetching data for {SYMBOL_h}: {e}")
        continue    
    historical_data[SYMBOL_h] = response_h.json()

# Save historical_data to CSV
historical_df.to_csv('historical_data.csv')


# Function: Get the adjusted_close price for the exact target_date, if it exists
# Inputs: date, dataframe with 'date' and 'adjusted_close'
def get_adj_close_price(date, historical_df):
    adj_close_price = None
    row = historical_df[historical_df['date'] == date]
    if not row.empty:
        adj_close_price = row.iloc[0]['adjusted_close']
        return adj_close_price
    if date < historical_df['date'].min():
        return None
    else:
        return get_adj_close_price(date-pd.Timedelta(days=1), historical_df)

def safe_float(val):
    return float(val) if not pd.isna(val) and val is not None else np.nan

#Function: to collect data for each Symbol in the universe... Mean to use this on each symbol in the universe
def collect_data(SYMBOL, target_date, fundamental_data, historical_data):
    try:
        # Get balance sheet data: 
        quarter_bs = fundamental_data[SYMBOL]['Financials']['Balance_Sheet']['quarterly']
        quarter_bs_df = pd.DataFrame(quarter_bs).T

        # Keep only rows where the index is a valid date
        is_date = quarter_bs_df.index.to_series().apply(lambda x: pd.to_datetime(x, errors='coerce')).notna()
        quarter_bs_df = quarter_bs_df[is_date]
        quarter_bs_df['filing_date'] = pd.to_datetime(quarter_bs_df.index)

        quarter_bs_df2 = quarter_bs_df[quarter_bs_df['filing_date'] < target_date][:5][['totalAssets', 'totalLiab', 'commonStockSharesOutstanding', 'netDebt']]

        if len(quarter_bs_df2) < 5:
            print(f"Not enough balance sheet data for {SYMBOL}, skipping.")
            return [SYMBOL, target_date] + [np.nan]*(len(data_df.columns)-2)

        # Get balance sheet data:  
        total_assets_value = safe_float(quarter_bs_df2['totalAssets'].iloc[0])
        prior_total_assets = safe_float(quarter_bs_df2['totalAssets'].iloc[4])
        total_liab_value = safe_float(quarter_bs_df2['totalLiab'].iloc[0])
        shares_outstanding = safe_float(quarter_bs_df2['commonStockSharesOutstanding'].iloc[0])
        net_debt = safe_float(quarter_bs_df2['netDebt'].iloc[0])

        # Get income statement data:
        quarter_is_df = pd.DataFrame(fundamental_data[SYMBOL]['Financials']['Income_Statement']['quarterly']).T
        is_date = quarter_is_df.index.to_series().apply(lambda x: pd.to_datetime(x, errors='coerce')).notna()
        quarter_is_df = quarter_is_df[is_date]
        quarter_is_df['filing_date'] = pd.to_datetime(quarter_is_df.index)
        quarter_is_df2 = quarter_is_df[quarter_is_df['filing_date'] < target_date][:4]

        ltm_revenue = quarter_is_df2['totalRevenue'].astype(float).sum()  
        ltm_gross_profit = quarter_is_df2['grossProfit'].astype(float).sum()  
        ltm_ebit = quarter_is_df2['ebit'].astype(float).sum()   
        ltm_ni = quarter_is_df2['netIncome'].astype(float).sum()   

        # Get last 4 quarters' operating cash flow
        quarter_cf_df = pd.DataFrame(fundamental_data[SYMBOL]['Financials']['Cash_Flow']['quarterly']).T
        is_date_cf = quarter_cf_df.index.to_series().apply(lambda x: pd.to_datetime(x, errors='coerce')).notna()
        quarter_cf_df = quarter_cf_df[is_date_cf]
        quarter_cf_df['filing_date'] = pd.to_datetime(quarter_cf_df.index)
        quarter_cf_df2 = quarter_cf_df[quarter_cf_df['filing_date'] < target_date][:4]
        ltm_ocf = quarter_cf_df2['totalCashFromOperatingActivities'].astype(float).sum()

        # Get Price as at target_date    
        historical_df = pd.DataFrame(historical_data[SYMBOL])
        historical_df['date'] = pd.to_datetime(historical_df['date'])
        historical_df[historical_df['date'] == target_date]

        price = get_adj_close_price(target_date, historical_df)
        # HOW IS get_adj_close_price getting price data from right symbol when not specified?
        if price is None or pd.isna(price):
            print(f"No price data for {SYMBOL} at {target_date}, skipping.")
            return [SYMBOL, target_date] + [np.nan]*(len(data_df.columns)-2)
        # Get valuation data: Price to Book, Forward PE, EV to Revenue    
        market_cap = price * shares_outstanding
        book_value = total_assets_value - total_liab_value
        
        if market_cap == 0 or pd.isna(market_cap):
            book_to_price = np.nan
        else:
            book_to_price = book_value / market_cap
        
        if ltm_ni == 0 or pd.isna(ltm_ni):
            pe_ttm = np.nan
        else:
            pe_ttm = market_cap / ltm_ni

        if (market_cap+net_debt) == 0 or pd.isna(market_cap+net_debt):
            sales_to_ev = np.nan
        else:
            sales_to_ev = ltm_revenue / (market_cap + net_debt)

        # Clean up: consolidate collected data into a single row dataFrame
        output = [SYMBOL, target_date, total_assets_value, prior_total_assets, ltm_revenue, ltm_gross_profit, ltm_ebit, ltm_ni, ltm_ocf, book_to_price, pe_ttm, sales_to_ev]
        return output
    except Exception as e:
        print(f"Error for {SYMBOL}: {e}, skipping.")
        return [SYMBOL, target_date] + [np.nan]*(len(data_df.columns)-2)

columns = ['symbol', 'target_date', 'total_assets_value', 'prior_total_assets','ltm_revenue', 'ltm_gross_profit', 'ltm_ebit', 'ltm_ni', 'ltm_ocf','book_to_price', 'pe_ttm', 'sales_to_ev']
data_df=pd.DataFrame(columns=columns)

for symbol in universe:
    data = collect_data(symbol, target_date, fundamental_data, historical_data)
    data_df.loc[len(data_df)] = data

data_df['asset_growth']=data_df['total_assets_value'] / data_df['prior_total_assets'] - 1
data_df['accruals']=data_df['ltm_ebit'] - data_df['ltm_ocf'] 
#only calculate gross_margin if ltm_revenue is not zero to avoid division by zero:

def calc_gross_margin(row):
    if row['ltm_revenue'] != 0:
        return row['ltm_gross_profit'] / row['ltm_revenue']
    else:
        return np.nan
data_df['gross_margin'] = np.nan  # Initialize the column with NaN
for i in range(len(data_df)):
    calc_gross_margin_value = calc_gross_margin(data_df.iloc[i])
    data_df.at[i, 'gross_margin'] = calc_gross_margin_value

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

def z_score(x):
    std = np.nanstd(x)
    mean = np.nanmean(x)
    if np.isnan(std) or std == 0:
        return np.full_like(x, np.nan, dtype=np.float64)
    return (x - mean) / std

for col in ['book_to_price', 'pe_ttm', 'sales_to_ev']:
    data_df[f'z_{col}'] = z_score(data_df[col])

#data_df = pd.read_csv('data_df.csv')
for col in ['accruals', 'gross_margin', 'asset_growth']:
    data_df[f'z_{col}'] = z_score(data_df[col])
    data_df[col] = pd.to_numeric(data_df[col], errors='coerce')

data_df['z_quality'] = (-data_df['z_accruals'] + data_df['z_gross_margin'] - data_df['z_asset_growth']) / 3 # note negative signs where higher=> poorer
data_df['z_value'] = (data_df['z_book_to_price'] + data_df['z_pe_ttm'] + data_df['z_sales_to_ev']) / 3 # note higher => more value

data_df.to_csv('data_df.csv', index=False)

wgt_value=0.5 # note, test weight around 0.5, 0.6, 0.7, 0.8, 0.9
wgt_quality=1-wgt_value

data_df['score']=wgt_value*data_df['z_value'] + wgt_quality*data_df['z_quality']
data_df=data_df.sort_values(by='score', ascending=False)

def portfolio_selection(data_df, percent=0.05):
    # Sort by score and select top 5%
    n =math.ceil(len(data_df) * percent)
    selected = data_df.sort_values(by='score', ascending=False).head(n)
    return selected['symbol'].tolist()

test_portfolio=portfolio_selection(data_df,40/len(data_df)) # take top 40 only

def equal_weighted_portfolio(selected_symbols):
    # Create an equal-weighted portfolio; output is symbol: weight
    portfolio = {symbol: 1/len(selected_symbols) for symbol in selected_symbols}
    return portfolio

equal_weighted_portfolio(test_portfolio)

def forward_returns(portfolio, historical_data, target_date, months=36):
    # Calculate forward returns for the next 36 months; should test across varying
    target_date = pd.to_datetime(target_date)
    end_date = target_date + pd.DateOffset(months=months)
    wgt=len(portfolio)
    
    forward_returns = {}
    for symbol in portfolio:
        historical_df = pd.DataFrame(historical_data[symbol])
        historical_df['date'] = pd.to_datetime(historical_df['date'])
        price_start = historical_df[historical_df['date'] == target_date]['adjusted_close'].values
        price_end = historical_df[historical_df['date'] == end_date]['adjusted_close'].values
        
        if price_end.size ==0:
            price_end = historical_df[historical_df['date'] == end_date - pd.DateOffset(days=1)]['adjusted_close'].values
            if price_end.size ==0:
                price_end = historical_df[historical_df['date'] == end_date - pd.DateOffset(days=1)]['adjusted_close'].values

        if price_start.size > 0 and price_end.size > 0:
            forward_returns[symbol] = ((price_end[0] - price_start[0]) / price_start[0])
        total_weighted_return = sum(forward_returns.values()) / wgt if wgt > 0 else 0
        forward_returns['total_weighted_return'] = total_weighted_return
    
    return forward_returns

forward_returns(test_portfolio, historical_data, target_date, 36)

price_history = {}

# ensure every symbol in historical_data has a row for every date up to end date
for symbol in historical_data:
    df = pd.DataFrame(historical_data[symbol])
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        # Create a full date range from the earliest date in the data to end_date
        full_range = pd.date_range(start=df['date'].min(), end=end_date, freq='D')
        df = df.set_index('date').reindex(full_range)
        df.index.name = 'date'
        # Forward fill adjusted_close manually
        adj_close = df['adjusted_close'].values
        last_valid = np.nan
        for i in range(len(adj_close)):
            if pd.isna(adj_close[i]):
                adj_close[i] = last_valid
            else:
                last_valid = adj_close[i]
        df['adjusted_close'] = adj_close
        df = df.reset_index()
        historical_data[symbol] = df.to_dict(orient='records')
#### testing this part

def price_history(test_portfolio, historical_data, target_date=target_date):
    # Create a dictionary of DataFrames with date and adjusted_close for each symbol
    price_history = {}

    for symbol in test_portfolio:
        historical_df = pd.DataFrame(historical_data[symbol])
        historical_df['date'] = pd.to_datetime(historical_df['date'])
        historical_df = historical_df[historical_df['date'] >= target_date]
        historical_df = historical_df[['date', 'adjusted_close']].set_index('date')
        
        # Manually forward-fill NaN in adjusted_close
        adj_close = historical_df['adjusted_close'].values
        last_valid = np.nan
        
        for i in range(len(adj_close)):
            
            if pd.isna(adj_close[i]):
                adj_close[i] = last_valid
            else:
                last_valid = adj_close[i]
        historical_df['adjusted_close'] = adj_close

        price_history[symbol] = historical_df

        base_price = price_history[symbol]['adjusted_close'].iloc[0]
        price_history[symbol]['return'] = price_history[symbol]['adjusted_close'] / base_price
        

    wgt = 1 / len(test_portfolio)
    price_history['total_return'] = sum(price_history[symbol]['return'] * wgt for symbol in test_portfolio)
    return price_history

cumul_return=(price_history(test_portfolio, historical_data,target_date))
cumul_return_df = pd.concat(cumul_return.values(), axis=1, keys=cumul_return.keys(), join='outer')

url_iwm = f'https://eodhd.com/api/eod/IWM?from={start_date_str}&to={end_date_str}&period=d&api_token={API_TOKEN}&fmt=json'
response_iwm = requests.get(url_iwm)
response_iwm.raise_for_status()
data_iwm = response_iwm.json()  
data_iwm_df = pd.DataFrame(data_iwm)
# Convert IWM data to DataFrame
iwm=['IWM']
data_iwm_dict = {'IWM': data_iwm}
iwm_return = price_history(iwm, data_iwm_dict, target_date)
#concat  iwm_return with cumul_return_df
# Convert IWM return to DataFrame
iwm_return_df = iwm_return['IWM'].copy()
iwm_return_df = iwm_return_df.reset_index()  # Ensure 'date' is a column
iwm_return_df = iwm_return_df[['date', 'return']].rename(columns={'return': 'iwm_return'})

# Reset index of cumul_return_df if needed
if 'date' not in cumul_return_df.columns:
    cumul_return_df = cumul_return_df.reset_index()

# Merge IWM return into cumul_return_df on 'date'
if isinstance(cumul_return_df.columns, pd.MultiIndex):
    cumul_return_df.columns = ['_'.join([str(i) for i in col if i]) for col in cumul_return_df.columns.values]

cumul_return_df = cumul_return_df.merge(iwm_return_df, on='date', how='left')
cumul_return_df['iwm_return'] = cumul_return_df['iwm_return'].ffill()
cumul_return_df.to_csv('cumul_return_2018-08.csv')

import matplotlib.pyplot as plt
# Ensure 'date' is the index
if 'date' in cumul_return_df.columns:
    cumul_return_df = cumul_return_df.set_index('date')

plt.figure(figsize=(10, 6))
cumul_return_df['total_return_return'].plot(label='Portfolio Total Return', linewidth=2, color='black')
cumul_return_df['iwm_return'].plot(label='IWM Total Return', linewidth=2, color='blue')
plt.title('Portfolio vs. IWM Total Return Over Time')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.show()
