import numpy as np
import pandas as pd
import requests
import json

filename="6.13.25_IWM.csv"

universe_raw = np.genfromtxt(filename, delimiter=',', dtype=str, skip_header=1)
universe=['AAPL','MSFT','TSLA']
#universe=universe_raw.tolist() .... USE THIS LINE WHEN READY TO USE THE FULL UNIVERSE

API_TOKEN = 'demo' # '6827ef52222123.02025464' # Or 'demo' for testing AAPL.US, BTC-USD.CC
start_date = pd.to_datetime('2010-01-01')   # Start date for backtest
end_date = pd.to_datetime('2021-01-01')    # End date   
target_date = pd.to_datetime("2019-01-04") ### THIS IS THE TARGET DATE FOR BACKTEST/ DATA COLLECTION

# Start collecting data here. Inputs: symbol, target_date. Output:  total assets, prior total assets, revenues, gross profit, ebit, ocf, 
# Test with AAPL.US  
# def data_collection(symbol, date):

fundamental_data = {}
for i in universe:
    SYMBOL = i
    
    url_fundamental = f'https://eodhd.com/api/fundamentals/{SYMBOL}?api_token={API_TOKEN}&fmt=json'
    
    response = requests.get(url_fundamental)
    fundamental_data[SYMBOL] = response.json()

historical_data = {}
for i in universe:
    SYMBOL = i
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    url_historical = f'https://eodhd.com/api/eod/{SYMBOL}?from={start_date_str}&to={end_date_str}&period=d&api_token={API_TOKEN}&fmt=json'
    
    response = requests.get(url_historical)
    historical_data[SYMBOL] = response.json()

# Function: Get the adjusted_close price for the exact target_date, if it exists
# Inputs: date, dataframe with 'date' and 'adjusted_close'
def get_adj_close_price(date, historical_df):
    adj_close_price = None
    row = historical_df[historical_df['date'] == target_date]
    if not row.empty:
        adj_close_price = row.iloc[0]['adjusted_close']
        return adj_close_price
    else:
        date=date-pd.Timedelta(days=1)
        return get_adj_close_price(target_date, historical_df)

#Function: to collect data for each Symbol in the universe... Mean to use this on each symbol in the universe
def collect_data(SYMBOL, target_date, fundamental_data, historical_data):
    # Get balance sheet data: 
    quarter_bs = fundamental_data[SYMBOL]['Financials']['Balance_Sheet']['quarterly']
    quarter_bs_df = pd.DataFrame(quarter_bs).T
    quarter_bs_df['filing_date'] = pd.to_datetime(quarter_bs_df['filing_date'])
    quarter_bs_df2 = quarter_bs_df[quarter_bs_df['filing_date'] < target_date][:5][['totalAssets', 'totalLiab', 'commonStockSharesOutstanding', 'netDebt']]

    # Get balance sheet data:  
    total_assets_value = float(quarter_bs_df2['totalAssets'].iloc[0])
    prior_total_assets = float(quarter_bs_df2['totalAssets'].iloc[4])

    total_liab_value = float(quarter_bs_df2['totalLiab'].iloc[0])

    shares_outstanding = float(quarter_bs_df2['commonStockSharesOutstanding'].iloc[0])

    net_debt = float(quarter_bs_df2['netDebt'].iloc[0])

    # Get income statement data:
    quarter_is_df = pd.DataFrame(fundamental_data[SYMBOL]['Financials']['Income_Statement']['quarterly']).T
    quarter_is_df['filing_date'] = pd.to_datetime(quarter_is_df['filing_date'])
    quarter_is_df2 = quarter_is_df[quarter_is_df['filing_date'] < target_date][:4][['totalRevenue', 'grossProfit', 'ebit', 'netIncome']]

    # Get last 4 quarters' revenues and Gross Profit
    ltm_revenue = quarter_is_df2['totalRevenue'].astype(float).sum()  
    ltm_gross_profit = quarter_is_df2['grossProfit'].astype(float).sum()  
    ltm_ebit = quarter_is_df2['ebit'].astype(float).sum()   
    ltm_ni = quarter_is_df2['netIncome'].astype(float).sum()   

    # Get last 4 quarters' operating cash flow
    quarter_cf_df = pd.DataFrame(fundamental_data[SYMBOL]['Financials']['Cash_Flow']['quarterly']).T
    quarter_cf_df['filing_date'] = pd.to_datetime(quarter_cf_df['filing_date'])
    quarter_cf_df2 = quarter_cf_df[quarter_cf_df['filing_date'] < target_date][:4][['totalCashFromOperatingActivities']]

    ltm_ocf = quarter_cf_df2['totalCashFromOperatingActivities'].astype(float).sum()

    
    # Get Price as at target_date    
    historical_df = pd.DataFrame(historical_data[SYMBOL])
    historical_df['date'] = pd.to_datetime(historical_df['date'])
    historical_df[historical_df['date'] == target_date]
    price=get_adj_close_price(target_date, historical_df)

    # Get valuation data: Price to Book, Forward PE, EV to Revenue    
    market_cap=price*shares_outstanding 

    book_value=total_assets_value - total_liab_value
    book_to_price= book_value/market_cap

    pe_ttm=market_cap/ltm_ni 

    sales_to_ev=ltm_revenue/(market_cap+net_debt)
    
    # Clean up: consolidate collected data into a single row dataFrame
    output=[SYMBOL, target_date, total_assets_value, prior_total_assets, ltm_revenue, ltm_gross_profit, ltm_ebit, ltm_ni, ltm_ocf, book_to_price, pe_ttm, sales_to_ev]
    
    return output   

columns = ['symbol', 'target_date', 'total_assets_value', 'prior_total_assets','ltm_revenue', 'ltm_gross_profit', 'ltm_ebit', 'ltm_ni', 'ltm_ocf','book_to_price', 'pe_ttm', 'sales_to_ev']
data_df=pd.DataFrame(columns=columns)

for symbol in universe:
    data = collect_data(symbol, target_date, fundamental_data, historical_data)
    data_df.loc[len(data_df)] = data

data_df['asset_growth']=data_df['total_assets_value'] / data_df['prior_total_assets'] - 1
data_df['accruals']=data_df['ltm_ebit'] - data_df['ltm_ocf'] 
data_df['gross_margin']=data_df['ltm_gross_profit'] / data_df['ltm_revenue']

# next define function for Z-scores
# for cheapness:
# for quality: accurals, gross margin, asset growth
# next apply z-scores, weighted z scores, total weighted quality and value scores

def z_score(x):
    return (x - np.nanmean(x)) / np.nanstd(x)

for i in ['book_to_price', 'pe_ttm', 'sales_to_ev']:
    data_df[f'z_{i}'] = z_score(data_df[i])

for i in ['accruals', 'gross_margin', 'asset_growth']:
    data_df[f'z_{i}'] = z_score(data_df[i])

data_df['z_quality'] = (-data_df['z_accruals'] + data_df['z_gross_margin'] - data_df['z_asset_growth']) / 3 # note negative signs where higher=> poorer
data_df['z_value'] = (data_df['z_book_to_price'] + data_df['z_pe_ttm'] + data_df['z_sales_to_ev']) / 3 # note higher => more value

wgt_value=0.5
wgt_quality=1-wgt_value

data_df['score']=wgt_value*data_df['z_value'] + wgt_quality*data_df['z_quality']