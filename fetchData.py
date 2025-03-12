
import time
import pandas as pd
import numpy as np
import yfinance as yf

def fetch_raw_data_yf(asset_basket, start_date = "2015-01-01", end_date="2018-01-01"):
    start = time.time()
    asset_errors = []

    unique_assets = list(set(asset_basket))

    try:
        data = yf.download(unique_assets, start=start_date, end=end_date)
    except Exception as e:
        raise ValueError(f"Error fetching data for assets: {e}")

    df = pd.DataFrame()
    for asset in unique_assets:
        if asset in data['Close'].columns:
            temp = data['Close'][[asset]].dropna()
            if not temp.empty:
                temp.rename(columns={asset: f"{asset}_Close"}, inplace=True)
                df = pd.merge(df, temp, left_index=True, right_index=True, how='outer') if not df.empty else temp
            else:
                asset_errors.append(asset)
        else:
            asset_errors.append(asset)

    if df.empty:
        raise ValueError("No data fetched for any assets.")

    df = df.dropna()
    max_combination = len(unique_assets) - len(asset_errors)

    print('Omitted assets (', len(asset_errors), '): ', asset_errors)
    print('Time to fetch data: %.2f seconds' % (time.time() - start))
    
    return df, asset_errors, max_combination

def getNasdaqStocks(num_assets=100):
    # Fetch the list of all NASDAQ tickers
    url = 'http://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt'
    nasdaq_listed = pd.read_csv(url, sep='|')
    nasdaq_tickers = nasdaq_listed['Symbol'].dropna().tolist()

    return np.random.choice(nasdaq_tickers, num_assets, replace=False)
