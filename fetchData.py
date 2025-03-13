
import time
import pandas as pd
import numpy as np
import yfinance as yf

def fetch_raw_data_yf(asset_basket, start_date="2015-01-01", end_date="2018-01-01"):
    start = time.time()
    asset_errors = []
    asset_data_frames = []

    unique_assets = list(set(asset_basket))

    for asset in unique_assets:
        data = yf.download(asset, start=start_date, end=end_date, auto_adjust=True)
        if not data.empty:
            close_prices = data['Close']
            close_prices.rename(columns={'Close': f"{asset}_Close"})
            asset_data_frames.append(close_prices)
        else:
            asset_errors.append(asset)

    df = pd.concat(asset_data_frames, axis=1, sort=True)
    df = df.dropna(axis=1)

    max_combination = df.shape[1]

    print('Omitted assets:', asset_errors)
    print('Time to fetch data: %.2f seconds' % (time.time() - start))
    print('Max combination of assets with complete data:', max_combination)

    return df, asset_errors, max_combination

def fetch_raw_data_yf_all(asset_basket, start_date = "2015-01-01", end_date="2018-01-01"):
    start = time.time()
    asset_errors = []

    unique_assets = list(set(asset_basket))

    data = yf.download(unique_assets, start=start_date, end=end_date)

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

def getSNP500():
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = table[0]
    stockdata = df['Symbol'].to_list()
    return stockdata

