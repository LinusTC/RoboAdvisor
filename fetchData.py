
import time
import quandl as q
from quandl.errors.quandl_error import NotFoundError 
from itertools import combinations
import pandas as pd
import numpy as np
from tqdm import tqdm
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

def get_matrices(df, portfolio_size, max_iters=None):
    features = [f for f in df.columns if "_Close" in f]
    combo_generator = combinations(features, portfolio_size)
    
    sim_comb = []
    count = 0
    
    for assets in tqdm(combo_generator):
        if max_iters is not None and count >= max_iters:
            break

        filtered_df = df[list(assets)].copy()
        returns = np.log(filtered_df / filtered_df.shift(1))
        return_matrix = returns.mean() * 252  # Annualize by number of trading days
        cov_matrix = returns.cov() * 252      # Annualize covariance matrix
        
        sim_comb.append([assets, cov_matrix, return_matrix])
        count += 1

    return sim_comb

def getNasdaqStocks(num_assets=100):
    # Fetch the list of all NASDAQ tickers
    url = 'http://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt'
    nasdaq_listed = pd.read_csv(url, sep='|')
    nasdaq_tickers = nasdaq_listed['Symbol'].dropna().tolist()

    return np.random.choice(nasdaq_tickers, num_assets, replace=False)

def fetch_data(asset_basket, auth_token, portfolio_size, max_iters):
    start = time.time()
    asset_errors = []
    df = pd.DataFrame()

    for asset in asset_basket:
        try:
            temp = q.get_table('WIKI/PRICES', ticker=[asset],
                            qopts={'columns': ['date', 'ticker', 'adj_close']},
                            date={'gte': '2015-01-01', 'lte': '2018-03-26'},
                            paginate=True,
                            api_key=auth_token)
            temp = temp.rename(columns={"adj_close": asset + "_adj_close"})
            temp.drop('ticker', axis=1, inplace=True)

            if df.empty:
                df = temp
            else:
                df = pd.merge(df, temp, how='outer', on='date')

        except NotFoundError:
            asset_errors.append(asset)

    if df.empty:
        raise ValueError("No data fetched for any assets.")

    df = df.dropna()
    features = [f for f in list(df) if "adj_close" in f]
    raw_asset_data = df.copy()
    asset_combos = list(combinations(features, portfolio_size))

    max_iters = min(len(asset_combos), max_iters if max_iters is not None else len(asset_combos))
    
    sim_comb = []
    for assets in asset_combos[:max_iters]:
        filtered_df = df[list(assets)].copy()
        returns = np.log(filtered_df / filtered_df.shift(1))
        return_matrix = returns.mean() * 252    #Trading days in a year
        cov_matrix = returns.cov() * 252    #Trading days in a year

        sim_comb.append([assets, cov_matrix, return_matrix])

    print('Omitted assets: ', asset_errors)
    print('Time to fetch data: %.2f seconds' % (time.time() - start))
    
    return raw_asset_data, sim_comb, asset_errors
