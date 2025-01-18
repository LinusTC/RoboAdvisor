
import time
import quandl as q
from quandl.errors.quandl_error import NotFoundError 
from itertools import combinations
import pandas as pd
import numpy as np

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