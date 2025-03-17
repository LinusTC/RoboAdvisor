import numpy as np
import scipy.optimize
from itertools import combinations
from tqdm import tqdm

def get_matrices_bf(df, portfolio_size, max_iters=None):
    features = [f for f in df.columns]
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

        correlation_matrix = create_correlation_matrix(cov_matrix)

        sim_comb.append([assets, cov_matrix, return_matrix, correlation_matrix])
        count += 1

    return sim_comb

def get_matrices(df, halflife=30):
    df = df.dropna()

    returns = np.log(df / df.shift(1)).dropna()
    ewm_returns = returns.ewm(halflife=halflife).mean()

    return_matrix = ewm_returns.mean() * 252

    cov_matrix = returns.cov() * 252

    correlation_matrix = create_correlation_matrix(cov_matrix)
    
    return df.columns.tolist(), return_matrix, cov_matrix, correlation_matrix


def maximize_sharpe(returns, covariances, risk_free_rate=0, min_weight = 0, max_weight = 1, return_power = 1, std_power = 1):
    num_assets = len(returns)
    
    def neg_sharpe(weights):
        portfolio_return = np.dot(weights, returns)
        portfolio_variance = np.dot(weights, covariances @ weights)
        sharpe_ratio = get_sharpe_ratio(portfolio_return, portfolio_variance, risk_free_rate, return_power, std_power)
        return -sharpe_ratio  

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds= tuple((min_weight, max_weight) for _ in range(num_assets))
    initializer = num_assets * [1. / num_assets,]

    optimized = scipy.optimize.minimize(neg_sharpe, initializer, method='SLSQP', bounds=bounds, constraints=constraints)

    return optimized.x

def minimize_volatility(returns, covariances):     
    num_assets = len(returns)

    def minimize_volatility(weights):      
        return np.dot(weights, np.dot(covariances, weights))

    constraints = ({'type':'eq', 'fun': lambda x: np.sum(x) -1})
    bounds = tuple((0,1) for x in range(num_assets))
    initializer = num_assets * [1. / num_assets,]

    optimized = scipy.optimize.minimize(minimize_volatility, initializer, method='SLSQP', bounds=bounds, constraints=constraints)

    return optimized.x

def create_correlation_matrix(cov_matrix):
    std_devs = np.sqrt(np.diag(cov_matrix))
    correlation_matrix = cov_matrix / (std_devs[:, None] * std_devs[None, :])

    return correlation_matrix

def get_sharpe_ratio(returns, variances, risk_free_rate = 0, return_power = 1, std_power = 1):

    std_devs = np.sqrt(variances)
    sharpe_ratios = ((returns - risk_free_rate) ** return_power) / (std_devs ** std_power)

    return sharpe_ratios