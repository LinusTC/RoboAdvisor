import numpy as np
import scipy.optimize
from tqdm import tqdm

def MLRBA(ticker, covariances, returns, num_iterations=10000):

    num_assets = np.random.randint(3, 6)
    rand_assets = np.random.choice(list(ticker), num_assets, replace=False)

    selected_returns = returns.loc[rand_assets].values
    selected_covariances = covariances.loc[rand_assets, rand_assets].values

    asset_weights = maximize_sharpe(selected_returns, selected_covariances)

    curr_portfolio_returns = np.dot(asset_weights, selected_returns)
    curr_portfolio_var = np.dot(asset_weights, selected_covariances @ asset_weights)        

    return selected_returns, selected_covariances, rand_assets

def maximize_sharpe(returns, covariances, risk_free_rate=0, min_weight = 0, max_weight = 1):
    num_assets = len(returns)
    
    def neg_sharpe(weights):
        portfolio_return = np.dot(weights, returns)
        portfolio_variance = np.dot(weights, covariances @ weights)
        sharpe_ratio = (portfolio_return - risk_free_rate) / np.sqrt(portfolio_variance)
        return -sharpe_ratio  

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds= tuple((min_weight, max_weight) for x in range(num_assets))
    initializer = num_assets * [1. / num_assets,]

    optimized = scipy.optimize.minimize(neg_sharpe, initializer, method='SLSQP', bounds=bounds, constraints=constraints)

    return optimized.x

def find_correlation_matrix(rand_assets, selected_covariances):
    std_devs = np.sqrt(np.diag(selected_covariances))
    correlation_matrix = selected_covariances / (std_devs[:, None] * std_devs[None, :])

    avg_correlations = correlation_matrix.mean(axis=1)  

    most_correlated_asset = rand_assets[np.argmax(avg_correlations)]

    return most_correlated_asset, avg_correlations, correlation_matrix

def find_best_asset_to_remove(rand_assets, selected_covariances, selected_returns, return_weight=0.5, correlation_weight=0.5):
    std_devs = np.sqrt(np.diag(selected_covariances))
    correlation_matrix = selected_covariances / (std_devs[:, None] * std_devs[None, :])

    avg_correlations = correlation_matrix.mean(axis=1)  

    norm_correlation = (avg_correlations - np.min(avg_correlations)) / (np.max(avg_correlations) - np.min(avg_correlations))
    norm_returns = 1 - (selected_returns - np.min(selected_returns)) / (np.max(selected_returns) - np.min(selected_returns))

    combined_score = correlation_weight * norm_correlation + return_weight * norm_returns  

    asset_to_remove = rand_assets[np.argmax(combined_score)]

    return asset_to_remove, avg_correlations, correlation_matrix