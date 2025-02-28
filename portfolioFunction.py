import numpy as np
import scipy.optimize

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

def find_correlated_asset(rand_assets, selected_covariances):

    correlation_matrix = create_correlation_matrix(selected_covariances)

    avg_correlations = correlation_matrix.mean(axis=1)  

    most_correlated_asset = rand_assets[np.argmax(avg_correlations)]

    return most_correlated_asset, avg_correlations