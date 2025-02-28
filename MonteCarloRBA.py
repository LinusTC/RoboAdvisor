import numpy as np
import scipy.optimize
from tqdm import tqdm

def MonteCarloRBA(ticker, covariances, returns, num_iterations=10000, max_on="Sharpe"):
    all_portfolios = []
    dominant_portfolios = []

    for _ in tqdm(range(num_iterations)):
        num_assets = np.random.randint(3, 6)
        rand_assets = np.random.choice(list(ticker), num_assets, replace=False)

        # Extract data for current portflio
        selected_returns = returns.loc[rand_assets].values
        selected_covariances = covariances.loc[rand_assets, rand_assets].values

        # Optimize weights of each portfolio based on Sharpe ratio
        if max_on=="vol":
            asset_weights = minimize_volatility(selected_returns, selected_covariances)
        
        else:
            asset_weights = maximize_sharpe(selected_returns, selected_covariances)

        # Calculate portfolio return and variance
        curr_portfolio_returns = np.dot(asset_weights, selected_returns)
        curr_portfolio_var = np.dot(asset_weights, selected_covariances @ asset_weights)

        portfolio_data = {
            "return": curr_portfolio_returns,
            "variance": curr_portfolio_var,
            "tickers": rand_assets,
            "weights": asset_weights
        }
        all_portfolios.append(portfolio_data)

        # Check if is dominated
        is_dominated = any(portfolio["return"] >= curr_portfolio_returns and portfolio["variance"] <= curr_portfolio_var for portfolio in dominant_portfolios)

        if not is_dominated:
            dominant_portfolios.append(portfolio_data)

    return all_portfolios, dominant_portfolios

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