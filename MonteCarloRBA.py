import numpy as np
from tqdm import tqdm
from portfolioFunction import minimize_volatility, maximize_sharpe

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