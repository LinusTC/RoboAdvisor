import numpy as np
from tqdm import tqdm

def MonteCarloRBA(ticker, covariances, returns, num_iterations=10000):
    all_portfolios = []
    dominant_portfolios = []

    for _ in tqdm(range(num_iterations)):
        # Generate a random portfolio
        num_assets = np.random.randint(3, 6)
        selected_assets = np.random.choice(list(ticker), num_assets, replace=False)
        asset_weights = np.random.dirichlet(np.ones(num_assets))

        # Calculate portfolio return and variance
        curr_portfolio_returns = np.dot(asset_weights, returns.loc[selected_assets])
        curr_portfolio_var = np.dot(asset_weights, covariances.loc[selected_assets, selected_assets] @ asset_weights)

        # Store portfolio information
        portfolio_data = {
            "return": curr_portfolio_returns,
            "variance": curr_portfolio_var,
            "tickers": selected_assets,
            "weights": asset_weights
        }
        all_portfolios.append(portfolio_data)

        is_dominated = any(portfolio["return"] >= curr_portfolio_returns and portfolio["variance"] <= curr_portfolio_var for portfolio in dominant_portfolios)

        # Only add non-dominated portfolios
        if not is_dominated:
            dominant_portfolios.append(portfolio_data)

    return all_portfolios, dominant_portfolios