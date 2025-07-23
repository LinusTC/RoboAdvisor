import numpy as np
from tqdm import tqdm
from PortfolioFunction import minimize_volatility, maximize_sharpe

def MonteCarloRBA(ticker, covariances, returns, num_iterations=10000, max_on="sharpe", min_assets = 3, max_assets = 5, min_weight=0, max_weight=1):
    all_portfolios = []
    dominant_portfolios = []
    frequency = {key: 1 for key in ticker}

    for i in tqdm(range(num_iterations)):
        num_assets = np.random.randint(min_assets, max_assets) if min_assets != max_assets else min_assets

        total_frequency = sum(1.0 / frequency[t] for t in ticker)
        probabilities = [(1.0 / frequency[t]) / total_frequency for t in ticker]
        rand_assets = np.random.choice(list(ticker), num_assets, replace=False, p=probabilities)

        for asset in rand_assets:
            frequency[asset] += 1

        selected_returns = returns.loc[rand_assets].values
        selected_covariances = covariances.loc[rand_assets, rand_assets].values

        # Optimize weights of each portfolio based on selected metric
        if max_on == "vol":
            asset_weights = minimize_volatility(selected_returns, selected_covariances)
        elif max_on == "random":
            weights = np.random.rand(num_assets)
            asset_weights = weights / sum(weights)
        else:
            asset_weights = maximize_sharpe(selected_returns, selected_covariances, 0, min_weight, max_weight)

        curr_portfolio_returns = np.dot(asset_weights, selected_returns)
        curr_portfolio_var = np.dot(asset_weights, selected_covariances @ asset_weights)

        portfolio_data = {
            "return": curr_portfolio_returns,
            "variance": curr_portfolio_var,
            "tickers": rand_assets,
            "weights": asset_weights,
            "sharpe": (curr_portfolio_returns-0)/np.sqrt(curr_portfolio_var),
            "iteration": i
        }
        all_portfolios.append(portfolio_data)

        # Check if is dominated
        is_dominated = any(portfolio["return"] >= curr_portfolio_returns and portfolio["variance"] <= curr_portfolio_var for portfolio in dominant_portfolios)
        if not is_dominated:
            dominant_portfolios.append(portfolio_data)

    return all_portfolios, dominant_portfolios
