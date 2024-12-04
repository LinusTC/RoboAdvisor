from roboadvisor.optimizer import PortfolioOptimizer


if __name__=='__main__':

    assets=['AAPL','MSFT','FB', 'AMZN', 'NVDA', 'GOOG', 'TSLA', 'UA', 'NFLX', 'AMD', 'UAA', 'ADBE']
    optimal_portfolio=PortfolioOptimizer(assets, portfolio_size=6,max_weight=0.40, min_weight=0.05)