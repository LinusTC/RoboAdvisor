from roboAdvisor import PortfolioOptimizer


if __name__=='__main__':

    assets=['AAPL','MSFT', 'AMZN', 'NVDA', 'GOOG', 'TSLA', 'UA', 'NFLX', 'AMD', 'UAA', 'ADBE', 'META', 'SBUX', 'CSCO', 'WMT']
    optimal_portfolio=PortfolioOptimizer(assets, portfolio_size=8,max_weight=0.40, min_weight=0.0)