from BruteForceRBA import PortfolioOptimizer

if __name__=='__main__':

    assets=['AAPL','MSFT', 'AMZN', 'NVDA', 'GOOG', 'TSLA', 'UA', 'NFLX', 'AMD', 'UAA', 'ADBE', 'META', 'SBUX', 'CSCO', 'WMT'] #15 Assets only, 6435 combinations
    optimal_portfolio=PortfolioOptimizer(assets, portfolio_size=8,max_weight=1.0, min_weight=0.0)