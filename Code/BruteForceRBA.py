import matplotlib
import pandas as pd
import numpy as np
import time
import random
import scipy.optimize as optimize
from operator import itemgetter

from fetchData import fetch_raw_data_yf
from PortfolioFunction import get_matrices_bf
class PortfolioOptimizer:

    def __init__(self, assets, risk_tolerance=5.0, portfolio_size=5, max_iters=None, print_init=True, max_weight=1.0, min_weight=0.0):

        matplotlib.use('PS')
        self.max_weight_=max_weight
        self.min_weight_=min_weight
        self.print_init_=print_init
        self.asset_basket_=assets
        self.max_iters_=max_iters
        self.portfolio_size_=portfolio_size
        self.assets_=assets
        self.num_assets_=portfolio_size
        self.risk_tolerance_=risk_tolerance
        self.sim_iterations_=2500
        self.raw_asset_data, self.asset_errors_, self.max_combination_ = fetch_raw_data_yf(self.asset_basket_)
        self.sim_comb = get_matrices_bf(self.raw_asset_data, self.portfolio_size_, self.max_iters_)
        self.optimize_for_sharpe()
        self.optimize_for_volatility()
        
    def portfolio_stats(self,weights):   
        returns=self.return_matrix_
        cov_matrix=self.cov_matrix_
        
        weights=np.array(weights)
        port_return=np.sum(returns * weights)
        port_vol=np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe=port_return / port_vol
        self._sharpe_ = sharpe
        self._port_return_ = port_return
        self._port_vol_ = port_vol     
        
        stats=[port_return, port_vol, sharpe]        
        self.portfolio_stats_=np.array(stats)
        
        return np.array(stats)

    def optimize_for_sharpe(self):
        min_con = self.min_weight_
        max_con = self.max_weight_  
        num_assets = self.portfolio_size_     

        constraints = ({'type':'eq', 'fun': lambda x: np.sum(x) -1})
        bounds= tuple((min_con, max_con) for x in range(num_assets))
        initializer=num_assets * [1. / num_assets,]
        sim_comb=self.sim_comb.copy()
            
        def _maximize_sharpe(weights):     
            self.portfolio_stats(weights)
            sharpe=self._sharpe_         
            return -sharpe
           
        self.sharpe_scores_=[]

        for _ in range(len(sim_comb)):

            curr_sim=sim_comb.pop()
            self.return_matrix_=np.array(curr_sim[2])
            self.cov_matrix_=np.array(curr_sim[1])
            self.assets_=curr_sim[0]
            
            optimal_sharpe=optimize.minimize(
                _maximize_sharpe,
                initializer,
                method = 'SLSQP',
                bounds = bounds,
                constraints = constraints,
            )
            
            optimal_sharpe_weights_ = optimal_sharpe['x'].round(4)
            optimal_sharpe_stats_ = self.portfolio_stats(optimal_sharpe_weights_)
            
            x = self.assets_
            asset_list = []
            for i in range(len(x)):
                temp=x[i].split('_')
                asset_list.append(temp[0])            
            
            optimal_sharpe_portfolio_ = list(zip(asset_list, list(optimal_sharpe_weights_)))
            self.sharpe_scores_.append([optimal_sharpe_weights_,
                                        optimal_sharpe_portfolio_,
                                        round(optimal_sharpe_stats_[0] * 100, 4),
                                        round(optimal_sharpe_stats_[1] * 100, 4),
                                        round(optimal_sharpe_stats_[2], 4)])
        
        self.sharpe_scores_ = sorted(self.sharpe_scores_, key = itemgetter(4), reverse=True)
        self.best_sharpe_portfolio_ = self.sharpe_scores_[0]
        temp = self.best_sharpe_portfolio_
        
        print('----- Portfolio: Sharpe Ratio ----')
        print('')
        print(*temp[1], sep = '\n')
        print('')
        print('Portfolio Return: ', temp[2],'%')
        print('Portfolio Volatility: ', temp[3],'%')
        print('Portfolio Sharpe Ratio: ', temp[4])
        print('')
        print('')
    
    def optimize_for_volatility(self): 
        num_assets = self.portfolio_size_       
        constraints = ({'type':'eq', 'fun': lambda x: np.sum(x) -1})
        bounds = tuple((0,1) for x in range(num_assets))
        initializer = num_assets * [1. / num_assets,]
        sim_comb = self.sim_comb.copy()
        
        def _minimize_volatility(weights):           
            self.portfolio_stats(weights)
            port_vol = self._port_vol_     
            return port_vol
        
        self.vol_scores_ = []
        for _ in range(len(sim_comb)):
            curr_sim = sim_comb.pop()
            self.return_matrix_ = np.array(curr_sim[2])
            self.cov_matrix_ = np.array(curr_sim[1])
            self.assets_ = curr_sim[0]
            
            optimal_vol=optimize.minimize(
                _minimize_volatility,
                initializer,
                method = 'SLSQP',
                bounds = bounds,
                constraints = constraints,
            )
            
            optimal_vol_weights_ = optimal_vol['x'].round(4)
            optimal_vol_stats_ = self.portfolio_stats(optimal_vol_weights_)
            
            x = self.assets_
            asset_list = []
            for i in range(len(x)):
                temp = x[i].split('_')
                asset_list.append(temp[0])
                
            optimal_vol_portfolio_ = list(zip(asset_list, list(optimal_vol_weights_)))
            self.vol_scores_.append([optimal_vol_weights_,
                                     optimal_vol_portfolio_,
                                     round(optimal_vol_stats_[0] * 100, 4),
                                     round(optimal_vol_stats_[1] * 100, 4),
                                     round(optimal_vol_stats_[2], 4)])
        
        self.vol_scores_ = sorted(self.vol_scores_, key = itemgetter(3))
        self.best_vol_portfolio_ = self.vol_scores_[0]
        temp = self.best_vol_portfolio_

        if (self.print_init_ == True):      
            print('----- Portfolio: Minimal Volatility ----')
            print('')
            print(*temp[1], sep='\n')
            print('')
            print('Portfolio Return: ', temp[2],'%')
            print('Portfolio Volatility: ', temp[3],'%')
            print('Portfolio Sharpe Ratio: ', temp[4])
            print('')
            print('')  