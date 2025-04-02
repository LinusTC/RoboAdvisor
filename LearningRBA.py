import math
import numpy as np
from PortfolioFunction import create_correlation_matrix, get_sharpe_ratio, maximize_sharpe
import pandas as pd
from tqdm import tqdm

def MLRBA_V1(ticker, covariances, returns, num_iterations=None, risk_free_rate = 0, 
             return_power = 1, std_power = 1, return_weight=1/3, corr_weight=1/3, vol_weight= 1/3, num_assets = 8, base_portfolio = None):
    
    if num_iterations is None:
        num_iterations = min(math.comb(len(ticker), num_assets), 100000)
    
    if base_portfolio is None:
        base_portfolio = np.random.choice(list(ticker), num_assets, replace=False)
        #base_portfolio = list(ticker)[:num_assets]
    
    def _get_portfolio_stats (portfolio_assets, risk_free_rate = 0):
        p_asset_ret = returns.loc[portfolio_assets].values
        p_asset_var = covariances.loc[portfolio_assets, portfolio_assets].values
        best_p_weights = maximize_sharpe(p_asset_ret, p_asset_var)
        p_ret = np.dot(best_p_weights,p_asset_ret)
        p_var = np.dot(best_p_weights, p_asset_var @ best_p_weights)
        sharpe = get_sharpe_ratio(p_ret, p_var, risk_free_rate, return_power, std_power)

        return p_asset_ret, p_asset_var, sharpe, p_ret, p_var, best_p_weights

    def _update_portfolios_array(portfolios, assets, weights, p_ret, p_var, iteration=None):
        portfolios.append({
            "tickers": assets,
            "weights": weights,
            "return": p_ret,
            "variance": p_var,
            "sharpe": (p_ret - risk_free_rate) / np.sqrt(p_var),
            "iteration": iteration
        })

    all_portfolios = []
    
    curr_ret, curr_var, curr_weighted_sharpe, curr_p_return, curr_p_variance, curr_p_weights = _get_portfolio_stats(base_portfolio, risk_free_rate)
    _update_portfolios_array(all_portfolios, base_portfolio, curr_p_weights, curr_p_return, curr_p_variance, iteration=0)

    good_portfolios = all_portfolios.copy()
    best_portfolio = base_portfolio.copy()

    highest_weighted_sharpe = -np.inf
    highest_weighted_sharpe = curr_weighted_sharpe
    
    portfolios_tested = 0
    best_iteration = 0

    progress_bar = tqdm(total=num_iterations, desc="Portfolios Tested")
    for _ in range(num_iterations):
        asset_to_remove = find_best_asset_to_remove(best_portfolio, curr_var, curr_ret)
        new_portfolio = [str(asset) for asset in best_portfolio if asset != asset_to_remove]

        ranked_assets = find_asset_to_add(new_portfolio, ticker, covariances, returns,
                                          return_weight, corr_weight, vol_weight)

        asset_added = False

        for asset in ranked_assets.index:
            if asset in new_portfolio:
                continue

            test_portfolio = new_portfolio + [asset]
            portfolios_tested += 1
            progress_bar.update(1)

            new_returns, new_var, new_weighted_sharpe, new_p_return, new_p_variance, new_p_weights = _get_portfolio_stats(test_portfolio, risk_free_rate)
            _update_portfolios_array(all_portfolios, test_portfolio, new_p_weights, new_p_return, new_p_variance, iteration=portfolios_tested+1)

            if new_weighted_sharpe > highest_weighted_sharpe:
                best_iteration = portfolios_tested
                best_portfolio = test_portfolio
                curr_ret, curr_var = new_returns, new_var
                highest_weighted_sharpe = new_weighted_sharpe

                _update_portfolios_array(good_portfolios, test_portfolio, new_p_weights, new_p_return, new_p_variance, iteration=portfolios_tested+1)

                asset_added = True
                break  # Accept first asset that improves Sharpe

        if not asset_added:
            print("All assets have been tested or no improvement found.")
            break

    progress_bar.close()

    base_details = good_portfolios[0]
    best_details = good_portfolios[-1]

    return base_details, best_details, good_portfolios, all_portfolios, best_iteration

def MLRBA_V2(ticker, covariances, returns, correlation_matrix, num_iterations=None, risk_free_rate = 0, 
             return_power = 1, std_power = 1, return_weight=1/3, corr_weight=1/3, vol_weight= 1/3, num_assets = 8, base_portfolio = None):
    
    if num_iterations is None:
        num_iterations = min(math.comb(len(ticker), num_assets), 100000)

    if base_portfolio is None:
        base_portfolio = np.random.choice(list(ticker), num_assets, replace=False)
        #base_portfolio = list(ticker)[:num_assets]

    def _get_portfolio_stats(portfolio_assets, risk_free_rate=0):
        p_asset_ret = returns.loc[portfolio_assets].values
        p_asset_var = covariances.loc[portfolio_assets, portfolio_assets].values
        best_p_weights = maximize_sharpe(p_asset_ret, p_asset_var)
        p_ret = np.dot(best_p_weights, p_asset_ret)
        p_var = np.dot(best_p_weights, p_asset_var @ best_p_weights)
        sharpe = get_sharpe_ratio(p_ret, p_var, risk_free_rate, return_power, std_power)
        return p_asset_ret, p_asset_var, sharpe, p_ret, p_var, best_p_weights

    def _update_portfolios_array(portfolios, assets, weights, p_ret, p_var, iteration=None):
        portfolios.append({
            "tickers": assets,
            "weights": weights,
            "return": p_ret,
            "variance": p_var,
            "sharpe": (p_ret - risk_free_rate) / np.sqrt(p_var),
            "iteration": iteration
        })

    all_portfolios = []

    curr_ret, curr_var, curr_weighted_sharpe, curr_p_return, curr_p_variance, curr_p_weights = _get_portfolio_stats(base_portfolio, risk_free_rate)
    _update_portfolios_array(all_portfolios, base_portfolio, curr_p_weights, curr_p_return, curr_p_variance, iteration=0)

    good_portfolios = all_portfolios.copy()
    best_portfolio = base_portfolio.copy()

    highest_weighted_sharpe = -np.inf
    highest_weighted_sharpe = curr_weighted_sharpe

    best_iteration = 0
    portfolios_tested = 0

    learning_rate = 0.03
    improvement_threshold = 0.001

    progress_bar = tqdm(total=num_iterations, desc="Portfolios Tested")
    for i in range(num_iterations):
        asset_to_remove = find_best_asset_to_remove(best_portfolio, curr_var, curr_ret)
        new_portfolio = [str(asset) for asset in best_portfolio if asset != asset_to_remove]

        ranked_assets = find_asset_to_add(new_portfolio, ticker, covariances, returns, return_weight, corr_weight, vol_weight)

        asset_added = False

        for asset in ranked_assets.index:
            portfolios_tested += 1
            progress_bar.update(1)
            
            copy_new_portfolio = new_portfolio.copy()
            copy_new_portfolio.append(asset)

            new_returns, new_var, new_weighted_sharpe, new_p_return, new_p_variance, new_p_weights = _get_portfolio_stats(copy_new_portfolio, risk_free_rate)
            _update_portfolios_array(all_portfolios, copy_new_portfolio, new_p_weights, new_p_return, new_p_variance, iteration=portfolios_tested+1)

            if new_weighted_sharpe > highest_weighted_sharpe:
                best_iteration = portfolios_tested
                improvement = new_weighted_sharpe - highest_weighted_sharpe
                highest_weighted_sharpe = new_weighted_sharpe
                best_portfolio = copy_new_portfolio
                curr_ret, curr_var = new_returns, new_var

                asset_added = True

                asset_return = returns.loc[asset]
                asset_vol = np.sqrt(covariances.loc[asset, asset])
                avg_return = returns.mean()
                avg_vol = np.sqrt(np.diag(covariances)).mean()

                corr_with_portfolio = correlation_matrix.loc[copy_new_portfolio, asset].drop(asset).mean()
                avg_corr_in_portfolio = correlation_matrix.loc[copy_new_portfolio].drop(asset, axis=1).mean().mean()

                # Update weights using the current learning rate
                return_weight += learning_rate * (asset_return - avg_return) / avg_return
                vol_weight    += learning_rate * (avg_vol - asset_vol) / avg_vol
                corr_weight   += learning_rate * (avg_corr_in_portfolio - corr_with_portfolio) / avg_corr_in_portfolio

                total = return_weight + corr_weight + vol_weight
                return_weight /= total
                corr_weight /= total
                vol_weight /= total

                if improvement < improvement_threshold:
                    learning_rate *= 0.95
                else:
                    learning_rate *= 1.01

                _update_portfolios_array(good_portfolios, copy_new_portfolio, new_p_weights, new_p_return, new_p_variance, iteration=portfolios_tested+1)
                break  # stop at first valid improving asset

        if not asset_added:
            print("All assets have been tested or no improvement possible.")
            break

    base_details = good_portfolios[0]
    best_details = good_portfolios[-1]

    return base_details, best_details, good_portfolios, all_portfolios, best_iteration

def find_best_asset_to_remove(rand_assets, selected_covariances, selected_returns, return_weight=1/3, corr_weight=1/3, vol_weight= 1/3):

    #Normalized Correlation
    corr_matrix = create_correlation_matrix(selected_covariances)
    avg_corrs = corr_matrix.mean(axis=1)  
    norm_corr = (avg_corrs - np.min(avg_corrs)) / (np.max(avg_corrs) - np.min(avg_corrs))

    # Normalized Returns
    norm_returns = 1 - (selected_returns - np.min(selected_returns)) / (np.max(selected_returns) - np.min(selected_returns))

    #Normalized Volatility
    std_devs = np.sqrt(np.diag(selected_covariances))
    norm_vols = (std_devs - np.min(std_devs)) / (np.max(std_devs) - np.min(std_devs))

    combined_score = corr_weight * norm_corr + return_weight * norm_returns + vol_weight * norm_vols

    asset_to_remove = rand_assets[np.argmax(combined_score)]

    return asset_to_remove

def find_asset_to_add(portfolio_assets, all_assets, all_covariance, all_returns, 
                      return_weight=1/3, corr_weight=1/3, vol_weight= 1/3):
    
    remaining_assets = [asset for asset in all_assets if asset not in portfolio_assets]
    all_covariance_df = pd.DataFrame(all_covariance, index=all_assets, columns=all_assets)

    corr_matrix = create_correlation_matrix(all_covariance_df)
    avg_corrs = corr_matrix.loc[remaining_assets, portfolio_assets].mean(axis=1)
    norm_corr = (avg_corrs - avg_corrs.min()) / (avg_corrs.max() - avg_corrs.min())

    norm_returns = (all_returns.loc[remaining_assets] - all_returns.min()) / (all_returns.max() - all_returns.min())

    std_devs = np.sqrt(np.diag(all_covariance_df.loc[remaining_assets, remaining_assets]))
    norm_vols = (std_devs - np.min(std_devs)) / (np.max(std_devs) - np.min(std_devs))

    combined_score = corr_weight * norm_corr + return_weight * norm_returns + vol_weight * norm_vols
    
    ranked_assets = combined_score.sort_values(ascending=False)
    
    return ranked_assets