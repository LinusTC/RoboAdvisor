import numpy as np
from portfolioFunction import create_correlation_matrix, maximize_sharpe
import pandas as pd
from tqdm import tqdm

def MLRBA(ticker, covariances, returns, num_iterations=10000):

    num_assets = np.random.randint(3, 6)
    rand_assets = np.random.choice(list(ticker), num_assets, replace=False)

    selected_returns = returns.loc[rand_assets].values
    selected_covariances = covariances.loc[rand_assets, rand_assets].values

    asset_weights = maximize_sharpe(selected_returns, selected_covariances)

    curr_portfolio_returns = np.dot(asset_weights, selected_returns)
    curr_portfolio_var = np.dot(asset_weights, selected_covariances @ asset_weights)        

    return selected_returns, selected_covariances, rand_assets

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