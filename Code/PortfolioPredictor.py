import os
import random
import numpy as np
import pandas as pd
import psutil
import multiprocessing
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
import plotly.graph_objects as go

from LearningRBA import MLRBA_V2
from PortfolioFunction import get_matrices

class PortfolioPredictor:
    def __init__(self, raw_data_train, raw_data_test, best_portfolio, n_steps=5, epochs=50, batch_size=32, activation_function = 'relu'):
        self.raw_data_train = raw_data_train
        self.raw_data_test = raw_data_test
        self.best_portfolio = best_portfolio
        self.n_steps = n_steps
        self.epochs = epochs
        self.batch_size = batch_size
        self.activation_function = activation_function
        self.model = None
        self.history = None

    def preprocess_data(self):
        best_portfolio_data_train = self.raw_data_train[self.best_portfolio['tickers']]
        best_portfolio_data_test = self.raw_data_test[self.best_portfolio['tickers']]
        weights = np.array(self.best_portfolio['weights'])

        # Use a scaler fitted on a broader dataset so that training/test normalization is consistent
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        # Fit on the entire raw_data (or on a fixed training period) for consistency
        full_data = pd.concat([best_portfolio_data_train, best_portfolio_data_test])
        self.scaler.fit(full_data)
        
        normalized_train_data = self.scaler.transform(best_portfolio_data_train)
        normalized_test_data = self.scaler.transform(best_portfolio_data_test)

        normalized_test_data = np.concatenate([normalized_train_data[-self.n_steps:], normalized_test_data],axis=0)

        self.weighted_returns_train = np.dot(normalized_train_data, weights)
        self.weighted_returns_test = np.dot(normalized_test_data, weights)

    def create_datasets(self, data):
        X, y = [], []
        for i in range(len(data) - self.n_steps):
            v = data[i:(i + self.n_steps), :]
            X.append(v)
            y.append(data[i + self.n_steps, :])
        return np.array(X), np.array(y)

    def build_model(self):
        self.model = Sequential([
            LSTM(250, activation=self.activation_function, return_sequences=True),
            Dropout(0.2),
            LSTM(50, activation=self.activation_function, return_sequences=False),
            Dropout(0.2),
            Dense(1),
        ])

        def tf_weighted_mse(y_true, y_pred, power=3):
            n = tf.shape(y_true)[0]
            normalized_index = tf.cond(
                tf.equal(n, 1),
                lambda: tf.ones([n], dtype=tf.float32),
                lambda: tf.cast(tf.range(n), tf.float32) / tf.cast(n - 1, tf.float32)
            )
            weights = tf.pow(normalized_index, power)
            weights += 1e-6
            weights /= tf.reduce_sum(weights)
            
            squared_errors = tf.square(y_true - y_pred)
            weighted_squared_errors = weights * squared_errors
            return tf.reduce_mean(weighted_squared_errors)

        self.model.compile(optimizer='adam', loss=tf_weighted_mse)

    def train_model(self):
        self.X_train_weighted, self.y_train_weighted = self.create_datasets(self.weighted_returns_train.reshape(-1, 1))
        self.history = self.model.fit(self.X_train_weighted, self.y_train_weighted, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.001, shuffle=False, verbose=0)

    def predict(self):
        X_test_weighted, y_test_weighted = self.create_datasets(self.weighted_returns_test.reshape(-1, 1))
        
        self.predictions = self.model.predict(X_test_weighted)
        self.y_test_weighted = y_test_weighted
        
        prediction_dates = self.raw_data_test.index  # This index length equals the number of predictions
        
        return self.predictions, prediction_dates


    def compute_cumulative_returns(self, data, baseline):
        data_series = pd.Series(data.flatten())
        cumulative_returns = data_series / data_series.iloc[0] * baseline
        return cumulative_returns

    def plot_loss(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(1, len(self.history.history['loss'])+1), y=self.history.history['loss'], mode='lines', name='Training Loss'))
        fig.add_trace(go.Scatter(x=np.arange(1, len(self.history.history['val_loss'])+1), y=self.history.history['val_loss'], mode='lines', name='Validation Loss'))
        fig.update_layout(title='Training and Validation Loss Over Epochs',
                            xaxis_title='Epoch',
                            yaxis_title='Loss',
                            legend_title='Type of Loss',
                            font=dict(
                                family="Cambria",
                                size=18,
                            ))
        fig.show()
        
    def plot_predictions(self):
        normalized_train = self.compute_cumulative_returns(self.y_train_weighted, 100)
        training_end_value = normalized_train.iloc[-1]
        normalized_test = self.compute_cumulative_returns(self.y_test_weighted, training_end_value)
        normalized_predicted = self.compute_cumulative_returns(self.predictions, training_end_value)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.raw_data_train.index[self.n_steps:],
            y=normalized_train,
            mode='lines',
            name='Actual Training Returns'
        ))
        fig.add_trace(go.Scatter(
            x=self.raw_data_test.index,
            y=normalized_test,
            mode='lines',
            name='Actual Test Returns'
        ))
        fig.add_trace(go.Scatter(
            x=self.raw_data_test.index,
            y=normalized_predicted,
            mode='lines',
            name='Predicted Test Returns'
        ))
        fig.update_layout(
            title='Actual vs Predicted Weighted Portfolio Returns',
            xaxis_title='Date',
            yaxis_title='Cumulative Returns',
            legend_title='Portfolio',
            font=dict(
                family="Cambria",
                size=18,
            )
        )
        #fig.write_html("OneYearPrediction.html")
        #fig.write_image("OneYearPrediction.png", format='png', width=1920, height=1080)
        fig.show()

    def compute_performance(self):
        actual = self.y_test_weighted.flatten()
        predicted = self.predictions.flatten()

        percentage_diff = np.abs((predicted - actual) / actual) * 100

        mean_percentage_error = np.mean(percentage_diff)

        accuracy = 100 - mean_percentage_error

        print(f"Mean Percentage Error:{mean_percentage_error}%, Accuracy: {accuracy}")

        return mean_percentage_error, accuracy

def select_representative_portfolios(portfolios, portfolio_per_division, divisions):
    n = len(portfolios)
    if divisions > n:
        divisions = n
    
    selected = []
    group_size = n // divisions
    
    for i in range(divisions):
        start = i * group_size
        if i == divisions - 1:
            group = portfolios[start:]
        else:
            group = portfolios[start:start + group_size]
        
        if len(group) <= portfolio_per_division:
            selected.extend(group)
        else:
            selected.extend(random.sample(group, portfolio_per_division))
            
    return selected

def run_prediction_in_process(args):
    raw_data_train, raw_data_test, portfolio, window_size, epochs = args

    import tensorflow as tf  # Import inside subprocess
    tf.keras.backend.clear_session()
    predictor = PortfolioPredictor(raw_data_train, raw_data_test, portfolio, n_steps=window_size, epochs=epochs)
    predictor.preprocess_data()
    predictor.build_model()
    predictor.train_model()
    predictions, pred_dates = predictor.predict()

    tf.keras.backend.clear_session()
    import gc
    gc.collect()
    
    if len(predictions) >= window_size:
        end_pred = predictions[window_size-1]
    else:
        end_pred = predictions[-1]
    percentage_diff = (end_pred - predictions[0]) / predictions[0]
    
    return predictions, pred_dates, percentage_diff

def evaluate_portfolios_over_time(raw_data, investment_starting_date, window_size=5, threshold=0.05, epochs=30, length_of_investment = None, candidates_per_divison = 2, candidates_divison = 3):
    investment_start = len(raw_data[:investment_starting_date])

    trading_days = len(raw_data[investment_starting_date:investment_starting_date + pd.Timedelta(days=length_of_investment)] if length_of_investment is not None else raw_data[investment_starting_date:])
    trading_days = trading_days - 1 if trading_days % window_size == 0 else trading_days
    num_windows = trading_days // window_size

    previous_best_portfolio = None
    chosen_portfolios = []
    for i in range(num_windows):
        print(f'Predicting Week {i} in {num_windows} Total Weeks')

        #Keep the training data to just 1 year.
        loop_raw_data_train = raw_data.iloc[i*window_size : investment_start + i*window_size]
        loop_raw_data_test = raw_data.iloc[investment_start + i*window_size:]
        
        loop_names, loop_annualized_returns, _, _, _, loop_cov, loop_correlation_matrix = get_matrices(loop_raw_data_train)
        
        _, loop_best_portfolio, loop_good_portfolios, _, _ = MLRBA_V2(loop_names, loop_cov, loop_annualized_returns, loop_correlation_matrix)
        best_sharpe = loop_best_portfolio['sharpe']
        
        candidate_list = []
        for j in range(len(loop_good_portfolios)):
            difference = abs((best_sharpe - loop_good_portfolios[j]['sharpe']) / best_sharpe)
            if difference < threshold:
                candidate_list.append(loop_good_portfolios[j])

        filtered_candidate_list = select_representative_portfolios(candidate_list, candidates_per_divison, candidates_divison)

        if previous_best_portfolio is not None:
            filtered_candidate_list.append(previous_best_portfolio)

        sharpe_list = [portfolio['sharpe'] for portfolio in filtered_candidate_list]
        print(f'Length of close to best is: {len(filtered_candidate_list)}, Sharpe ratios:{sharpe_list}')

        
        portfolio_results = {}
        for id, portfolio in enumerate(filtered_candidate_list):
            print(f"RAM Usage: {psutil.Process(os.getpid()).memory_info().rss / 1024**2:.2f} MB")            
            args = (loop_raw_data_train, loop_raw_data_test, portfolio, window_size, epochs)
            with multiprocessing.get_context("spawn").Pool(1) as pool:
                result = pool.map(run_prediction_in_process, [args])
            predictions, pred_dates, percentage_diff = result[0]

            if len(predictions) >= window_size:
                end_pred = predictions[window_size-1]
            else:
                end_pred = predictions[-1]
            
            percentage_diff = (end_pred - predictions[0]) / predictions[0]

            num_to_print = min(window_size, len(predictions))
            for date, pred in zip(pred_dates[:num_to_print], predictions[:num_to_print]):
                print(f"  {date}: {pred}")
            print("First prediction:", predictions[0], "Percentage change:", percentage_diff * 100)
            
            portfolio_results[id] = percentage_diff 
            
        best_id = None

        # Check if all predictions (percentage_diff) are negative
        if max(portfolio_results.values()) < 0:
            print("All percentage differences are negative. Choosing an empty portfolio(not holding anything).")
            predicted_best_portfolio = {}
            previous_best_portfolio = None
        else:
            best_id = max(portfolio_results, key=portfolio_results.get)
            predicted_best_portfolio = filtered_candidate_list[best_id]
            previous_best_portfolio = predicted_best_portfolio
        
        portfolio_start_date = loop_raw_data_test.index[0]
        portfolio_end_date = loop_raw_data_test.index[window_size-1]
        
        chosen_portfolios.append({
            "portfolio": predicted_best_portfolio,
            "start_date": portfolio_start_date,
            "end_date": portfolio_end_date
        })
        if best_id is not None:
            print(f'Current iteration: {i}, the best portfolio found was portfolio: {best_id}')
        else:
            print(f'Current iteration: {i}, no portfolio selected (empty portfolio chosen).')
    
    return chosen_portfolios

def extract_asset_returns(raw_data, assets, start_date, end_date):
    if not isinstance(raw_data.index, pd.DatetimeIndex):
        raw_data.index = pd.to_datetime(raw_data.index)

    filtered_data = raw_data.loc[start_date:end_date, assets]

    return filtered_data

def chain_portfolio_performance(weekly_series_list, starting_value=100):
    continuous_series = pd.Series()
    current_value = starting_value

    for week_series in weekly_series_list:
        # Normalize the week so that it starts at 1 (or current_value)
        week_normalized = week_series / week_series.iloc[0]
        # Scale the normalized week to start at current_value
        week_scaled = week_normalized * current_value
        # Update the current_value to the last value of this week
        current_value = week_scaled.iloc[-1]
        # Append the week_series to the continuous_series
        continuous_series = pd.concat([continuous_series, week_scaled])
    
    return continuous_series