import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input
from stable_baselines3 import PPO
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import talib
import time
import logging
import pickle
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
from gym import spaces
import joblib
from collections import deque
import os
from hmmlearn import hmm
import requests
from newsapi import NewsApiClient
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from datetime import datetime
from threading import Thread, Lock
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Constants
VOLATILITY_THRESHOLD = 0.02
PROFIT_TAKING_THRESHOLD = 0.1
GLOBAL_STOP_LOSS_PERCENTAGE = 0.2
MAX_LEVERAGE = 10
RETRAINING_COOLDOWN = 3600
NEWS_UPDATE_INTERVAL = 3600
STRATEGY_OPTIMIZATION_INTERVAL = 24 * 3600 

class DynamicCurrencySelector:
    def __init__(self, exchange, max_pairs=5):
        self.exchange = exchange
        self.max_pairs = max_pairs

    def select_pairs(self):
        markets = self.exchange.load_markets()
        usdt_pairs = [symbol for symbol, market in markets.items() if market['quote'] == 'USDT']
        
        pair_data = []
        for pair in usdt_pairs:
            try:
                ohlcv = self.exchange.fetch_ohlcv(pair, '1d', limit=30)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                volatility = df['close'].pct_change().std()
                volume = df['volume'].mean()
                liquidity = df['close'].iloc[-1] * df['volume'].iloc[-1]
                trend = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
                
                pair_data.append({
                    'pair': pair,
                    'volatility': volatility,
                    'volume': volume,
                    'liquidity': liquidity,
                    'trend': trend
                })
            except Exception as e:
                print(f"Error processing {pair}: {e}")
        
        df_pairs = pd.DataFrame(pair_data)
        df_pairs = self.apply_filters(df_pairs)
        
        for column in ['volatility', 'volume', 'liquidity', 'trend']:
            df_pairs[column] = (df_pairs[column] - df_pairs[column].min()) / (df_pairs[column].max() - df_pairs[column].min())
        
        df_pairs['score'] = (
            df_pairs['volatility'] * 0.3 +
            df_pairs['volume'] * 0.3 +
            df_pairs['liquidity'] * 0.2 +
            df_pairs['trend'] * 0.2
        )
        
        top_pairs = df_pairs.nlargest(self.max_pairs, 'score')['pair'].tolist()
        return top_pairs

    def apply_filters(self, df):
        min_volume = df['volume'].quantile(0.25)
        df = df[df['volume'] > min_volume]

        min_liquidity = df['liquidity'].quantile(0.25)
        df = df[df['liquidity'] > min_liquidity]

        vol_lower, vol_upper = df['volatility'].quantile([0.25, 0.75])
        df = df[(df['volatility'] > vol_lower) & (df['volatility'] < vol_upper)]

        return df

class ImprovedMarketRegimeDetector:
    def __init__(self, n_regimes=7, lstm_lookback=60):
        self.n_regimes = n_regimes
        self.lstm_lookback = lstm_lookback
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_regimes, random_state=42)
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.lstm_model = self._build_lstm_model()

    def _build_lstm_model(self):
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.lstm_lookback, 5)),
            LSTM(50, return_sequences=False),
            Dense(self.n_regimes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        return model

    def fit(self, df):
        tech_features = self.get_technical_features(df)
        scaled_features = self.scaler.fit_transform(tech_features)
        
        self.kmeans.fit(scaled_features)
        
        X_lstm, y_lstm = self._prepare_lstm_data(scaled_features)
        self.lstm_model.fit(X_lstm, y_lstm, epochs=100, batch_size=32, verbose=0)

    def detect_regime(self, df, news):
        tech_features = self.get_technical_features(df)
        scaled_features = self.scaler.transform(tech_features)
        
        kmeans_regime = self.kmeans.predict(scaled_features)[-1]
        lstm_regime = self._predict_lstm(scaled_features)
        sentiment_score = self.get_sentiment_score(news)
        
        return self.ensemble_regime(kmeans_regime, lstm_regime, sentiment_score)

    def get_technical_features(self, df):
        return df[['close', 'volume', 'rsi', 'macd', 'macdsignal']].values

    def get_sentiment_score(self, news):
        scores = [self.sentiment_analyzer.polarity_scores(article['title'])['compound'] for article in news]
        return np.mean(scores)

    def _prepare_lstm_data(self, scaled_features):
        X, y = [], []
        for i in range(len(scaled_features) - self.lstm_lookback):
            X.append(scaled_features[i:(i + self.lstm_lookback)])
            y.append(scaled_features[i + self.lstm_lookback])
        return np.array(X), np.array(y)

    def _predict_lstm(self, scaled_features):
        X = scaled_features[-self.lstm_lookback:].reshape(1, self.lstm_lookback, 5)
        prediction = self.lstm_model.predict(X)
        return np.argmax(prediction)

    def ensemble_regime(self, kmeans_regime, lstm_regime, sentiment_score):
        regime_map = {
            0: "Strong Bull",
            1: "Bull",
            2: "Weak Bull",
            3: "Neutral",
            4: "Weak Bear",
            5: "Bear",
            6: "Strong Bear"
        }
        
        ensemble_score = (kmeans_regime + lstm_regime) / 2
        if sentiment_score > 0.5:
            ensemble_score = max(0, ensemble_score - 1)
        elif sentiment_score < -0.5:
            ensemble_score = min(6, ensemble_score + 1)
        
        return regime_map[round(ensemble_score)]

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        
        joblib.dump(self.scaler, os.path.join(path, 'scaler.joblib'))
        joblib.dump(self.kmeans, os.path.join(path, 'kmeans.joblib'))
        self.lstm_model.save(os.path.join(path, 'lstm_model.h5'))

    def load(self, path):
        self.scaler = joblib.load(os.path.join(path, 'scaler.joblib'))
        self.kmeans = joblib.load(os.path.join(path, 'kmeans.joblib'))
        self.lstm_model = load_model(os.path.join(path, 'lstm_model.h5'))

class GoogleSpreadsheetTracker:
    def __init__(self, credentials_file, spreadsheet_name):
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_file, scope)
        self.client = gspread.authorize(creds)
        self.sheet = self.client.open(spreadsheet_name).worksheet("Performance")
        self.initialize_sheet()

    def initialize_sheet(self):
        headers = ["Timestamp", "Total Balance", "Profit/Loss", "Active Trades", "Market Regime"]
        if not self.sheet.get('A1:E1'):
            self.sheet.update('A1:E1', [headers])

    def update_performance(self, bot):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_balance = bot.client.fetch_balance()['total']['USDT']
        profit_loss = bot.total_profit
        active_trades = len(bot.client.fetch_open_orders())
        market_regime = bot.detect_market_regime(bot.currency_pairs[0])  # Using the first pair as a proxy

        row_data = [current_time, total_balance, profit_loss, active_trades, market_regime]
        self.sheet.append_row(row_data)
        bot.logger.info(f"Performance data updated in Google Sheet at {current_time}")

class BinanceTradingBot:
    def __init__(self, api_key, api_secret, initial_capital, email_alerts=False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.initial_capital = initial_capital
        self.email_alerts = email_alerts
        self.client = ccxt.binance({'apiKey': self.api_key, 'secret': self.api_secret})
        self.setup_logger()
        
        self.currency_selector = DynamicCurrencySelector(self.client, max_pairs=5)
        self.update_currency_pairs()
        
        self.spreadsheet_tracker = GoogleSpreadsheetTracker('path_to_credentials.json', "BotPerformance")
        
        self.total_profit = 0
        self.transaction_fee = 0.001  # 0.1% fee
        self.slippage = 0.001  # 0.1% slippage

        self.strategy_manager = StrategyManager(self)
        self.capital_manager = CapitalManager(self)

        self.marl_env = MultiAgentTradingEnv(self, self.currency_pairs)

        self.models = {pair: {} for pair in self.currency_pairs}
        self.regime_detector = ImprovedMarketRegimeDetector()

        self.stop_learning = False
        self.learning_thread = Thread(target=self.real_time_learning)
        self.learning_thread.start()

        self.backtesting_data = {}
        self.max_drawdown = GLOBAL_STOP_LOSS_PERCENTAGE

        self.newsapi = NewsApiClient(api_key="your_newsapi_key")
        self.news_update_interval = NEWS_UPDATE_INTERVAL
        self.last_news_update = 0

        self.strategy_optimization_interval = STRATEGY_OPTIMIZATION_INTERVAL
        self.last_strategy_optimization = 0

        self.model_lock = Lock()
        self.performance_lock = Lock()
        self.strategy_lock = Lock()
        self.balance_lock = Lock()

        self.model_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(self.model_dir, exist_ok=True)

        self.stop_loss_orders = {}
        self.trailing_stop_loss_orders = {}

        self.max_leverage = MAX_LEVERAGE
        self.current_leverage = {pair: 1 for pair in self.currency_pairs}

        self.global_stop_loss_percentage = GLOBAL_STOP_LOSS_PERCENTAGE

        self.strategy_weights = {strategy: 1 for strategy in self.strategy_manager.strategies}

        self.retraining_cooldown = RETRAINING_COOLDOWN
        self.last_retraining = {pair: 0 for pair in self.currency_pairs}

        self.profit_taking_threshold = PROFIT_TAKING_THRESHOLD

        self.stop_loss_thread = Thread(target=self.check_stop_losses_continuously)
        self.stop_loss_thread.start()

        self.portfolio_values = [initial_capital]

        self.load_models()
        self.respect_api_limits()

    def setup_logger(self):
        logging.basicConfig(filename='bot_log.log', level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger()

    def update_currency_pairs(self):
        self.currency_pairs = self.currency_selector.select_pairs()
        self.logger.info(f"Updated currency pairs: {self.currency_pairs}")

    def fetch_market_data(self, pair, interval='1h', limit=100):
        ohlcv = self.client.fetch_ohlcv(pair, timeframe=interval, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        df['ma_50'] = df['close'].rolling(window=50).mean()
        df['ma_200'] = df['close'].rolling(window=200).mean()
        df['volatility'] = df['close'].rolling(window=20).std()
        df['rsi'] = talib.RSI(df['close'])
        df['macd'], df['macdsignal'], _ = talib.MACD(df['close'])
        
        return df

    def detect_market_regime(self, pair):
        df = self.fetch_market_data(pair)
        news = self.fetch_news(pair)
        
        if not hasattr(self.regime_detector.scaler, 'n_features_in_'):
            self.regime_detector.fit(df)
            self.save_models()
        
        return self.regime_detector.detect_regime(df, news)

    def get_recent_performance(self, pair):
        df = self.fetch_market_data(pair)
        return (df['close'].iloc[-1] / df['close'].iloc[-7] - 1) * 100

    def fetch_news(self, pair):
        base_currency = pair.split('/')[0]
        try:
            news = self.newsapi.get_everything(q=base_currency, language='en', sort_by='publishedAt', page_size=10)
            return news['articles']
        except Exception as e:
            self.logger.error(f"Error fetching news for {pair}: {e}")
            return []

    def set_dynamic_leverage(self, pair, volatility):
        base_volatility = 0.02
        leverage = int(min(self.max_leverage, max(1, self.max_leverage * (base_volatility / volatility))))
        
        try:
            self.client.fapiPrivate_post_leverage({
                'symbol': pair.replace('/', ''),
                'leverage': leverage
            })
            self.current_leverage[pair] = leverage
            self.logger.info(f"Leverage for {pair} set to {leverage}x")
        except Exception as e:
            self.logger.error(f"Error setting leverage for {pair}: {e}")

    def calculate_atr(self, pair, period=14):
        df = self.fetch_market_data(pair)
        atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=period)
        return atr.iloc[-1]

    def predict_with_ensemble(self, pair):
        df = self.fetch_market_data(pair)
        X = self.prepare_data_for_ensemble(df)[-1].reshape(1, -1)
        ensemble = self.models[pair]['ensemble']
        return ensemble.predict(X)[0]

    def prepare_data_for_ensemble(self, df):
        X_rf = df[['ma_50', 'ma_200', 'volatility', 'rsi', 'macd', 'macdsignal']].values
        X_lstm = self.prepare_lstm_data(df[['close', 'volume', 'rsi', 'macd', 'macdsignal']].values)
        X = np.hstack((X_rf, X_lstm.reshape(X_lstm.shape[0], -1)))
        y = np.where(df['close'].shift(-1) > df['close'], 1, 0)[:-1]
        return X, y

    def prepare_lstm_data(self, data, lookback=60):
        X = []
        for i in range(len(data) - lookback):
            X.append(data[i:i+lookback])
        return np.array(X)

    def train_random_forest_model(self, pair):
        df = self.fetch_market_data(pair)
        df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
        features = ['ma_50', 'ma_200', 'volatility', 'rsi', 'macd', 'macdsignal']
        X = df[features]
        y = df['target']
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model

    def train_lstm_model(self, pair):
        df = self.fetch_market_data(pair)
        df_scaled = MinMaxScaler(feature_range=(0, 1)).fit_transform(df[['close', 'volume', 'rsi', 'macd', 'macdsignal']])
        
        X, y = [], []
        for i in range(60, len(df_scaled)):
            X.append(df_scaled[i-60:i])
            y.append(df_scaled[i, 0])
        X, y = np.array(X), np.array(y)

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(60, 5)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, batch_size=32, epochs=10)
        return model

    def create_ensemble_model(self, pair):
        rf = self.train_random_forest_model(pair)
        lstm = self.train_lstm_model(pair)
        
        ensemble = VotingClassifier(
            estimators=[('rf', rf), ('lstm', lstm)],
            voting='soft'
        )
        
        df = self.fetch_market_data(pair)
        X, y = self.prepare_data_for_ensemble(df)
        ensemble.fit(X, y)
        
        return ensemble

    def train_and_update_models(self, pair):
        with self.model_lock:
            new_rf = self.train_random_forest_model(pair)
            new_lstm = self.train_lstm_model(pair)
            new_ensemble = self.create_ensemble_model(pair)
            
            self.models[pair] = {
                'rf': new_rf,
                'lstm': new_lstm,
                'ensemble': new_ensemble
            }
            
            self.logger.info(f"Models updated for {pair}")

    def real_time_learning(self):
        while not self.stop_learning:
            for pair in self.currency_pairs:
                df = self.fetch_market_data(pair)
                volatility = df['volatility'].iloc[-1]
                
                if volatility > VOLATILITY_THRESHOLD:
                    self.train_and_update_models(pair)
                    self.logger.info(f"Models retrained due to volatility spike for {pair}")
            
            df = self.fetch_market_data(self.currency_pairs[0])
            self.regime_detector.fit(df)
            self.save_models()
            self.logger.info("Regime detector updated and saved")
            
            time.sleep(3600)

    def respect_api_limits(self):
        self.client.enableRateLimit = True
        self.logger.info("API rate limiting enabled")

    def load_models(self):
        for pair in self.currency_pairs:
            model_path = os.path.join(self.model_dir, f"{pair}_models.joblib")
            try:
                if os.path.exists(model_path):
                    self.models[pair] = joblib.load(model_path)
                else:
                    self.logger.warning(f"No saved model found for {pair}. Initializing new models.")
                    self.models[pair] = self.create_new_models(pair)
            except Exception as e:
                self.logger.error(f"Error loading model for {pair}: {e}")
                self.models[pair] = self.create_new_models(pair)

        regime_detector_path = os.path.join(self.model_dir, 'regime_detector')
        if os.path.exists(regime_detector_path):
            self.regime_detector.load(regime_detector_path)
        else:
            self.logger.warning("No saved regime detector found. Will train a new one.")

        self.logger.info("Models loaded successfully")

    def save_models(self):
        for pair, models in self.models.items():
            joblib.dump(models, os.path.join(self.model_dir, f"{pair}_models.joblib"))
        
        self.regime_detector.save(os.path.join(self.model_dir, 'regime_detector'))
        self.logger.info("Models saved successfully")

    def create_new_models(self, pair):
        return {
            'rf': self.train_random_forest_model(pair),
            'lstm': self.train_lstm_model(pair),
            'ensemble': self.create_ensemble_model(pair)
        }

class BinanceTradingBot:
    # ... (previous methods remain the same)

    def handle_error(self, error):
        self.logger.error(f"An error occurred: {error}")
        if self.email_alerts:
            self.send_email_alert(f"Trading Bot Error: {error}")

    def send_email_alert(self, message):
        sender_email = "your_email@example.com"
        receiver_email = "alert_recipient@example.com"
        password = "your_email_password"

        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = "Trading Bot Alert"

        msg.attach(MIMEText(message, 'plain'))

        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender_email, password)
            text = msg.as_string()
            server.sendmail(sender_email, receiver_email, text)
            server.quit()
            self.logger.info(f"Email alert sent: {message}")
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")

    def execute_trade(self, pair, side, amount):
        if side == 'buy':
            return self.execute_buy_order(pair, 'neutral', 0, 0, amount)
        elif side == 'sell':
            return self.execute_sell_order(pair, 'neutral', 0, 0, amount)
        else:
            self.logger.warning(f"Invalid trade side '{side}' for {pair}")
            return None

    def execute_buy_order(self, pair, market_regime, recent_performance, volatility, amount):
        try:
            order = self.client.create_market_buy_order(pair, amount)
            self.logger.info(f"Buy order executed for {pair}: {order}")
            
            entry_price = float(order['price'])
            self.set_trailing_stop_loss(pair, entry_price, initial_stop_loss_percentage=0.02, trailing_distance_percentage=0.01)
            
            return order
        except Exception as e:
            self.logger.error(f"Error executing buy order for {pair}: {e}")
            return None

    def execute_sell_order(self, pair, market_regime, recent_performance, volatility, amount):
        try:
            order = self.client.create_market_sell_order(pair, amount)
            self.logger.info(f"Sell order executed for {pair}: {order}")
            return order
        except Exception as e:
            self.logger.error(f"Error executing sell order for {pair}: {e}")
            return None

    def set_trailing_stop_loss(self, pair, entry_price, initial_stop_loss_percentage, trailing_distance_percentage):
        initial_stop_loss = entry_price * (1 - initial_stop_loss_percentage)
        trailing_distance = entry_price * trailing_distance_percentage
        self.trailing_stop_loss_orders[pair] = (initial_stop_loss, trailing_distance)
        self.logger.info(f"Trailing stop loss set for {pair} at {initial_stop_loss} with distance {trailing_distance}")

    def update_trailing_stop_loss(self, pair, current_price):
        if pair in self.trailing_stop_loss_orders:
            stop_loss, trailing_distance = self.trailing_stop_loss_orders[pair]
            new_stop_loss = current_price - trailing_distance
            if new_stop_loss > stop_loss:
                self.trailing_stop_loss_orders[pair] = (new_stop_loss, trailing_distance)
                self.logger.info(f"Updated trailing stop loss for {pair} to {new_stop_loss}")

    def check_stop_losses_continuously(self):
        while not self.stop_learning:
            self.check_stop_losses()
            time.sleep(1)  # Check every second

    def check_stop_losses(self):
        for pair, (stop_loss, _) in list(self.trailing_stop_loss_orders.items()):
            try:
                current_price = self.client.fetch_ticker(pair)['last']
                self.update_trailing_stop_loss(pair, current_price)
                if current_price <= stop_loss:
                    self.execute_sell_order(pair, 'neutral', 0, 0, self.get_position_size(pair))
                    del self.trailing_stop_loss_orders[pair]
                    self.logger.info(f"Trailing stop loss triggered for {pair} at {current_price}")
            except Exception as e:
                self.logger.error(f"Error checking stop loss for {pair}: {e}")

    def place_limit_order(self, pair, side, price, amount):
        try:
            order = self.client.create_limit_order(pair, side, amount, price)
            self.logger.info(f"Placed {side} limit order for {pair} at {price}")
            return order
        except Exception as e:
            self.logger.error(f"Error placing limit order: {e}")
            return None

    def get_position_size(self, pair):
        try:
            positions = self.client.fetch_positions([pair])
            for position in positions:
                if position['symbol'] == pair:
                    return float(position['amount'])
            return 0
        except Exception as e:
            self.logger.error(f"Error getting position size for {pair}: {e}")
            return 0

    def close_all_positions(self):
        for pair in self.currency_pairs:
            position_size = self.get_position_size(pair)
            if position_size > 0:
                self.execute_sell_order(pair, 'neutral', 0, 0, position_size)
        self.logger.info("All positions closed")

    def check_profit_taking(self, pair):
        position_size = self.get_position_size(pair)
        if position_size > 0:
            try:
                current_price = self.client.fetch_ticker(pair)['last']
                entry_price = self.get_entry_price(pair)
                profit_percentage = (current_price - entry_price) / entry_price
                if profit_percentage >= self.profit_taking_threshold:
                    self.execute_sell_order(pair, 'neutral', 0, 0, position_size / 2)
                    self.logger.info(f"Profit taking: Sold half position of {pair} at {profit_percentage*100:.2f}% profit")
            except Exception as e:
                self.logger.error(f"Error in profit taking for {pair}: {e}")

    def get_entry_price(self, pair):
        try:
            positions = self.client.fetch_positions([pair])
            for position in positions:
                if position['symbol'] == pair:
                    return float(position['entryPrice'])
            return None
        except Exception as e:
            self.logger.error(f"Error getting entry price for {pair}: {e}")
            return None

    def optimize_strategy(self, strategy):
        param_space = strategy.get_param_space()
        best_params = self.bayesian_optimize_hyperparameters(strategy, param_space)
        strategy.set_params(best_params)
        self.logger.info(f"Strategy {strategy.__class__.__name__} optimized with parameters: {best_params}")

    def bayesian_optimize_hyperparameters(self, strategy, param_space, n_iter=50):
        @skopt.utils.use_named_args(param_space)
        def objective(**params):
            for param, value in params.items():
                setattr(strategy, param, value)
            performance = self.backtest(strategy)
            return -performance  # We want to maximize performance, so return negative

        result = skopt.gp_minimize(objective, param_space, n_calls=n_iter)
        self.logger.info(f"Bayesian optimization completed for {strategy.__class__.__name__}")
        return result.x

    def backtest(self, strategy, start_date=None, end_date=None):
        if start_date is None:
            start_date = self.client.parse8601('1 month ago UTC')
        if end_date is None:
            end_date = self.client.milliseconds()

        total_profit = 0
        for pair in self.currency_pairs:
            df = self.fetch_market_data(pair, start_date=start_date, end_date=end_date)
            position = None
            entry_price = 0
            for i in range(len(df)):
                signal = strategy.execute(pair)
                current_price = df['close'].iloc[i]
                if signal == 'buy' and position is None:
                    position = 'long'
                    entry_price = current_price * (1 + self.slippage)
                elif signal == 'sell' and position == 'long':
                    exit_price = current_price * (1 - self.slippage)
                    profit = (exit_price - entry_price) / entry_price
                    profit -= self.transaction_fee * 2
                    total_profit += profit
                    position = None
                elif signal == 'close' and position is not None:
                    exit_price = current_price * (1 - self.slippage)
                    profit = (exit_price - entry_price) / entry_price if position == 'long' else (entry_price - exit_price) / entry_price
                    profit -= self.transaction_fee * 2
                    total_profit += profit
                    position = None

        return total_profit

    def update_news(self, pair):
        news = self.fetch_news(pair)
        self.logger.info(f"Updated news for {pair}")

    def run(self):
        self.logger.info("Starting trading bot")
        while True:
            try:
                observations = self.marl_env.reset()
                for _ in range(1000):  # Example episode length
                    actions = [agent.predict(obs)[0] for agent, obs in zip(self.marl_env.agents, observations)]
                    observations, rewards, dones, _ = self.marl_env.step(actions)
                    if all(dones):
                        break

                for pair in self.currency_pairs:
                    self.strategy_manager.update_active_strategies(pair)
                    
                    df = self.fetch_market_data(pair)
                    market_regime = self.detect_market_regime(pair)
                    volatility = df['volatility'].iloc[-1]
                    recent_performance = self.get_recent_performance(pair)

                    self.set_dynamic_leverage(pair, volatility)

                    ensemble_prediction = self.predict_with_ensemble(pair)
                    
                    weighted_signals = self.strategy_manager.get_weighted_signals(pair)
                    
                    if weighted_signals['buy'] > weighted_signals['sell'] and ensemble_prediction == 1:
                        position_size = self.capital_manager.calculate_position_size(pair, risk_per_trade=0.01)
                        buy_order = self.execute_buy_order(pair, market_regime, recent_performance, volatility, position_size)
                        if buy_order:
                            entry_price = float(buy_order['price'])
                            self.set_trailing_stop_loss(pair, entry_price, initial_stop_loss_percentage=0.02, trailing_distance_percentage=0.01)
                    elif weighted_signals['sell'] > weighted_signals['buy'] and ensemble_prediction == 0:
                        position_size = self.get_position_size(pair)
                        if position_size > 0:
                            self.execute_sell_order(pair, market_regime, recent_performance, volatility, position_size)
                    
                    self.check_profit_taking(pair)
                    self.strategy_manager.execute_all_strategies(pair)
                    
                    self.capital_manager.rebalance_capital()
                    self.capital_manager.reinvest_gains()
                    self.spreadsheet_tracker.update_performance(self)
                    
                    current_balance = self.client.fetch_balance()['total']['USDT']
                    self.portfolio_values.append(current_balance)
                    drawdown = (max(self.portfolio_values) - current_balance) / max(self.portfolio_values)
                    if drawdown > self.global_stop_loss_percentage:
                        self.logger.warning(f"Global stop loss triggered. Drawdown: {drawdown*100:.2f}%")
                        self.close_all_positions()
                        return

                    if time.time() - self.last_strategy_optimization > self.strategy_optimization_interval:
                        self.strategy_manager.optimize_all_strategies()
                        self.last_strategy_optimization = time.time()
                    
                    if time.time() - self.last_news_update > self.news_update_interval:
                        self.update_news(pair)
                        self.last_news_update = time.time()
                    
                    if time.time() % 3600 < 60:  # Once an hour
                        self.save_models()
                    
                    self.strategy_manager.update_strategy_weights()

                for agent in self.marl_env.agents:
                    agent.learn(total_timesteps=1000)  # Adjust as needed

            except ccxt.NetworkError:
                self.logger.error("Network error occurred. Attempting to reconnect...")
                self.reconnect()
            except Exception as e:
                self.logger.error(f"An error occurred: {e}")
                self.handle_error(e)

            time.sleep(60)  # Check every minute

    def stop(self):
        self.stop_learning = True
        self.learning_thread.join()
        self.stop_loss_thread.join()
        self.logger.info("Bot stopped")

    def reconnect(self):
        max_retries = 5
        for i in range(max_retries):
            try:
                self.client = ccxt.binance({'apiKey': self.api_key, 'secret': self.api_secret})
                self.client.load_markets()
                self.logger.info("Reconnected successfully")
                return
            except Exception as e:
                self.logger.error(f"Reconnection attempt {i+1} failed: {e}")
                time.sleep(2 ** i)  # Exponential backoff
        self.logger.critical("Failed to reconnect after multiple attempts")

class StrategyManager:
    def __init__(self, bot):
        self.bot = bot
        self.strategies = {
            'trend_following': TrendFollowingStrategy(bot),
            'mean_reversion': MeanReversionStrategy(bot),
            'breakout': BreakoutStrategy(bot),
            'arbitrage': ArbitrageStrategy(bot),
            'market_making': MarketMakingStrategy(bot),
            'pairs_trading': PairsTradingStrategy(bot),
            'news_based': NewsBasedEventStrategy(bot),
            'grid_trading': GridTradingStrategy(bot),
            'dollar_cost_averaging': DollarCostAveragingStrategy(bot),
            'counter_trend': CounterTrendStrategy(bot),
            'risk_parity': RiskParityStrategy(bot),
            'reversal': ReversalStrategy(bot),
            'scalping': ScalpingStrategy(bot),
            'momentum': MomentumStrategy(bot),
            'sentiment': SentimentBasedStrategy(bot)
        }
        self.active_strategies = {}

    def update_active_strategies(self, pair):
        market_regime = self.bot.detect_market_regime(pair)
        self.active_strategies[pair] = self.select_strategies_for_regime(market_regime)

    def select_strategies_for_regime(self, regime):
        if regime in ['Strong Bull', 'Bull', 'Volatile Bull']:
            return ['trend_following', 'breakout', 'momentum', 'sentiment']
        elif regime in ['Strong Bear', 'Bear', 'Volatile Bear']:
            return ['mean_reversion', 'risk_parity', 'counter_trend', 'sentiment']
        elif regime == 'Sideways':
            return ['market_making', 'pairs_trading', 'grid_trading', 'scalping']
        elif regime == 'Trending':
            return ['trend_following', 'momentum', 'breakout', 'news_based']
        elif regime == 'Reversal':
            return ['mean_reversion', 'counter_trend', 'reversal', 'news_based']
        else:  # Default case
            return ['market_making', 'scalping', 'arbitrage', 'dollar_cost_averaging']

    def get_weighted_signals(self, pair):
        weighted_signals = {'buy': 0, 'sell': 0}
        for strategy_name in self.active_strategies[pair]:
            signal = self.strategies[strategy_name].execute(pair)
            if signal in ['buy', 'sell']:
                weighted_signals[signal] += self.bot.strategy_weights[strategy_name]
        return weighted_signals

    def execute_all_strategies(self, pair):
        for strategy_name in self.active_strategies[pair]:
            self.strategies[strategy_name].execute(pair)

    def optimize_all_strategies(self):
        for strategy in self.strategies.values():
            self.bot.optimize_strategy(strategy)

    def update_strategy_weights(self):
        for strategy_name, strategy in self.strategies.items():
            performance = strategy.get_performance()
            self.bot.strategy_weights[strategy_name] = max(0.1, performance)  # Minimum weight of 0.1

class CapitalManager:
    def __init__(self, bot):
        self.bot = bot

    def calculate_position_size(self, pair, risk_per_trade):
        with self.bot.balance_lock:
            balance = self.bot.client.fetch_balance()
            equity = balance['total']['USDT']
        current_price = self.bot.client.fetch_ticker(pair)['close']
        
        atr = self.bot.calculate_atr(pair)
        stop_loss_pips = atr * 2  # Example: Set stop loss at 2 * ATR
        
        leverage = self.bot.current_leverage[pair]
        position_size = (equity * risk_per_trade * leverage) / stop_loss_pips
        return position_size / current_price

    def rebalance_capital(self):
        with self.bot.balance_lock:
            balances = self.bot.client.fetch_balance()
            total_balance = balances['total']['USDT']
            target_allocation = total_balance / len(self.bot.currency_pairs)

            for pair in self.bot.currency_pairs:
                base_asset = pair.split('/')[0]
                available_amount = balances['total'][base_asset]
                current_price = self.bot.client.fetch_ticker(pair)['close']
                target_amount = target_allocation / current_price

                if available_amount < target_amount * 0.9:  # 10% threshold
                    amount_to_buy = target_amount - available_amount
                    self.bot.execute_buy_order(pair, 'neutral', 0, 0, amount_to_buy)
                elif available_amount > target_amount * 1.1:  # 10% threshold
                    amount_to_sell = available_amount - target_amount
                    self.bot.execute_sell_order(pair, 'neutral', 0, 0, amount_to_sell)

    def reinvest_gains(self):
        with self.bot.balance_lock:
            current_balance = self.bot.client.fetch_balance()['total']['USDT']
            gains = current_balance - self.bot.initial_capital
            if gains > 0:
                self.bot.initial_capital = current_balance
                self.bot.total_profit += gains
                self.bot.logger.info(f"Reinvested gains: {gains} USDT. New capital: {self.bot.initial_capital} USDT")
                self.rebalance_capital()

class BaseStrategy:
    def __init__(self, bot):
        self.bot = bot
        self.performance = deque(maxlen=100)

    def execute(self, pair):
        raise NotImplementedError("Subclass must implement abstract method")

    def get_performance(self):
        return sum(self.performance) / len(self.performance) if self.performance else 0

    def update_performance(self, profit):
        self.performance.append(profit)

    def get_param_space(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def set_params(self, params):
        raise NotImplementedError("Subclass must implement abstract method")

class TrendFollowingStrategy(BaseStrategy):
    def __init__(self, bot):
        super().__init__(bot)
        self.short_period = 20
        self.long_period = 50

    def execute(self, pair):
        df = self.bot.fetch_market_data(pair)
        ema_short = talib.EMA(df['close'], timeperiod=self.short_period)
        ema_long = talib.EMA(df['close'], timeperiod=self.long_period)
        
        if ema_short.iloc[-1] > ema_long.iloc[-1] and ema_short.iloc[-2] <= ema_long.iloc[-2]:
            return 'buy'
        elif ema_short.iloc[-1] < ema_long.iloc[-1] and ema_short.iloc[-2] >= ema_long.iloc[-2]:
            return 'sell'
        return 'hold'

    def get_param_space(self):
        return [
            Integer(10, 30, name='short_period'),
            Integer(40, 60, name='long_period')
        ]

    def set_params(self, params):
        self.short_period, self.long_period = params

class MeanReversionStrategy(BaseStrategy):
    def __init__(self, bot):
        super().__init__(bot)
        self.window = 20
        self.std_dev = 2

    def execute(self, pair):
        df = self.bot.fetch_market_data(pair)
        z_score = (df['close'] - df['close'].rolling(window=self.window).mean()) / df['close'].rolling(window=self.window).std()
        
        if z_score.iloc[-1] < -self.std_dev:
            return 'buy'
        elif z_score.iloc[-1] > self.std_dev:
            return 'sell'
        return 'hold'

    def get_param_space(self):
        return [
            Integer(10, 50, name='window'),
            Real(1.5, 3.0, name='std_dev')
        ]

    def set_params(self, params):
        self.window, self.std_dev = params

# Implement other strategy classes (BreakoutStrategy, ArbitrageStrategy, etc.) similarly

class MultiAgentTradingEnv(gym.Env):
    def __init__(self, bot, pairs, num_agents=3):
        super().__init__()
        self.bot = bot
        self.pairs = pairs
        self.num_agents = num_agents
        self.action_space = spaces.Discrete(3)  # Buy, Sell, Hold
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,))
        self.agents = [PPO('MlpPolicy', self) for _ in range(num_agents)]

    def reset(self):
        self.current_step = 0
        self.total_profit = 0
        self.portfolio_value = self.bot.initial_capital
        return [self._next_observation(pair) for pair in self.pairs]

    def step(self, actions):
        rewards = []
        observations = []
        for i, pair in enumerate(self.pairs):
            action = actions[i]
            self._take_action(pair, action)
            reward = self._calculate_reward(pair)
            rewards.append(reward)
            observations.append(self._next_observation(pair))

        self.current_step += 1
        done = self.current_step > 1000
        info = {'portfolio_value': self.portfolio_value}
        return observations, rewards, [done] * self.num_agents, info

    def _next_observation(self, pair):
        df = self.bot.fetch_market_data(pair)
        return df[['close', 'volume', 'rsi', 'macd', 'macdsignal']].iloc[-1].values

    def _take_action(self, pair, action):
        amount = self.bot.capital_manager.calculate_position_size(pair, risk_per_trade=0.01)
        if action == 0:  # Buy
            self.bot.execute_buy_order(pair, 'neutral', 0, 0, amount)
        elif action == 1:  # Sell
            self.bot.execute_sell_order(pair, 'neutral', 0, 0, amount)

    def _calculate_reward(self, pair):
        current_price = self.bot.client.fetch_ticker(pair)['last']
        position = self.bot.get_position_size(pair)
        if position > 0:
            profit = (current_price - self.bot.get_entry_price(pair)) * position
        else:
            profit = 0
        
        portfolio_change = self.bot.client.fetch_balance()['total']['USDT'] - self.portfolio_value
        self.portfolio_value += portfolio_change
        
        volatility = self.bot.fetch_market_data(pair)['volatility'].iloc[-1]
        sharpe_ratio = portfolio_change / (volatility * np.sqrt(252))  # Annualized Sharpe ratio
        
        reward = profit + sharpe_ratio + portfolio_change
        return reward

if __name__ == "__main__":
    api_key = "your_binance_api_key"
    api_secret = "your_binance_api_secret"
    initial_capital = 10000  # USDT
    
    bot = BinanceTradingBot(api_key, api_secret, initial_capital, email_alerts=True)
    
    try:
        bot.run()
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Stopping the bot...")
        bot.stop()
        print("Bot stopped successfully")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        bot.logger.error(f"Critical error: {e}")
        bot.send_email_alert(f"Critical error: {e}")
    finally:
        print("Cleaning up...")
        bot.close_all_positions()
        bot.save_models()
        print("Cleanup completed. Bot shutdown.")

print("Trading bot script execution completed.")
