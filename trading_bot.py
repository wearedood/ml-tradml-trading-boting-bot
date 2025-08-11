#!/usr/bin/env python3
"""
ML-Powered Cryptocurrency Trading Bot

This bot uses machine learning algorithms to analyze market data
and execute automated trading strategies.
"""

import numpy as np
import pandas as pd
import ccxt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import ta
import logging
from datetime import datetime, timedelta
import time
import json

class MLTradingBot:
    def __init__(self, exchange_id='binance', api_key=None, secret=None):
        self.exchange = getattr(ccxt, exchange_id)({
            'apiKey': api_key,
            'secret': secret,
            'sandbox': True,  # Use testnet
            'enableRateLimit': True,
        })
        
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Trading parameters
        self.symbol = 'BTC/USDT'
        self.timeframe = '1h'
        self.lookback_period = 100
        self.risk_per_trade = 0.02  # 2% risk per trade
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def fetch_ohlcv_data(self, limit=500):
        """Fetch OHLCV data from exchange"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                self.symbol, self.timeframe, limit=limit
            )
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            self.logger.error(f"Error fetching data: {e}")
            return None
    
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators for ML features"""
        # Moving averages
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
        
        # RSI
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_lower'] = bollinger.bollinger_lband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        
        # Volume indicators
        df['volume_sma'] = ta.volume.volume_sma(df['close'], df['volume'])
        
        # Price change features
        df['price_change'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for ML model"""
        feature_columns = [
            'sma_20', 'sma_50', 'ema_12', 'ema_26', 'rsi',
            'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_lower', 'bb_middle',
            'volume_sma', 'price_change', 'high_low_ratio'
        ]
        
        # Create target variable (future price movement)
        df['target'] = df['close'].shift(-1) / df['close'] - 1
        
        # Remove NaN values
        df = df.dropna()
        
        X = df[feature_columns].values
        y = df['target'].values
        
        return X, y, df
    
    def train_model(self):
        """Train the ML model"""
        self.logger.info("Fetching training data...")
        df = self.fetch_ohlcv_data(limit=1000)
        
        if df is None:
            return False
        
        df = self.calculate_technical_indicators(df)
        X, y, _ = self.prepare_features(df)
        
        if len(X) < 50:
            self.logger.error("Insufficient data for training")
            return False
        
        # Split data for training (use 80% for training)
        split_idx = int(len(X) * 0.8)
        X_train, y_train = X[:split_idx], y[:split_idx]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        self.logger.info("Model training completed")
        return True
    
    def predict_price_movement(self):
        """Predict future price movement"""
        if not self.is_trained:
            self.logger.error("Model not trained yet")
            return None
        
        df = self.fetch_ohlcv_data(limit=self.lookback_period)
        if df is None:
            return None
        
        df = self.calculate_technical_indicators(df)
        X, _, _ = self.prepare_features(df)
        
        if len(X) == 0:
            return None
        
        # Use the latest data point for prediction
        latest_features = X[-1].reshape(1, -1)
        latest_features_scaled = self.scaler.transform(latest_features)
        
        prediction = self.model.predict(latest_features_scaled)[0]
        return prediction
    
    def calculate_position_size(self, account_balance, entry_price, stop_loss_price):
        """Calculate position size based on risk management"""
        risk_amount = account_balance * self.risk_per_trade
        price_diff = abs(entry_price - stop_loss_price)
        position_size = risk_amount / price_diff
        return position_size
    
    def execute_trade(self, signal, confidence):
        """Execute trade based on ML prediction"""
        try:
            # Get current balance
            balance = self.exchange.fetch_balance()
            usdt_balance = balance['USDT']['free']
            
            # Get current price
            ticker = self.exchange.fetch_ticker(self.symbol)
            current_price = ticker['last']
            
            # Calculate position size
            if signal == 'BUY' and confidence > 0.6:
                stop_loss_price = current_price * 0.98  # 2% stop loss
                position_size = self.calculate_position_size(
                    usdt_balance, current_price, stop_loss_price
                )
                
                if position_size * current_price <= usdt_balance:
                    order = self.exchange.create_market_buy_order(
                        self.symbol, position_size
                    )
                    self.logger.info(f"BUY order executed: {order}")
                    
            elif signal == 'SELL' and confidence > 0.6:
                btc_balance = balance['BTC']['free']
                if btc_balance > 0:
                    order = self.exchange.create_market_sell_order(
                        self.symbol, btc_balance
                    )
                    self.logger.info(f"SELL order executed: {order}")
                    
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
    
    def run_trading_loop(self):
        """Main trading loop"""
        self.logger.info("Starting ML Trading Bot...")
        
        # Train model initially
        if not self.train_model():
            self.logger.error("Failed to train model. Exiting.")
            return
        
        while True:
            try:
                # Get prediction
                prediction = self.predict_price_movement()
                
                if prediction is not None:
                    confidence = abs(prediction)
                    
                    if prediction > 0.005:  # 0.5% threshold
                        signal = 'BUY'
                    elif prediction < -0.005:
                        signal = 'SELL'
                    else:
                        signal = 'HOLD'
                    
                    self.logger.info(
                        f"Prediction: {prediction:.4f}, Signal: {signal}, "
                        f"Confidence: {confidence:.4f}"
                    )
                    
                    # Execute trade if confidence is high enough
                    if signal != 'HOLD':
                        self.execute_trade(signal, confidence)
                
                # Wait before next iteration
                time.sleep(3600)  # 1 hour
                
            except KeyboardInterrupt:
                self.logger.info("Trading bot stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying

if __name__ == "__main__":
    # Initialize bot (replace with your API credentials)
    bot = MLTradingBot(
        exchange_id='binance',
        api_key='your_api_key_here',
        secret='your_secret_here'
    )
    
    # Start trading
    bot.run_trading_loop()
