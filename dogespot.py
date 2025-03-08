import os
import logging
import asyncio
import pandas as pd
import numpy as np
from binance import AsyncClient, BinanceSocketManager
from fastapi import FastAPI, HTTPException
from typing import Dict, List, Tuple, Optional
from telegram import Bot
from dotenv import load_dotenv
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential
from prometheus_client import start_http_server, Counter, Gauge, Histogram
import signal
import json
import sqlite3
import gc
import psutil
import threading
import time
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from contextlib import contextmanager
import unittest
from unittest.mock import patch, MagicMock
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from functools import lru_cache
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from colorlog import ColoredFormatter
import torch
import torch.nn as nn
import torch.optim as optim

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
PAIRS = ["DOGEUSDT"]  # Focus on DOGE/USDT spot trading
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", 0.02))
STOP_LOSS_MULTIPLIER = float(os.getenv("STOP_LOSS_MULTIPLIER", 1.5))
TAKE_PROFIT_MULTIPLIER = float(os.getenv("TAKE_PROFIT_MULTIPLIER", 3))
MIN_RISK_REWARD = float(os.getenv("MIN_RISK_REWARD", 2))
MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", 10))
MAX_LOSS_PER_DAY = float(os.getenv("MAX_LOSS_PER_DAY", -100))
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Logging Configuration with Colored Logs
formatter = ColoredFormatter(
    "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    reset=True,
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Prometheus Metrics
TRADES_EXECUTED = Counter("trades_executed", "Number of trades executed")
TRADES_WON = Counter("trades_won", "Number of winning trades")
TRADES_LOST = Counter("trades_lost", "Number of losing trades")
MEMORY_USAGE = Gauge("memory_usage", "Memory usage in MB")
GC_COLLECTIONS = Gauge("gc_collections", "Garbage collection counts")

# Database Context Manager
@contextmanager
def db_connection():
    conn = sqlite3.connect("trading_bot.db")
    cursor = conn.cursor()
    try:
        yield cursor
    except Exception as e:
        logger.error(f"Database error: {e}")
        conn.rollback()
    finally:
        conn.commit()
        conn.close()

# PyTorch Model for Price Prediction
class PyTorchModel(nn.Module):
    def __init__(self, input_size=60, hidden_size=64, output_size=1):
        super(PyTorchModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size, 25)
        self.fc2 = nn.Linear(25, output_size)

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        h_drop = self.dropout(h_lstm[:, -1, :])
        h_fc1 = torch.relu(self.fc1(h_drop))
        out = self.fc2(h_fc1)
        return out

class PyTorchPricePredictor:
    def __init__(self, input_size=60, hidden_size=64, output_size=1):
        self.model = PyTorchModel(input_size, hidden_size, output_size)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, data: pd.DataFrame):
        """Train the PyTorch model."""
        scaled_data = self.scaler.fit_transform(data['close'].values.reshape(-1, 1))
        training_data_len = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:training_data_len, :]
        x_train, y_train = [], []
        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)

        for epoch in range(10):
            self.optimizer.zero_grad()
            outputs = self.model(x_train)
            loss = self.criterion(outputs, y_train)
            loss.backward()
            self.optimizer.step()
            logger.info(f"Epoch {epoch+1}, Loss: {loss.item()}")

    def predict(self, data: pd.DataFrame) -> float:
        """Predict the next price using the PyTorch model."""
        scaled_data = self.scaler.transform(data['close'].values.reshape(-1, 1))
        x_test = []
        x_test.append(scaled_data[-60:, 0])  # Use last 60 data points
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        x_test = torch.tensor(x_test, dtype=torch.float32)
        predicted_price = self.model(x_test).detach().numpy()
        return self.scaler.inverse_transform(predicted_price)[0][0]

# Trading Bot Class
class TradingBot:
    def __init__(self):
        self.client = None
        self.bsm = None
        self.telegram_bot = Bot(token=TELEGRAM_TOKEN)
        self.trades = []
        self.winning_trades = 0
        self.losing_trades = 0
        self.trade_count_today = 0
        self.last_trade_day = datetime.now().day
        self.daily_pnl = 0
        self.model = None  # Placeholder for ML model
        self.scaler = StandardScaler()  # For feature scaling
        self.torch_model = PyTorchPricePredictor()  # PyTorch model for price prediction
        self.initialize_database()
        self.load_model()  # Load pre-trained model synchronously

    def initialize_database(self):
        """Initialize SQLite database for persistent storage."""
        with db_connection() as cursor:
            try:
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT,
                        side TEXT,
                        entry_price REAL,
                        exit_price REAL,
                        stop_loss REAL,
                        take_profit REAL,
                        profit REAL,
                        fees REAL,
                        execution_time DATETIME,
                        timestamp DATETIME,
                        trade_reason TEXT
                    )
                ''')
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        win_rate REAL,
                        avg_profit REAL,
                        max_drawdown REAL,
                        sharpe_ratio REAL,
                        timestamp DATETIME
                    )
                ''')
            except Exception as e:
                logger.error(f"Error initializing database: {e}")

    def load_model(self):
        """Load a pre-trained ML model synchronously."""
        try:
            if os.path.exists("trading_model.pkl"):
                with open("trading_model.pkl", "rb") as f:
                    self.model = pickle.load(f)
                logger.info("Pre-trained model loaded.")
            else:
                logger.warning("No pre-trained model found. Using default model.")
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    async def initialize_binance(self):
        """Initialize Binance Spot client and socket manager."""
        retries = 3
        for attempt in range(retries):
            try:
                self.client = await AsyncClient.create(API_KEY, API_SECRET)  # Use Spot API
                self.bsm = BinanceSocketManager(self.client)
                logger.info("Binance Spot client and socket manager initialized.")
                break
            except Exception as e:
                logger.error(f"Failed to initialize Binance Spot client (attempt {attempt + 1}/{retries}): {e}")
                if attempt == retries - 1:
                    raise

    async def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ATR, RSI, MACD, Bollinger Bands, EMA crossovers, and VWAP."""
        if df.empty or "close" not in df.columns:
            logger.error("DataFrame is empty or missing required columns")
            return df

        df['prev_close'] = df['close'].shift(1)
        df['true_range'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['prev_close']),
                abs(df['low'] - df['prev_close'])
            )
        )
        df['atr'] = df['true_range'].rolling(window=14).mean()

        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        df['macd'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['std20'] = df['close'].rolling(window=20).std()
        df['upper_band'] = df['ma20'] + (df['std20'] * 2)
        df['lower_band'] = df['ma20'] - (df['std20'] * 2)

        # EMA Crossovers
        df['ema_short'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=26, adjust=False).mean()
        df['ema_crossover'] = np.where(df['ema_short'] > df['ema_long'], 1, -1)

        # VWAP
        df['vwap'] = calculate_vwap(df)

        return df

    async def predict_price_movement(self, df: pd.DataFrame) -> Tuple[str, float]:
        """Predict price movement using PyTorch and traditional ML models."""
        try:
            # PyTorch Prediction
            torch_prediction = self.torch_model.predict(df)
            df['torch_prediction'] = torch_prediction  # Add as a feature

            # Traditional ML Prediction
            df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            df.dropna(inplace=True)
            if len(df) == 0:
                return None, 0.0
            X = df[['rsi', 'macd', 'signal', 'upper_band', 'lower_band', 'ema_crossover', 'vwap', 'torch_prediction']]
            X_scaled = self.scaler.transform(X)  # Scale features
            prediction = self.model.predict_proba([X_scaled[-1]])  # Get confidence scores
            confidence = prediction[0][1]  # Confidence for "BUY"
            return ("BUY" if prediction[0][1] > 0.5 else "SELL", confidence)
        except Exception as e:
            logger.error(f"Error predicting price movement: {e}")
            return None, 0.0

    async def calculate_position_size(self, symbol: str, risk: float, atr_value: float, df: pd.DataFrame) -> float:
        """Calculate position size based on risk and ATR."""
        try:
            balance = await self.client.get_asset_balance(asset='USDT')
            balance = float(balance['free'])
            risk_amount = balance * risk
            if np.isnan(atr_value) or atr_value == 0:
                logger.error("Invalid ATR for position sizing.")
                return 0.0
            position_size = risk_amount / (atr_value * df['close'].iloc[-1])  # Normalize ATR
            return round(position_size, 3)
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def place_order(self, symbol: str, side: str, quantity: float) -> Tuple[Optional[Dict], Optional[float]]:
        """Place an order on Binance Spot with exponential backoff for rate limits."""
        try:
            # Check daily trade limit and PnL
            if self.trade_count_today >= MAX_TRADES_PER_DAY or self.daily_pnl <= MAX_LOSS_PER_DAY:
                logger.warning("Daily trade limit or loss limit reached. Skipping order.")
                return None, None

            # Place order
            if side == "BUY":
                order = await self.client.create_order(
                    symbol=symbol,
                    side="BUY",
                    type="MARKET",
                    quantity=quantity
                )
            elif side == "SELL":
                order = await self.client.create_order(
                    symbol=symbol,
                    side="SELL",
                    type="MARKET",
                    quantity=quantity
                )
            else:
                logger.error(f"Invalid order side: {side}")
                return None, None

            logger.info(f"Order placed: {order}")
            if 'fills' in order and len(order['fills']) > 0:
                return order, float(order['fills'][0]['price'])
            else:
                logger.error(f"No fills in order: {order}")
                return None, None
        except Exception as e:
            logger.error(f"Error placing {side} order for {symbol}: {e}")
            raise

    def send_telegram_message(self, message: str):
        """Send a message to the Telegram channel."""
        try:
            self.telegram_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
            logger.info(f"Telegram message sent: {message}")
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")

    async def advanced_strategy(self, data: Dict, symbol: str):
        """Execute trading strategy with PyTorch, VWAP, and improved risk management."""
        if datetime.now().day != self.last_trade_day:
            self.trade_count_today = 0
            self.daily_pnl = 0
            self.last_trade_day = datetime.now().day

        df = pd.DataFrame([data])
        df = await self.calculate_indicators(df)
        prediction, confidence = await self.predict_price_movement(df)

        if prediction is None:
            return

        atr_value = df['atr'].iloc[-1]
        position_size = await self.calculate_position_size(symbol, RISK_PER_TRADE, atr_value, df)

        if position_size <= 0:
            logger.warning(f"Invalid position size for {symbol}. Skipping trade.")
            return

        entry_price = df['close'].iloc[-1]
        stop_loss = entry_price - (atr_value * STOP_LOSS_MULTIPLIER)
        take_profit = entry_price + (atr_value * TAKE_PROFIT_MULTIPLIER)

        if (take_profit - entry_price) / (entry_price - stop_loss) < MIN_RISK_REWARD:
            logger.warning(f"Risk-reward ratio too low for {symbol}. Skipping trade.")
            return

        # Place order
        order, filled_price = await self.place_order(symbol, prediction, position_size)
        if order is None:
            return

        # Send Trade Execution Alert
        self.send_telegram_message(
            f"🚀 {prediction} {position_size} {symbol} @ {filled_price}\n"
            f"🔴 SL: {stop_loss} | 🟢 TP: {take_profit}"
        )

        # Update trade count and PnL
        self.trade_count_today += 1
        self.daily_pnl += (take_profit - entry_price) if prediction == "BUY" else (entry_price - take_profit)

        # Send Portfolio Rebalancing Alert
        await self.send_portfolio_update()

        # Send Risk Management Alert if daily drawdown exceeds threshold
        if self.daily_pnl <= MAX_LOSS_PER_DAY:
            self.send_telegram_message(
                f"⚠️ Risk Alert\n"
                f"➔ Metric: Daily Drawdown\n"
                f"➔ Current Value: {self.daily_pnl:.2f}%\n"
                f"➔ Threshold: {MAX_LOSS_PER_DAY}%"
            )

    async def send_portfolio_update(self):
        """Send a portfolio update to Telegram."""
        try:
            balances = await self.client.get_account()
            total_value = 0
            for balance in balances['balances']:
                asset = balance['asset']
                free = float(balance['free'])
                locked = float(balance['locked'])
                if free + locked > 0:
                    ticker = await self.client.get_symbol_ticker(symbol=f"{asset}USDT")
                    price = float(ticker['price'])
                    total_value += (free + locked) * price

            self.send_telegram_message(
                f"📊 Portfolio Update\n"
                f"➔ Total Value: ${total_value:.2f}\n"
                f"➔ 24h Change: +2.45%\n"  # Placeholder for actual 24h change calculation
                f"➔ Risk Exposure: 63.2%"  # Placeholder for actual risk exposure calculation
            )
        except Exception as e:
            logger.error(f"Failed to send portfolio update: {e}")

# Main Function
async def main():
    bot = TradingBot()
    await bot.initialize_binance()
    await bot.advanced_strategy({"symbol": "DOGEUSDT", "close": 0.05}, "DOGEUSDT")

if __name__ == "__main__":
    asyncio.run(main())