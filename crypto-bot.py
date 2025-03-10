import os
import logging
import asyncio
import pandas as pd
import numpy as np
import sqlite3
import signal
import psutil
import threading
import time
import joblib
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager
from binance import AsyncClient, BinanceSocketManager
from binance.exceptions import BinanceAPIException
from telegram import Bot
from dotenv import load_dotenv
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential
from prometheus_client import start_http_server, Counter, Gauge, Histogram
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from colorlog import ColoredFormatter

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
SYMBOL = "DOGEUSDT"
DB_NAME = "trading_bot.db"
MODEL_DIR = "models/"
HISTORICAL_DAYS = 30
CANDLE_INTERVAL = '1h'
PATTERN_CONFIRMATION_CANDLES = 3

# Initialize logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = ColoredFormatter(
    "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class TradingBot:
    def __init__(self):
        self.client = None
        self.bsm = None
        self.models = {
            'random_forest': RandomForestRegressor(),
            'xgboost': XGBRegressor()
        }
        self.scaler = StandardScaler()
        self.running = False
        self.db_lock = threading.Lock()
        self.active_trades = {}
        os.makedirs(MODEL_DIR, exist_ok=True)

    # Database and initialization methods (similar to previous version)
    # ... [Include previous database and initialization methods here] ...

    async def start(self):
        """Start main trading loop with pattern detection"""
        self.running = True
        asyncio.create_task(self._monitor_system())
        asyncio.create_task(self._trade_loop())
        asyncio.create_task(self._pattern_analysis_loop())
        asyncio.create_task(self._monitor_active_trades())
        asyncio.create_task(self._retrain_models())
        logger.info("All systems operational")

    async def _pattern_analysis_loop(self):
        """Analyze candlestick patterns continuously"""
        async with self.bsm.kline_socket(SYMBOL.lower(), interval=CANDLE_INTERVAL) as ks:
            while self.running:
                try:
                    msg = await ks.recv()
                    df = self._process_kline_message(msg)
                    patterns = self._detect_candlestick_patterns(df)
                    
                    if patterns:
                        await self._handle_patterns(patterns, df)
                        
                    await asyncio.sleep(1)
                except Exception as e:
                    logger.error(f"Pattern analysis error: {str(e)}")
                    await asyncio.sleep(5)

    def _process_kline_message(self, msg: Dict) -> pd.DataFrame:
        """Convert websocket message to DataFrame"""
        kline = msg['k']
        return pd.DataFrame([{
            'timestamp': datetime.fromtimestamp(msg['E']/1000),
            'open': float(kline['o']),
            'high': float(kline['h']),
            'low': float(kline['l']),
            'close': float(kline['c']),
            'volume': float(kline['v']),
            'closed': kline['x']
        }])

    def _detect_candlestick_patterns(self, df: pd.DataFrame) -> List[str]:
        """Detect common candlestick patterns"""
        patterns = []
        
        # Calculate candle properties
        df['body'] = df['close'] - df['open']
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['total_range'] = df['high'] - df['low']
        
        # Detect patterns
        for idx, row in df.iterrows():
            if self._is_doji(row):
                patterns.append('DOJI')
            if self._is_hammer(row):
                patterns.append('HAMMER')
            if self._is_engulfing(df, idx):
                patterns.append('ENGULFING')
            if self._is_morning_star(df, idx):
                patterns.append('MORNING_STAR')
                
        return patterns

    def _is_doji(self, row: pd.Series) -> bool:
        """Detect Doji pattern"""
        body_size = abs(row['body'])
        total_range = row['total_range']
        return total_range > 0 and (body_size / total_range) < 0.1

    def _is_hammer(self, row: pd.Series) -> bool:
        """Detect Hammer pattern"""
        lower_wick_ratio = row['lower_wick'] / row['total_range']
        body_ratio = abs(row['body']) / row['total_range']
        return (lower_wick_ratio > 0.6 and 
                body_ratio < 0.3 and 
                row['close'] > row['open'])

    def _is_engulfing(self, df: pd.DataFrame, idx: int) -> bool:
        """Detect Engulfing pattern"""
        if idx == 0:
            return False
        prev = df.iloc[idx-1]
        current = df.iloc[idx]
        return (current['body'] > 0 and 
                prev['body'] < 0 and 
                current['open'] < prev['close'] and 
                current['close'] > prev['open'])

    def _is_morning_star(self, df: pd.DataFrame, idx: int) -> bool:
        """Detect Morning Star pattern"""
        if idx < 2:
            return False
        candle1 = df.iloc[idx-2]
        candle2 = df.iloc[idx-1]
        candle3 = df.iloc[idx]
        return (candle1['body'] < 0 and
                abs(candle2['body']) < candle1['body'] * 0.3 and
                candle3['body'] > 0 and
                candle3['close'] > candle1['close'])

    async def _handle_patterns(self, patterns: List[str], df: pd.DataFrame):
        """Handle detected candlestick patterns"""
        current_price = df.iloc[-1]['close']
        
        for pattern in patterns:
            logger.info(f"Detected {pattern} pattern at {current_price}")
            
            # Store pattern in database
            with self.db_connection() as conn:
                conn.execute('''
                    INSERT INTO patterns (timestamp, symbol, pattern, price)
                    VALUES (?, ?, ?, ?)
                ''', (datetime.now(), SYMBOL, pattern, current_price))
                conn.commit()
            
            # Generate trading signal
            signal = self._pattern_to_signal(pattern)
            if signal:
                await self._execute_trade(signal, current_price)

    def _pattern_to_signal(self, pattern: str) -> Optional[str]:
        """Convert pattern to trading signal"""
        bullish = ['HAMMER', 'MORNING_STAR', 'ENGULFING']
        bearish = ['DOJI']  # Simplified example
        
        if pattern in bullish:
            return 'BUY'
        elif pattern in bearish:
            return 'SELL'
        return None

    async def _execute_trade(self, signal: str, price: float):
        """Execute trade based on signal"""
        try:
            # Implement your actual trade execution logic here
            trade_id = str(int(time.time()))
            
            self.active_trades[trade_id] = {
                'entry_price': price,
                'symbol': SYMBOL,
                'size': self._calculate_position_size(price),
                'status': 'OPEN',
                'stop_loss': price * 0.95,
                'take_profit': price * 1.05,
                'timestamp': datetime.now()
            }
            
            logger.info(f"Executed {signal} order at {price}")
            
            # Store trade in database
            with self.db_connection() as conn:
                conn.execute('''
                    INSERT INTO trades 
                    (id, symbol, price, quantity, status, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (trade_id, SYMBOL, price, 
                      self.active_trades[trade_id]['size'],
                      'OPEN', datetime.now()))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Trade execution failed: {str(e)}")

    def _calculate_position_size(self, price: float) -> float:
        """Calculate position size based on risk management"""
        # Implement your risk management logic here
        return 100.0 / price  # Example: $100 position

    async def _monitor_active_trades(self):
        """Monitor and manage open trades"""
        while self.running:
            try:
                current_price = await self._get_current_price()
                
                for trade_id, trade in list(self.active_trades.items()):
                    if trade['status'] == 'OPEN':
                        profit = (current_price - trade['entry_price']) / trade['entry_price']
                        
                        # Check exit conditions
                        if current_price <= trade['stop_loss']:
                            await self._close_trade(trade_id, current_price, 'STOP_LOSS')
                        elif current_price >= trade['take_profit']:
                            await self._close_trade(trade_id, current_price, 'TAKE_PROFIT')
                            
                await asyncio.sleep(10)
            except Exception as e:
                logger.error(f"Trade monitoring error: {str(e)}")
                await asyncio.sleep(5)

    async def _close_trade(self, trade_id: str, price: float, reason: str):
        """Close an active trade"""
        try:
            trade = self.active_trades[trade_id]
            trade['exit_price'] = price
            trade['status'] = 'CLOSED'
            trade['close_reason'] = reason
            trade['close_time'] = datetime.now()
            
            # Update database
            with self.db_connection() as conn:
                conn.execute('''
                    UPDATE trades SET
                    exit_price = ?,
                    status = ?,
                    close_reason = ?,
                    close_time = ?
                    WHERE id = ?
                ''', (price, 'CLOSED', reason, datetime.now(), trade_id))
                conn.commit()
            
            logger.info(f"Closed trade {trade_id} ({reason}) at {price}")
            del self.active_trades[trade_id]
            
        except Exception as e:
            logger.error(f"Failed to close trade {trade_id}: {str(e)}")

    async def _get_current_price(self) -> float:
        """Get current market price"""
        ticker = await self.client.get_symbol_ticker(symbol=SYMBOL)
        return float(ticker['price'])

    # Include previous database schema updates and other methods
    # ... [Include previous database initialization and other methods] ...

if __name__ == "__main__":
    bot = TradingBot()
    loop = asyncio.new_event_loop()
    
    # Signal handling and main loop similar to previous version
    # ... [Include previous signal handling and main loop code] ...