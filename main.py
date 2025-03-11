import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import websocket
import json
import threading
import time
from datetime import datetime, time as dt_time
import logging
from telegram import Bot, Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from hurst import compute_Hc
import requests
import pytz
import zlib
import shap

logging.basicConfig(level=logging.INFO)

# --- CONFIGURATION ---
SYMBOL = "BTC/USDT"
TIMEFRAME = "5m"
POSITION_SIZE = 0.01
MEXC_API_KEY = ""
MEXC_SECRET = ""
BYBIT_API_KEY = "YOUR_BYBIT_KEY"
BYBIT_SECRET = "YOUR_BYBIT_SECRET"
TELEGRAM_BOT_TOKEN = ""

# Initialize exchanges
mexc = ccxt.mexc({
    'apiKey': MEXC_API_KEY,
    'secret': MEXC_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'}
})

bybit = ccxt.bybit({
    'apiKey': BYBIT_API_KEY,
    'secret': BYBIT_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'}
})

# --- REAL-TIME DATA HANDLER (MEXC) ---
class RealTimeData:
    def __init__(self, symbol, timeframe):
        self.symbol = symbol
        self.timeframe = timeframe
        self.data = []
        self.order_book = {'bids': [], 'asks': []}
        self.ws = None

    def start(self):
        def on_message(ws, message):
            decompressed = zlib.decompress(message).decode('utf-8')
            data = json.loads(decompressed)
            
            if 'k' in data and data['k']['x']:
                self.data.append({
                    'timestamp': data['k']['t'],
                    'open': float(data['k']['o']),
                    'high': float(data['k']['h']),
                    'low': float(data['k']['l']),
                    'close': float(data['k']['c']),
                    'volume': float(data['k']['v'])
                })
                if len(self.data) > 200:
                    self.data.pop(0)
                logging.info(f"New candle received: {data['k']}")
            
            if 'bids' in data and 'asks' in data:
                self.order_book['bids'] = [[float(b[0]), float(b[1])] for b in data['bids'][:20]]
                self.order_book['asks'] = [[float(a[0]), float(a[1])] for a in data['asks'][:20]]
                logging.info("Order book updated.")
        
        self.ws = websocket.WebSocketApp(
            f"wss://wbs.mexc.com/ws?symbol={self.symbol.lower().replace('/', '')}_swap&interval={self.timeframe}",
            on_message=on_message
        )
        threading.Thread(target=self.ws.run_forever).start()

# --- MACHINE LEARNING MODELS ---
def train_models(df):
    FEATURES = ['close', 'volume', 'high', 'low']
    X = df[FEATURES]
    y = np.where(df['close'].shift(-5) / df['close'] - 1 > 0.003, 1, -1)
    
    # Random Forest
    model_rf = RandomForestClassifier(n_estimators=300, max_depth=12)
    model_rf.fit(X, y)
    logging.info("Random Forest model trained.")
    
    # LSTM Model (PyTorch Implementation)
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return torch.sigmoid(out)
    
    model_lstm = LSTMModel(input_size=len(FEATURES), hidden_size=150, num_layers=2, output_size=1)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model_lstm.parameters(), lr=0.001)
    
    X_tensor = torch.tensor([X.values[i:i+20] for i in range(len(X)-20)], dtype=torch.float32)
    y_tensor = torch.tensor(y[20:], dtype=torch.float32).unsqueeze(1)
    
    for epoch in range(25):
        model_lstm.train()
        optimizer.zero_grad()
        outputs = model_lstm(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        logging.info(f"LSTM Epoch [{epoch+1}/25], Loss: {loss.item():.4f}")
    
    logging.info("LSTM model trained.")
    return model_rf, model_lstm

# --- AUTO-RETRAINING ---
def auto_retrain():
    while True:
        colombo_tz = pytz.timezone('Asia/Colombo')
        now = datetime.now(colombo_tz)
        next_retrain = (now.replace(hour=0, minute=0, second=0, microsecond=0) + 
                        pd.Timedelta(days=1))
        time.sleep((next_retrain - now).total_seconds())
        
        df = pd.DataFrame(mexc.fetch_ohlcv(SYMBOL, TIMEFRAME))
        with threading.Lock():
            global model_rf, model_lstm
            model_rf, model_lstm = train_models(df)
        logging.info("Models retrained with fresh data")

# --- TELEGRAM BOT COMMAND HANDLERS ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    await context.bot.send_message(chat_id=chat_id, text="Bot started successfully!")
    logging.info(f"Bot started in chat {chat_id}")

async def send_update(context: ContextTypes.DEFAULT_TYPE, chat_id, message):
    await context.bot.send_message(chat_id=chat_id, text=message)

# --- RUN ---
if __name__ == "__main__":
    # Start auto-retraining in a separate thread
    threading.Thread(target=auto_retrain, daemon=True).start()

    # Initialize Telegram bot
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Add command handlers
    application.add_handler(CommandHandler("start", start))

    # Start the bot
    application.run_polling()
