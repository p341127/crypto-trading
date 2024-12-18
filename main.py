import krakenex
from pykrakenapi import KrakenAPI
import pandas as pd
import numpy as np
import time
import json
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
from datetime import datetime

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*'T' is deprecated.*")


# -------- CONFIGURATION --------
api_key = "cxVZamOZkwu0dLVmHPAeea/TufQr4S+AL8DcNQSLYDThWqqYRdMQekFZ"  # Replace with your Kraken API Key
api_secret = "cnmV5W4/kS6ExUuOJ5G/lQQrhUedF3KrlUMPL27ifIFJ9Od9IGFIpZyaDCi4IoAPlS3Jllm+qjTzffmIQOK0qA=="  # Replace with your Kraken API Secret
pair = "XBTUSD"  # Trading pair: BTC/USD on Kraken
lookback = 50  # Number of candles to use
stop_loss_percent = 2.0  # Stop-loss percentage
take_profit_percent = 3.0  # Take-profit percentage
position_size = 10.0  # Simulated trade size in USD
log_file = "trading_log.json"
trade_history_file = "trade_history.csv"

# -------- KRAKEN API SETUP --------
kraken = krakenex.API(key=api_key, secret=api_secret)
kapi = KrakenAPI(kraken)

# -------- LOAD PROGRESS --------
def load_progress():
    if os.path.exists(log_file):
        with open(log_file, "r") as file:
            return json.load(file)
    else:
        return {"balance": 1000.0, "positions": []}

# -------- SAVE PROGRESS --------
def save_progress(balance, positions):
    with open(log_file, "w") as file:
        json.dump({"balance": balance, "positions": positions}, file, indent=4)

# -------- LOG TRADE HISTORY --------
def log_trade(trade_type, entry_price, exit_price, stop_loss, take_profit, profit, balance):
    trade_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trade_type": trade_type,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "profit": profit,
        "balance": balance
    }
    file_exists = os.path.exists(trade_history_file)
    with open(trade_history_file, "a") as file:
        headers = "timestamp,trade_type,entry_price,exit_price,stop_loss,take_profit,profit,balance\n"
        if not file_exists:
            file.write(headers)
        file.write(",".join(str(trade_data[key]) for key in trade_data) + "\n")

# -------- FETCH LIVE OHLCV DATA --------
def fetch_live_ohlcv(pair, interval=1, lookback=lookback):
    ohlcv, _ = kapi.get_ohlc_data(pair, interval=interval, ascending=True)
    return ohlcv[-lookback:]

# -------- ADD INDICATORS --------
def add_indicators(df):
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    df['sma'] = SMAIndicator(df['close'], window=14).sma_indicator()
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df.dropna(inplace=True)
    return df

# -------- MACHINE LEARNING MODEL --------
def train_model(df):
    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    features = ['rsi', 'sma', 'macd', 'macd_signal']
    X = df[features]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

# -------- VISUAL PRINTING SYSTEM --------
def display_status(latest_close, prev_close, balance, positions):
    change = ((latest_close - prev_close) / prev_close) * 100
    direction = "📈" if change > 0 else "📉"
    print(f"\n{'='*50}")
    print(f" Latest Price: ${latest_close:.2f} ({direction} {change:.2f}%)")
    print(f" Current Balance: ${balance:.2f}")

    if positions:
        for pos in positions:
            print(f" Open Position: ${pos['entry_price']:.2f}")
            print(f"   Stop-Loss (Floor): ${pos['stop_loss']:.2f}")
            print(f"   Take-Profit (Ceiling): ${pos['take_profit']:.2f}")
    else:
        print(" No open positions.")
    print(f"{'='*50}")

# -------- SIMULATED TRADING --------
def simulated_live_trading():
    progress = load_progress()
    balance = progress["balance"]
    positions = progress["positions"]

    df = fetch_live_ohlcv(pair)
    df = add_indicators(df)
    model = train_model(df)

    prev_close = df['close'].iloc[-1]

    try:
        while True:
            df = fetch_live_ohlcv(pair)
            df = add_indicators(df)
            latest_close = df['close'].iloc[-1]

            features = pd.DataFrame([df.iloc[-1][['rsi', 'sma', 'macd', 'macd_signal']].values],
                                    columns=['rsi', 'sma', 'macd', 'macd_signal'])
            prediction = model.predict(features)[0]

            if prediction == 1 and not positions:
                stop_loss = latest_close * (1 - stop_loss_percent / 100)
                take_profit = latest_close * (1 + take_profit_percent / 100)
                positions.append({"entry_price": latest_close, "stop_loss": stop_loss, "take_profit": take_profit})
                print(f"🟢 Simulated BUY at ${latest_close:.2f}")

            for pos in positions[:]:
                if latest_close <= pos['stop_loss'] or latest_close >= pos['take_profit']:
                    profit = (latest_close - pos['entry_price']) if latest_close >= pos['take_profit'] else (pos['entry_price'] - latest_close)
                    balance += profit
                    log_trade("SELL", pos['entry_price'], latest_close, pos['stop_loss'], pos['take_profit'], profit, balance)
                    print(f"🔴 Simulated SELL at ${latest_close:.2f}, Profit: ${profit:.2f}")
                    positions.remove(pos)

            display_status(latest_close, prev_close, balance, positions)
            save_progress(balance, positions)
            prev_close = latest_close
            time.sleep(20)

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
        save_progress(balance, positions)

# -------- MAIN --------
if __name__ == "__main__":
    print("Kraken Simulated Trading Bot with Real-Time UI")
    simulated_live_trading()