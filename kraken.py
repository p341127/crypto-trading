import krakenex
from pykrakenapi import KrakenAPI
import pandas as pd
import time
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------- CONFIGURATION --------
api_key = "YOUR_KRAKEN_API_KEY"
api_secret = "YOUR_KRAKEN_API_SECRET"
pair = "XBTUSD"  # Trading pair: BTC/USD on Kraken
position_size = 10.0  # Trade size in USD
stop_loss_percent = 2.0
take_profit_percent = 3.0
lookback = 50

# -------- KRAKEN API SETUP --------
kraken = krakenex.API(key=api_key, secret=api_secret)
kapi = KrakenAPI(kraken)


# -------- FETCH ACCOUNT BALANCE --------
def get_balance():
    balance = kapi.get_account_balance()
    print("Account Balance:")
    print(balance)


# -------- FETCH HISTORICAL DATA --------
def fetch_ohlcv(pair, interval=1, lookback=lookback):
    ohlcv, _ = kapi.get_ohlc_data(pair, interval=interval, ascending=True)
    return ohlcv[-lookback:]


# -------- ADD TECHNICAL INDICATORS --------
def add_indicators(df):
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    df['sma'] = SMAIndicator(df['close'], window=14).sma_indicator()
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df.dropna(inplace=True)
    return df


# -------- TRAIN ML MODEL --------
def train_model(df):
    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    features = ['rsi', 'sma', 'macd', 'macd_signal']
    X = df[features]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    return model


# -------- PLACE MARKET ORDER --------
def place_market_order(side, volume):
    print(f"Placing {side} order for {volume} USD")
    if side == "buy":
        kraken.query_private('AddOrder', {'pair': pair, 'type': 'buy', 'ordertype': 'market', 'volume': volume})
    elif side == "sell":
        kraken.query_private('AddOrder', {'pair': pair, 'type': 'sell', 'ordertype': 'market', 'volume': volume})


# -------- LIVE TRADING WITH RISK MANAGEMENT --------
def live_trading():
    print("Starting live trading...")
    df = fetch_ohlcv(pair)
    df = add_indicators(df)
    model = train_model(df)

    positions = []

    while True:
        try:
            df = fetch_ohlcv(pair)
            df = add_indicators(df)

            features = df[['rsi', 'sma', 'macd', 'macd_signal']].iloc[-1:].values
            prediction = model.predict(features)[0]
            latest_close = df['close'].iloc[-1]

            # Place a BUY order if prediction is 1 and no open position
            if prediction == 1 and not positions:
                stop_loss = latest_close * (1 - stop_loss_percent / 100)
                take_profit = latest_close * (1 + take_profit_percent / 100)
                place_market_order("buy", position_size)
                positions.append({'entry_price': latest_close, 'stop_loss': stop_loss, 'take_profit': take_profit})

            # Check Stop-Loss/Take-Profit
            for pos in positions[:]:
                if latest_close <= pos['stop_loss'] or latest_close >= pos['take_profit']:
                    print(f"Exiting position at {latest_close} (SL/TP hit)")
                    place_market_order("sell", position_size)
                    positions.remove(pos)

            time.sleep(60)  # Wait 1 minute before the next check
        except Exception as e:
            print("Error during live trading:", e)
            time.sleep(60)


# -------- MAIN FUNCTION --------
if __name__ == "__main__":
    print("Kraken Auto-Trading Bot with Risk Management")
    get_balance()
    live_trading()
