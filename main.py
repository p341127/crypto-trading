import ccxt
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# -------- CONFIGURATION --------
api_key = 'YOUR_BINANCE_API_KEY'
api_secret = 'YOUR_BINANCE_SECRET_KEY'
symbol = 'BTC/USDT'
trade_amount = 0.001  # Amount to trade in BTC
interval = '1m'  # 1-minute timeframe
lookback = 50  # Number of candles for analysis
train_size = 1000  # Number of rows for training

# Initialize Binance Exchange
exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True
})


# Function to fetch OHLCV data
def fetch_data():
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=interval, limit=train_size + lookback)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df


# Add technical indicators
def add_indicators(df):
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    df['sma'] = SMAIndicator(df['close'], window=14).sma_indicator()
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df.dropna(inplace=True)  # Remove NaN values after adding indicators
    return df


# Train ML model
def train_model(df):
    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)  # 1 = price up, 0 = price down
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


# Live trading logic
def live_trading(model):
    while True:
        try:
            print("Fetching live data...")
            df = fetch_data()
            df = add_indicators(df)

            # Predict on latest row
            features = df[['rsi', 'sma', 'macd', 'macd_signal']].iloc[-1:].values
            prediction = model.predict(features)[0]

            # Execute buy/sell based on prediction
            if prediction == 1:  # Price expected to rise
                print("Prediction: BUY")
                order = exchange.create_market_buy_order(symbol, trade_amount)
                print("BUY Order Executed:", order)
            else:  # Price expected to fall
                print("Prediction: SELL")
                order = exchange.create_market_sell_order(symbol, trade_amount)
                print("SELL Order Executed:", order)

            time.sleep(60)  # Wait 1 minute before the next prediction
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(60)


# Main Function
def main():
    print("Starting Cryptocurrency Auto-Trading Bot...")
    df = fetch_data()
    df = add_indicators(df)
    model = train_model(df)
    live_trading(model)


if __name__ == "__main__":
    main()
