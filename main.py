import krakenex
from pykrakenapi import KrakenAPI
import pandas as pd
import numpy as np
import time
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------- CONFIGURATION --------
api_key = "cxVZamOZkwu0dLVmHPAeea/TufQr4S+AL8DcNQSLYDThWqqYRdMQekFZ"  # Replace with your Kraken API Key
api_secret = "cnmV5W4/kS6ExUuOJ5G/lQQrhUedF3KrlUMPL27ifIFJ9Od9IGFIpZyaDCi4IoAPlS3Jllm+qjTzffmIQOK0qA=="  # Replace with your Kraken API Secret
pair = "XBTUSD"  # Trading pair: BTC/USD on Kraken
lookback = 50  # Number of candles to use
stop_loss_percent = 2.0  # Stop-loss percentage
take_profit_percent = 3.0  # Take-profit percentage
position_size = 10.0  # Simulated trade size in USD
start_balance = 1000.0  # Starting simulated balance

# -------- KRAKEN API SETUP --------
kraken = krakenex.API(key=api_key, secret=api_secret)
kapi = KrakenAPI(kraken)

# -------- FETCH LIVE OHLCV DATA --------
def fetch_live_ohlcv(pair, interval=1, lookback=lookback):
    """Fetch the latest OHLCV data."""
    ohlcv, _ = kapi.get_ohlc_data(pair, interval=interval, ascending=True)
    ohlcv = ohlcv[-lookback:]
    ohlcv.index.freq = 'min'  # Explicit frequency
    return ohlcv

# -------- ADD TECHNICAL INDICATORS --------
def add_indicators(df):
    """Add technical indicators to the dataframe."""
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    df['sma'] = SMAIndicator(df['close'], window=14).sma_indicator()
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df.dropna(inplace=True)
    return df

# -------- TRAIN MACHINE LEARNING MODEL --------
def train_model(df):
    """Train a Random Forest model to predict price direction."""
    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)  # 1 = Buy, 0 = Sell/Hold
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

# -------- SIMULATED LIVE TRADING --------
def simulated_live_trading():
    """Simulate live trading using live Kraken data."""
    print("Starting simulated live trading...")
    balance = start_balance
    positions = []

    # Fetch initial data and train model
    df = fetch_live_ohlcv(pair)
    df = add_indicators(df)
    model = train_model(df)

    try:
        while True:
            print("\nFetching latest market data...")
            df = fetch_live_ohlcv(pair)
            df = add_indicators(df)

            # Predict the next signal
            features = pd.DataFrame([df.iloc[-1][['rsi', 'sma', 'macd', 'macd_signal']].values],
                                    columns=['rsi', 'sma', 'macd', 'macd_signal'])
            prediction = model.predict(features)[0]
            latest_close = df['close'].iloc[-1]

            # Simulate buy logic
            if prediction == 1 and not positions:
                stop_loss = latest_close * (1 - stop_loss_percent / 100)
                take_profit = latest_close * (1 + take_profit_percent / 100)
                print(f"Simulated BUY at {latest_close:.2f}, Stop-Loss: {stop_loss:.2f}, Take-Profit: {take_profit:.2f}")
                positions.append({'entry_price': latest_close, 'stop_loss': stop_loss, 'take_profit': take_profit})

            # Simulate sell logic
            for pos in positions[:]:
                if latest_close <= pos['stop_loss'] or latest_close >= pos['take_profit']:
                    print(f"Simulated SELL at {latest_close:.2f} (SL/TP hit)")
                    profit = (latest_close - pos['entry_price']) if latest_close >= pos['take_profit'] else (pos['entry_price'] - latest_close)
                    balance += profit
                    print(f"Profit: {profit:.2f}, New Balance: {balance:.2f}")
                    positions.remove(pos)

            print(f"Current Balance: {balance:.2f}, Open Positions: {len(positions)}")
            time.sleep(60)  # Fetch new data every 1 minute

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
        print(f"Final Balance: {balance:.2f}")

# -------- MAIN FUNCTION --------
if __name__ == "__main__":
    print("Kraken Simulated Live Trading Bot")
    simulated_live_trading()
