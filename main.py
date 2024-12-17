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
interval = '1m'
lookback = 50
train_size = 1000
simulation_mode = True  # Set to False for live trading
# Double represents percentage; eg. 2.0 = 2%
stop_loss_percent = 2.0  # Stop-loss percentage
take_profit_percent = 3.0  # Take-profit percentage
position_size_percent = 5.0  # Trade size as a percentage of balance

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
    df.dropna(inplace=True)
    return df

# Train ML model
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

# Risk management: Calculate position size
def calculate_position_size():
    balance = 1000 if simulation_mode else exchange.fetch_balance()['USDT']['free']
    position_size = (position_size_percent / 100) * balance
    return round(position_size, 6)

# Live trading logic with stop-loss and take-profit
def live_trading(model):
    positions = []
    while True:
        try:
            print("Fetching live data...")
            df = fetch_data()
            df = add_indicators(df)

            # Predict on latest row
            features = df[['rsi', 'sma', 'macd', 'macd_signal']].iloc[-1:].values
            prediction = model.predict(features)[0]
            latest_close = df['close'].iloc[-1]
            position_size = calculate_position_size()

            if prediction == 1:  # Price expected to rise
                print("Prediction: BUY")
                entry_price = latest_close
                stop_loss = entry_price * (1 - stop_loss_percent / 100)
                take_profit = entry_price * (1 + take_profit_percent / 100)

                if simulation_mode:
                    print(f"SIMULATION - BUY @ {entry_price}, Stop-Loss: {stop_loss}, Take-Profit: {take_profit}")
                else:
                    order = exchange.create_market_buy_order(symbol, position_size)
                    print("LIVE BUY Order Executed:", order)
                positions.append({'side': 'buy', 'entry_price': entry_price, 'stop_loss': stop_loss, 'take_profit': take_profit})

            # Check stop-loss/take-profit
            for pos in positions[:]:
                current_price = latest_close
                if pos['side'] == 'buy' and (current_price <= pos['stop_loss'] or current_price >= pos['take_profit']):
                    if simulation_mode:
                        print(f"SIMULATION - SELL @ {current_price} (Exit Trade: Stop-Loss/Take-Profit Hit)")
                    else:
                        order = exchange.create_market_sell_order(symbol, position_size)
                        print("LIVE SELL Order Executed:", order)
                    positions.remove(pos)

            time.sleep(60)  # Wait 1 minute before the next iteration

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(60)

# Main Function
def main():
    print("Starting Cryptocurrency Auto-Trading Bot with Risk Management...")
    df = fetch_data()
    df = add_indicators(df)
    model = train_model(df)
    live_trading(model)

if __name__ == "__main__":
    main()
