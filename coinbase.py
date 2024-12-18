import time
import requests
import hashlib
import base64
from ecdsa import SigningKey
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json

# -------- CONFIGURATION --------
api_key = "organizations/aee37f4c-d661-4b8e-8641-e1061cb74fde/apiKeys/20bb6d3a-554d-4c21-8c85-8ad578328267"
private_key = """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIJsmp4sI2Gv0Zmhc85kh3msrz0QXUwPqB+u4k4PLdZFJoAoGCCqGSM49
AwEHoUQDQgAETGEfAUtWBMzg6B7EfNzoWIrysfrPjLHNT+KSAhEIoDwUrc+rzLid
85cDxS9XUE/95x9DxFv3XLQNCLEoLhBnTg==
-----END EC PRIVATE KEY-----\n"""
api_url = "https://api.sandbox.coinbase.com"

symbol = "BTC-USD"  # Trading pair
interval = 60       # Candlestick granularity (1 minute)
lookback = 50       # Number of candles to look back
position_size_usd = 10.0  # Trade size in USD
stop_loss_percent = 2.0   # Stop-loss percentage
take_profit_percent = 3.0  # Take-profit percentage

# -------- SIGNATURE GENERATION --------
def generate_signature(timestamp, method, path, body=""):
    """Generates the signature for Coinbase Advanced Trade API."""
    message = f"{timestamp}{method}{path}{body}"
    print(f"DEBUG - Timestamp: {timestamp}")
    print(f"DEBUG - Message: {message}")

    signing_key = SigningKey.from_pem(private_key)
    signature = signing_key.sign_deterministic(
        message.encode('utf-8'), hashfunc=hashlib.sha256
    )
    encoded_signature = base64.b64encode(signature).decode()
    print(f"DEBUG - Signature: {encoded_signature}")
    return encoded_signature


# -------- FETCH ACCOUNT BALANCES --------
def fetch_accounts():
    method = "GET"
    path = "/api/v3/brokerage/accounts"
    timestamp = str(int(time.time()))
    signature = generate_signature(timestamp, method, path)

    headers = {
        "CB-ACCESS-KEY": api_key,
        "CB-ACCESS-TIMESTAMP": timestamp,
        "CB-ACCESS-SIGN": signature,
        "Content-Type": "application/json"
    }

    response = requests.get(api_url + path, headers=headers)
    if response.status_code == 200:
        print("Account Balances:")
        for account in response.json()['accounts']:
            print(f"Currency: {account['currency']}, Balance: {account['available_balance']['value']}")
    else:
        print("Error fetching accounts:", response.status_code, response.text)

# -------- FETCH HISTORICAL DATA --------
def fetch_historical_data():
    method = "GET"
    path = f"/api/v3/brokerage/products/{symbol}/candles?granularity={interval}"
    timestamp = str(int(time.time()))
    signature = generate_signature(timestamp, method, path)

    headers = {
        "CB-ACCESS-KEY": api_key,
        "CB-ACCESS-TIMESTAMP": timestamp,
        "CB-ACCESS-SIGN": signature,
        "Content-Type": "application/json"
    }

    response = requests.get(api_url + path, headers=headers)
    if response.status_code == 200:
        data = response.json()['candles']
        df = pd.DataFrame(data, columns=["start", "low", "high", "open", "close", "volume"])
        df["time"] = pd.to_datetime(df["start"], unit="s")
        df = df.sort_values("time")
        return df
    else:
        print("Error fetching historical data:", response.status_code, response.text)
        return pd.DataFrame()

# -------- ADD TECHNICAL INDICATORS --------
def add_indicators(df):
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    df['sma'] = SMAIndicator(df['close'], window=14).sma_indicator()
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df.dropna(inplace=True)
    return df

# -------- TRAIN MACHINE LEARNING MODEL --------
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
def place_market_order(side, funds):
    method = "POST"
    path = "/api/v3/brokerage/orders"
    body = json.dumps({
        "type": "MARKET",
        "side": side,
        "product_id": symbol,
        "quote_size": str(funds)
    })

    timestamp = str(int(time.time()))
    signature = generate_signature(timestamp, method, path, body)

    headers = {
        "CB-ACCESS-KEY": api_key,
        "CB-ACCESS-TIMESTAMP": timestamp,
        "CB-ACCESS-SIGN": signature,
        "Content-Type": "application/json"
    }

    response = requests.post(api_url + path, headers=headers, data=body)
    if response.status_code in [200, 201]:
        print(f"Order Executed: {side.capitalize()}, Details:", response.json())
    else:
        print("Error placing order:", response.status_code, response.text)

# -------- MAIN TRADING LOOP WITH RISK MANAGEMENT --------
def live_trading():
    print("Starting live trading...")
    df = fetch_historical_data()
    if df.empty:
        print("No data fetched, exiting...")
        return

    df = add_indicators(df)
    model = train_model(df)
    positions = []

    while True:
        try:
            print("Fetching latest data...")
            df = fetch_historical_data()
            df = add_indicators(df)

            features = df[['rsi', 'sma', 'macd', 'macd_signal']].iloc[-1:].values
            prediction = model.predict(features)[0]
            latest_close = df['close'].iloc[-1]

            if prediction == 1 and not positions:  # BUY signal
                stop_loss = latest_close * (1 - stop_loss_percent / 100)
                take_profit = latest_close * (1 + take_profit_percent / 100)
                print(f"BUY: {latest_close}, Stop-Loss: {stop_loss}, Take-Profit: {take_profit}")
                place_market_order("BUY", position_size_usd)
                positions.append({'entry_price': latest_close, 'stop_loss': stop_loss, 'take_profit': take_profit})

            # Risk management: check stop-loss and take-profit
            for pos in positions[:]:
                current_price = latest_close
                if current_price <= pos['stop_loss'] or current_price >= pos['take_profit']:
                    print(f"SELL at price {current_price} (Stop-Loss/Take-Profit triggered)")
                    place_market_order("SELL", position_size_usd)
                    positions.remove(pos)

            time.sleep(60)  # Wait for the next minute
        except Exception as e:
            print("Error during live trading:", str(e))
            time.sleep(60)

# -------- MAIN FUNCTION --------
if __name__ == "__main__":
    print("Coinbase Advanced Trade API Auto-Trading Bot with Risk Management")
    fetch_accounts()
    live_trading()
