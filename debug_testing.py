import time
import requests
import hashlib
import base64
from ecdsa import SigningKey

api_key = "organizations/aee37f4c-d661-4b8e-8641-e1061cb74fde/apiKeys/bddeea87-7018-4715-b72b-420ff7c4595b"
private_key = """-----BEGIN EC PRIVATE KEY-----\nMHcCAQEEII3xYttxCSEhXOR0IJL0CwNnwC74qc5ZgLEZ7YBLXVuMoAoGCCqGSM49\nAwEHoUQDQgAEz/I1fLhrP29/aTaaze1ZmgG+t6BW2EHR3OjujAM/zmHdgEixAi3G\nHNF9hKSp7t0GDY/ggvweXwYTFX4ux3YC6A==\n-----END EC PRIVATE KEY-----\n"""
api_url = "https://api.sandbox.coinbase.com"

def generate_signature(timestamp, method, path, body=""):
    message = f"{timestamp}{method}{path}{body}".encode('utf-8')
    signing_key = SigningKey.from_pem(private_key)
    signature = signing_key.sign_deterministic(message, hashfunc=hashlib.sha256)
    return base64.b64encode(signature).decode()

def test_connection():
    method = "GET"
    path = "/api/v3/brokerage/accounts"
    body = ""
    timestamp = str(int(time.time()))
    signature = generate_signature(timestamp, method, path, body)

    headers = {
        "CB-ACCESS-KEY": api_key,
        "CB-ACCESS-TIMESTAMP": timestamp,
        "CB-ACCESS-SIGN": signature,
        "Content-Type": "application/json"
    }

    response = requests.get(api_url + path, headers=headers)
    print("Response Code:", response.status_code)
    print("Response Text:", response.text)

test_connection()
