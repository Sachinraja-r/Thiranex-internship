"""01_generate_data.py — Generate OHLCV stock data via Geometric Brownian Motion."""
import numpy as np, pandas as pd, os
from datetime import datetime

SEED = 42; np.random.seed(SEED)
STOCKS = {
    "AAPL":  {"S0": 150.0,  "mu": 0.00045, "sigma": 0.015},
    "GOOGL": {"S0": 2800.0, "mu": 0.00040, "sigma": 0.016},
    "MSFT":  {"S0": 290.0,  "mu": 0.00050, "sigma": 0.014},
    "AMZN":  {"S0": 3400.0, "mu": 0.00035, "sigma": 0.018},
}
NAMES = {"AAPL":"Apple Inc.","GOOGL":"Alphabet Inc.","MSFT":"Microsoft Corp.","AMZN":"Amazon.com Inc."}
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

def gbm(S0, mu, sigma, n):
    W = np.random.standard_normal(n)
    return S0 * np.exp(np.cumsum((mu - 0.5*sigma**2) + sigma*W))

def make_ohlcv(ticker, params, dates):
    n      = len(dates)
    closes = gbm(params["S0"], params["mu"], params["sigma"], n)
    dv     = np.random.uniform(0.005, 0.025, n)
    highs  = closes*(1+dv); lows = closes*(1-dv)
    opens  = closes*(1+np.random.uniform(-0.01,0.01,n))
    highs  = np.maximum(highs, np.maximum(opens,closes))
    lows   = np.minimum(lows,  np.minimum(opens,closes))
    bvol   = {"AAPL":80e6,"GOOGL":1.2e6,"MSFT":25e6,"AMZN":4e6}
    vol    = (bvol[ticker]*np.random.lognormal(0,0.4,n)).astype(int)
    return pd.DataFrame({"Date":dates,"Ticker":ticker,
        "Open":opens.round(2),"High":highs.round(2),"Low":lows.round(2),
        "Close":closes.round(2),"Volume":vol,"Adj_Close":closes.round(2)})

dates = pd.bdate_range("2019-01-02","2024-01-01")
print(f"Trading days: {len(dates)}")
frames = []
for ticker, p in STOCKS.items():
    df = make_ohlcv(ticker, p, dates)
    df.to_csv(os.path.join(RAW_DIR, f"{ticker}.csv"), index=False)
    frames.append(df)
    print(f"  {ticker}: ${df['Close'].iloc[0]:.2f} -> ${df['Close'].iloc[-1]:.2f}")
pd.concat(frames).to_csv(os.path.join(RAW_DIR,"all_stocks.csv"), index=False)
print("Done.")
