"""03_feature_engineering.py"""
import numpy as np, pandas as pd, os, warnings; warnings.filterwarnings("ignore")
BASE_DIR=os.path.dirname(os.path.dirname(__file__))
PROC_DIR=os.path.join(BASE_DIR,"data","processed"); os.makedirs(PROC_DIR,exist_ok=True)
STOCKS=["AAPL","GOOGL","MSFT","AMZN"]

def engineer(ticker):
    df=pd.read_csv(os.path.join(PROC_DIR,f"{ticker}_processed.csv"),parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    e12=df["Close"].ewm(span=12,adjust=False).mean(); e26=df["Close"].ewm(span=26,adjust=False).mean()
    df["MACD"]=e12-e26; df["Signal"]=df["MACD"].ewm(span=9,adjust=False).mean(); df["MACD_hist"]=df["MACD"]-df["Signal"]
    hl=(df["High"]-df["Low"]); hc=(df["High"]-df["Close"].shift()).abs(); lc=(df["Low"]-df["Close"].shift()).abs()
    df["ATR"]=pd.concat([hl,hc,lc],axis=1).max(axis=1).rolling(14).mean()
    for lag in [1,2,3,5,10]:
        df[f"Return_lag{lag}"]=df["Return"].shift(lag); df[f"Close_lag{lag}"]=df["Close"].shift(lag)
    for w in [5,10,20,50]:
        df[f"Roll_mean_{w}"]=df["Close"].rolling(w).mean(); df[f"Roll_std_{w}"]=df["Close"].rolling(w).std()
        df[f"Roll_vol_{w}"]=df["Return"].rolling(w).std()*np.sqrt(252)
        df[f"Roll_max_{w}"]=df["Close"].rolling(w).max(); df[f"Roll_min_{w}"]=df["Close"].rolling(w).min()
    df["DayOfWeek"]=df["Date"].dt.dayofweek; df["Month"]=df["Date"].dt.month; df["Quarter"]=df["Date"].dt.quarter
    df["Price_to_SMA20"]=df["Close"]/(df["SMA_20"]+1e-9); df["Price_to_SMA50"]=df["Close"]/(df["SMA_50"]+1e-9)
    df["High_Low_Ratio"]=(df["High"]-df["Low"])/(df["Close"]+1e-9); df["Close_Open_Ratio"]=(df["Close"]-df["Open"])/(df["Open"]+1e-9)
    df["Target_Price"]=df["Close"].shift(-1); df["Target_Return"]=df["Close"].shift(-1)/df["Close"]-1
    df.dropna(inplace=True); df.reset_index(drop=True,inplace=True)
    return df

for t in STOCKS:
    df=engineer(t); df.to_csv(os.path.join(PROC_DIR,f"{t}_features.csv"),index=False)
    print(f"  {t}: {len(df)} rows x {df.shape[1]} cols")
print("Feature engineering complete.")
