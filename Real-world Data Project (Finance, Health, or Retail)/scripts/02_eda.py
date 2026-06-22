"""02_eda.py — EDA plots."""
import numpy as np, pandas as pd, matplotlib, os, warnings
matplotlib.use("Agg"); warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BASE_DIR  = os.path.dirname(os.path.dirname(__file__))
DATA_DIR  = os.path.join(BASE_DIR,"data","raw")
PLOTS_DIR = os.path.join(BASE_DIR,"plots")
PROC_DIR  = os.path.join(BASE_DIR,"data","processed")
os.makedirs(PLOTS_DIR, exist_ok=True); os.makedirs(PROC_DIR, exist_ok=True)
STOCKS = ["AAPL","GOOGL","MSFT","AMZN"]
COLORS = {"AAPL":"#6C63FF","GOOGL":"#FF6584","MSFT":"#43D39E","AMZN":"#FFB347"}
plt.rcParams.update({"figure.facecolor":"#0D1117","axes.facecolor":"#161B22",
    "axes.edgecolor":"#30363D","axes.labelcolor":"#C9D1D9","xtick.color":"#8B949E",
    "ytick.color":"#8B949E","text.color":"#C9D1D9","grid.color":"#21262D","grid.linewidth":0.8})

def load(ticker):
    df = pd.read_csv(os.path.join(DATA_DIR,f"{ticker}.csv"), parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["Return"] = df["Close"].pct_change()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_200"]= df["Close"].rolling(200).mean()
    df["Volatility"] = df["Return"].rolling(20).std()*np.sqrt(252)
    std20=df["Close"].rolling(20).std()
    df["BB_upper"]=df["SMA_20"]+2*std20; df["BB_lower"]=df["SMA_20"]-2*std20
    delta=df["Close"].diff(); gain=delta.clip(lower=0).rolling(14).mean()
    loss=(-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"]=100-(100/(1+gain/(loss+1e-9)))
    return df

data = {t: load(t) for t in STOCKS}
for t,df in data.items(): print(f"  {t}: {len(df)} rows")

# Plot 1 — price history
fig,axes=plt.subplots(2,2,figsize=(16,10)); fig.suptitle("Stock Price History",fontsize=16,color="#E6EDF3",y=1.01)
plt.subplots_adjust(hspace=0.4,wspace=0.35)
for ax,t in zip(axes.flatten(),STOCKS):
    df=data[t]; col=COLORS[t]
    ax.fill_between(df["Date"],df["Close"],alpha=0.1,color=col)
    ax.plot(df["Date"],df["Close"],color=col,lw=1.5,label="Close")
    ax.plot(df["Date"],df["SMA_20"],color="#FFF",lw=0.8,linestyle="--",label="SMA20",alpha=0.7)
    ax.plot(df["Date"],df["SMA_50"],color="#FFD700",lw=0.8,linestyle="--",label="SMA50",alpha=0.7)
    ax.plot(df["Date"],df["SMA_200"],color="#FF4560",lw=1.0,linestyle="-.",label="SMA200",alpha=0.9)
    ax.set_title(t,color=col,fontweight="bold"); ax.grid(True,alpha=0.3)
    ax.legend(fontsize=8); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"${x:,.0f}"))
plt.savefig(os.path.join(PLOTS_DIR,"01_price_history.png"),dpi=150,bbox_inches="tight",facecolor="#0D1117"); plt.close()
print("  Plot 1 saved")

# Plot 2 — normalised returns
fig,ax=plt.subplots(figsize=(14,6)); fig.patch.set_facecolor("#0D1117")
for t,df in data.items():
    ax.plot(df["Date"],(df["Close"]/df["Close"].iloc[0])*100,color=COLORS[t],lw=2,label=t)
ax.axhline(100,color="#30363D",lw=1,linestyle="--"); ax.set_title("Normalised Returns (Base=100)",fontsize=14,color="#E6EDF3")
ax.legend(fontsize=11); ax.grid(True,alpha=0.3)
plt.savefig(os.path.join(PLOTS_DIR,"02_normalized_returns.png"),dpi=150,bbox_inches="tight",facecolor="#0D1117"); plt.close()
print("  Plot 2 saved")

# Plot 3 — correlation heatmap
returns=pd.DataFrame({t:d["Return"] for t,d in data.items()}).dropna(); corr=returns.corr()
fig,ax=plt.subplots(figsize=(7,6)); fig.patch.set_facecolor("#0D1117")
cmap=LinearSegmentedColormap.from_list("f",["#FF6584","#161B22","#43D39E"])
im=ax.imshow(corr.values,cmap=cmap,vmin=-1,vmax=1)
ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_xticklabels(STOCKS); ax.set_yticklabels(STOCKS)
for i in range(4):
    for j in range(4): ax.text(j,i,f"{corr.values[i,j]:.2f}",ha="center",va="center",fontsize=11,color="#E6EDF3",fontweight="bold")
plt.colorbar(im,ax=ax,shrink=0.85); ax.set_title("Returns Correlation",fontsize=13,color="#E6EDF3")
plt.savefig(os.path.join(PLOTS_DIR,"03_correlation_heatmap.png"),dpi=150,bbox_inches="tight",facecolor="#0D1117"); plt.close()
print("  Plot 3 saved")

# Plot 4 — returns distribution
fig,axes=plt.subplots(1,4,figsize=(18,5)); fig.suptitle("Daily Returns Distribution",fontsize=15,color="#E6EDF3")
for ax,t in zip(axes,STOCKS):
    r=(data[t]["Return"].dropna()*100); n,e=np.histogram(r,60); c=(e[:-1]+e[1:])/2
    ax.bar(c,n,width=(e[1]-e[0]),color=COLORS[t],alpha=0.75)
    mu,std=r.mean(),r.std()
    ax.axvline(mu,color="#FFF",lw=1.5,linestyle="--"); ax.set_title(t,color=COLORS[t],fontweight="bold"); ax.grid(True,alpha=0.3)
plt.savefig(os.path.join(PLOTS_DIR,"04_returns_distribution.png"),dpi=150,bbox_inches="tight",facecolor="#0D1117"); plt.close()
print("  Plot 4 saved")

# Plot 5 — Bollinger + RSI
df=data["AAPL"].iloc[-252:]
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(14,9),sharex=True,gridspec_kw={"height_ratios":[3,1]}); fig.patch.set_facecolor("#0D1117"); plt.subplots_adjust(hspace=0.08)
ax1.fill_between(df["Date"],df["BB_lower"],df["BB_upper"],alpha=0.15,color="#6C63FF")
ax1.plot(df["Date"],df["Close"],color="#6C63FF",lw=2,label="AAPL Close")
ax1.plot(df["Date"],df["SMA_20"],color="#FFD700",lw=1.2,linestyle="--",label="SMA 20")
ax1.plot(df["Date"],df["BB_upper"],color="#43D39E",lw=0.8,linestyle="-.",label="Upper Band")
ax1.plot(df["Date"],df["BB_lower"],color="#FF6584",lw=0.8,linestyle="-.",label="Lower Band")
ax1.set_title("AAPL Bollinger Bands & RSI (Last 12 Months)",fontsize=14,color="#E6EDF3"); ax1.legend(fontsize=9); ax1.grid(True,alpha=0.3)
ax2.plot(df["Date"],df["RSI"],color="#FF6584",lw=1.5)
ax2.axhline(70,color="#FF4560",lw=1,linestyle="--"); ax2.axhline(30,color="#43D39E",lw=1,linestyle="--")
ax2.fill_between(df["Date"],df["RSI"],70,where=df["RSI"]>=70,alpha=0.25,color="#FF4560")
ax2.fill_between(df["Date"],df["RSI"],30,where=df["RSI"]<=30,alpha=0.25,color="#43D39E")
ax2.set_ylabel("RSI"); ax2.set_ylim(0,100); ax2.grid(True,alpha=0.3)
plt.savefig(os.path.join(PLOTS_DIR,"05_bollinger_rsi.png"),dpi=150,bbox_inches="tight",facecolor="#0D1117"); plt.close()
print("  Plot 5 saved")

# Plot 6 — volume
fig,axes=plt.subplots(2,2,figsize=(16,10)); fig.suptitle("Volume Analysis",fontsize=15,color="#E6EDF3",y=1.01)
plt.subplots_adjust(hspace=0.4,wspace=0.35)
for ax,t in zip(axes.flatten(),STOCKS):
    df=data[t]; vm=df["Volume"].rolling(30).mean()
    ax.bar(df["Date"],df["Volume"]/1e6,color=COLORS[t],alpha=0.35,width=1)
    ax.plot(df["Date"],vm/1e6,color="#FFF",lw=1.5); ax.set_title(t,color=COLORS[t],fontweight="bold"); ax.grid(True,alpha=0.3)
plt.savefig(os.path.join(PLOTS_DIR,"06_volume_analysis.png"),dpi=150,bbox_inches="tight",facecolor="#0D1117"); plt.close()
print("  Plot 6 saved")

# Plot 7 — volatility
fig,ax=plt.subplots(figsize=(14,6)); fig.patch.set_facecolor("#0D1117")
for t,df in data.items(): ax.plot(df["Date"],df["Volatility"]*100,color=COLORS[t],lw=1.5,label=t)
ax.set_title("Rolling 20-Day Annualised Volatility",fontsize=14,color="#E6EDF3"); ax.legend(fontsize=11); ax.grid(True,alpha=0.3)
plt.savefig(os.path.join(PLOTS_DIR,"07_volatility.png"),dpi=150,bbox_inches="tight",facecolor="#0D1117"); plt.close()
print("  Plot 7 saved")

# Save processed
for t,df in data.items(): df.to_csv(os.path.join(PROC_DIR,f"{t}_processed.csv"),index=False)
print("EDA complete.")
