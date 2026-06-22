"""04_model_training.py"""
import numpy as np, pandas as pd, matplotlib, os, json, warnings, pickle
matplotlib.use("Agg"); warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

BASE_DIR=os.path.dirname(os.path.dirname(__file__))
PROC_DIR=os.path.join(BASE_DIR,"data","processed"); MODELS_DIR=os.path.join(BASE_DIR,"models"); PLOTS_DIR=os.path.join(BASE_DIR,"plots")
os.makedirs(MODELS_DIR,exist_ok=True); os.makedirs(PLOTS_DIR,exist_ok=True)
STOCKS=["AAPL","GOOGL","MSFT","AMZN"]
COLORS={"AAPL":"#6C63FF","GOOGL":"#FF6584","MSFT":"#43D39E","AMZN":"#FFB347"}
plt.rcParams.update({"figure.facecolor":"#0D1117","axes.facecolor":"#161B22","axes.edgecolor":"#30363D",
    "axes.labelcolor":"#C9D1D9","xtick.color":"#8B949E","ytick.color":"#8B949E","text.color":"#C9D1D9","grid.color":"#21262D"})

FCOLS=["Return","SMA_20","SMA_50","SMA_200","Volatility","BB_upper","BB_lower","RSI",
       "MACD","Signal","MACD_hist","ATR","Return_lag1","Return_lag2","Return_lag3","Return_lag5",
       "Roll_mean_5","Roll_mean_10","Roll_mean_20","Roll_std_5","Roll_std_10",
       "Price_to_SMA20","Price_to_SMA50","High_Low_Ratio","Close_Open_Ratio"]

def models():
    return {"Linear Regression":Pipeline([("s",StandardScaler()),("m",LinearRegression())]),
            "Ridge Regression":Pipeline([("s",StandardScaler()),("m",Ridge(alpha=1.0))]),
            "Random Forest":RandomForestRegressor(n_estimators=100,max_depth=8,random_state=42,n_jobs=-1),
            "Gradient Boosting":GradientBoostingRegressor(n_estimators=100,max_depth=4,learning_rate=0.1,random_state=42)}

all_res={}; summary={}
for ticker in STOCKS:
    df=pd.read_csv(os.path.join(PROC_DIR,f"{ticker}_features.csv"),parse_dates=["Date"])
    cols=[c for c in FCOLS if c in df.columns]
    X=df[cols].values; y=df["Target_Price"].values; dates=df["Date"].values
    sp=int(len(X)*0.8); Xtr,Xte=X[:sp],X[sp:]; ytr,yte=y[:sp],y[sp:]; dt=dates[sp:]
    res={}; best_r2=-np.inf; bp=None; bn=""
    for name,m in models().items():
        m.fit(Xtr,ytr); p=m.predict(Xte)
        mae=mean_absolute_error(yte,p); rmse=np.sqrt(mean_squared_error(yte,p))
        r2=r2_score(yte,p); mape=np.mean(np.abs((yte-p)/(yte+1e-9)))*100
        res[name]={"MAE":mae,"RMSE":rmse,"R2":r2,"MAPE":mape}
        if r2>best_r2: best_r2,bp,bn=r2,p,name
    all_res[ticker]=res; summary[ticker]={"best_model":bn,**res[bn]}
    print(f"  {ticker}: best={bn}, R2={best_r2:.4f}")
    # Save best model
    m2=list(models().values())[list(models().keys()).index(bn)]; m2.fit(Xtr,ytr)
    with open(os.path.join(MODELS_DIR,f"{ticker}_best.pkl"),"wb") as f: pickle.dump(m2,f)
    # Prediction plot
    col=COLORS[ticker]
    fig,(a1,a2)=plt.subplots(2,1,figsize=(14,9),gridspec_kw={"height_ratios":[3,1]}); fig.patch.set_facecolor("#0D1117"); plt.subplots_adjust(hspace=0.12)
    a1.plot(dt,yte,color="#C9D1D9",lw=1.5,label="Actual"); a1.plot(dt,bp,color=col,lw=2,linestyle="--",label=f"{bn} Pred")
    a1.fill_between(dt,yte,bp,alpha=0.15,color=col); a1.set_title(f"{ticker} Prediction vs Actual",fontsize=14,color="#E6EDF3")
    a1.set_ylabel("Price (USD)"); a1.legend(fontsize=10); a1.grid(True,alpha=0.3)
    a1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"${x:,.0f}"))
    err=bp-yte; a2.bar(range(len(err)),err,color=np.where(err>0,"#43D39E","#FF6584"),alpha=0.7,width=1)
    a2.axhline(0,color="#8B949E",lw=0.8); a2.set_ylabel("Residual"); a2.grid(True,alpha=0.3)
    plt.savefig(os.path.join(PLOTS_DIR,f"08_prediction_{ticker}.png"),dpi=150,bbox_inches="tight",facecolor="#0D1117"); plt.close()
    # Feature importance
    m3=list(models().values())[list(models().keys()).index(bn)]; m3.fit(Xtr,ytr)
    if hasattr(m3,"feature_importances_"):
        fi=m3.feature_importances_
    else:
        try: fi=np.abs(m3.named_steps["m"].coef_)
        except: fi=None
    if fi is not None:
        idx=np.argsort(fi)[-20:]
        fig,ax=plt.subplots(figsize=(10,7)); fig.patch.set_facecolor("#0D1117")
        ax.barh(range(len(idx)),fi[idx],color=plt.cm.plasma(np.linspace(0.2,0.9,len(idx))))
        ax.set_yticks(range(len(idx))); ax.set_yticklabels([cols[i] for i in idx],fontsize=9)
        ax.set_title(f"{ticker} Feature Importance",fontsize=13,color="#E6EDF3"); ax.grid(True,alpha=0.3,axis="x")
        plt.savefig(os.path.join(PLOTS_DIR,f"10_feature_importance_{ticker}.png"),dpi=150,bbox_inches="tight",facecolor="#0D1117"); plt.close()

# Model comparison
mn=list(models().keys()); pal=["#6C63FF","#FF6584","#43D39E","#FFB347"]
fig,axes=plt.subplots(1,3,figsize=(18,6)); fig.suptitle("Model Comparison",fontsize=14,color="#E6EDF3"); plt.subplots_adjust(wspace=0.35)
for ax,met in zip(axes,["MAE","RMSE","R2"]):
    vals=[np.mean([all_res[t][m][met] for t in STOCKS]) for m in mn]
    bars=ax.bar(range(4),vals,color=pal,alpha=0.8); ax.set_xticks(range(4)); ax.set_xticklabels([m.replace(" ","\n") for m in mn],fontsize=9)
    ax.set_title(met,color="#E6EDF3",fontsize=13,fontweight="bold"); ax.grid(True,alpha=0.3,axis="y")
    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,b.get_height()+max(vals)*0.01,f"{v:.3f}" if met=="R2" else f"{v:.2f}",ha="center",fontsize=9,color="#E6EDF3")
plt.savefig(os.path.join(PLOTS_DIR,"09_model_comparison.png"),dpi=150,bbox_inches="tight",facecolor="#0D1117"); plt.close()

with open(os.path.join(MODELS_DIR,"results.json"),"w") as f: json.dump({"all_results":all_res,"summary":summary},f,indent=2)
print("Model training complete.")
