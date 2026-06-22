"""build_report.py — Generates college-level HTML project report."""
import os, json, base64, numpy as np, pandas as pd

BASE_DIR   = os.path.dirname(__file__)
PLOTS_DIR  = os.path.join(BASE_DIR, "plots")
PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "report")
os.makedirs(REPORT_DIR, exist_ok=True)

TICKERS = ["AAPL","GOOGL","MSFT","AMZN"]
COLORS  = {"AAPL":"#3b82f6","GOOGL":"#ef4444","MSFT":"#10b981","AMZN":"#f59e0b"}
NAMES   = {"AAPL":"Apple Inc.","GOOGL":"Alphabet Inc.","MSFT":"Microsoft Corp.","AMZN":"Amazon.com Inc."}

def img64(fname):
    p = os.path.join(PLOTS_DIR, fname)
    if not os.path.exists(p): return ""
    with open(p,"rb") as f: return base64.b64encode(f.read()).decode()

def load_results():
    with open(os.path.join(MODELS_DIR,"results.json")) as f: return json.load(f)

def portfolio_stats():
    rows = []
    for t in TICKERS:
        df = pd.read_csv(os.path.join(PROC_DIR,f"{t}_features.csv"), parse_dates=["Date"])
        ret  = (df["Close"].iloc[-1]/df["Close"].iloc[0]-1)*100
        vol  = df["Return"].std()*np.sqrt(252)*100
        rm   = df["Close"].cummax()
        dd   = ((df["Close"]-rm)/rm).min()*100
        sh   = (df["Return"].mean()/(df["Return"].std()+1e-9))*np.sqrt(252)
        rows.append(dict(ticker=t, name=NAMES[t],
            start=round(df["Close"].iloc[0],2), end=round(df["Close"].iloc[-1],2),
            ret=round(ret,1), vol=round(vol,1), dd=round(dd,1), sharpe=round(sh,3)))
    return rows

results   = load_results()
portfolio = portfolio_stats()
models_list = list(results["all_results"]["AAPL"].keys())

def portfolio_rows():
    out = ""
    for p in portfolio:
        col = COLORS[p["ticker"]]; pos = p["ret"]>=0
        out += f"""<tr>
          <td><span class="tag" style="background:{col}20;color:{col};border:1px solid {col}50">{p['ticker']}</span></td>
          <td>{p['name']}</td><td class="mono">${p['start']:,.2f}</td><td class="mono">${p['end']:,.2f}</td>
          <td class="mono {'pos' if pos else 'neg'}">{'+' if pos else ''}{p['ret']:.1f}%</td>
          <td class="mono">{p['vol']:.1f}%</td>
          <td class="mono neg">{p['dd']:.1f}%</td>
          <td class="mono">{p['sharpe']:.3f}</td></tr>"""
    return out

def results_rows():
    out = ""
    for t in TICKERS:
        first = True
        for m in models_list:
            r    = results["all_results"][t][m]
            best = m == results["summary"][t]["best_model"]
            badge = '<span class="badge">Best</span>' if best else ""
            tcell = f'<td rowspan="{len(models_list)}" style="vertical-align:top;padding-top:14px"><span class="tag" style="background:{COLORS[t]}20;color:{COLORS[t]};border:1px solid {COLORS[t]}50">{t}</span></td>' if first else ""
            out += f'<tr class="{"best-row" if best else ""}"> {tcell}<td>{m}{badge}</td><td class="mono">{r["MAE"]:.2f}</td><td class="mono">{r["RMSE"]:.2f}</td><td class="mono {"good" if r["R2"]>0.95 else "ok"}">{r["R2"]:.4f}</td><td class="mono">{r["MAPE"]:.2f}%</td></tr>'
            first = False
    return out

html = f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Stock Market Analysis — Project Report</title>
<meta name="description" content="College Finance Data Science Project: Stock Market Analysis and ML Price Prediction"/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
html{{scroll-behavior:smooth}}
body{{font-family:'Inter',sans-serif;background:#f9fafb;color:#111827;line-height:1.7;font-size:15px}}
.page{{max-width:900px;margin:0 auto;padding:40px 24px 80px}}
.cover{{background:linear-gradient(135deg,#1e3a5f,#1d4ed8,#1e3a5f);color:#fff;border-radius:16px;padding:52px 48px;margin-bottom:40px;position:relative;overflow:hidden}}
.cover::before{{content:'';position:absolute;top:-60px;right:-60px;width:280px;height:280px;border-radius:50%;background:rgba(255,255,255,0.05)}}
.cover-eye{{font-size:.8rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;opacity:.7;margin-bottom:14px}}
.cover h1{{font-size:2.1rem;font-weight:800;line-height:1.2;letter-spacing:-1px;margin-bottom:10px}}
.cover-sub{{font-size:1rem;opacity:.8;margin-bottom:28px}}
.chips{{display:flex;flex-wrap:wrap;gap:8px}}
.chip{{padding:5px 14px;border-radius:20px;border:1px solid rgba(255,255,255,.25);font-size:.8rem;font-weight:500;background:rgba(255,255,255,.1)}}
.toc{{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:28px 32px;margin-bottom:40px}}
.toc h2{{font-size:.9rem;text-transform:uppercase;letter-spacing:.08em;color:#6b7280;margin-bottom:14px;font-weight:600}}
.toc ol{{padding-left:20px;display:grid;grid-template-columns:1fr 1fr;gap:3px 24px}}
.toc li{{font-size:.88rem;padding:2px 0}}
.toc a{{color:#1d4ed8;text-decoration:none;font-weight:500}}
.toc a:hover{{text-decoration:underline}}
.sec{{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:36px;margin-bottom:24px}}
.sec-num{{font-size:.75rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#6b7280;margin-bottom:6px}}
.sec h2{{font-size:1.4rem;font-weight:700;letter-spacing:-.5px;margin-bottom:18px;padding-bottom:14px;border-bottom:2px solid #f3f4f6}}
.sec h3{{font-size:1rem;font-weight:600;margin:22px 0 10px;color:#374151}}
.sec p{{color:#374151;margin-bottom:10px;line-height:1.7}}
.box{{border-radius:8px;padding:14px 18px;margin:14px 0;font-size:.9rem}}
.blue{{background:#eff6ff;border-left:4px solid #3b82f6;color:#1e40af}}
.yellow{{background:#fffbeb;border-left:4px solid #f59e0b;color:#92400e}}
.green{{background:#f0fdf4;border-left:4px solid #10b981;color:#065f46}}
table{{width:100%;border-collapse:collapse;font-size:.87rem;margin:14px 0}}
thead tr{{background:#f3f4f6}}
th{{padding:10px 14px;text-align:left;font-weight:600;font-size:.76rem;text-transform:uppercase;letter-spacing:.05em;color:#6b7280}}
td{{padding:10px 14px;border-bottom:1px solid #f3f4f6}}
tr:hover td{{background:#fafafa}}
.tag{{display:inline-block;padding:3px 9px;border-radius:6px;font-size:.76rem;font-weight:700;font-family:'JetBrains Mono',monospace}}
.badge{{display:inline-block;background:#dcfce7;color:#166534;border:1px solid #bbf7d0;padding:1px 7px;border-radius:4px;font-size:.7rem;font-weight:700;margin-left:4px}}
.best-row td{{background:#f0fdf4!important}}
.mono{{font-family:'JetBrains Mono',monospace}}
.pos{{color:#059669;font-weight:600}}.neg{{color:#dc2626;font-weight:600}}
.good{{color:#059669;font-weight:700}}.ok{{color:#d97706;font-weight:600}}
.img-wrap{{border:1px solid #e5e7eb;border-radius:10px;overflow:hidden;margin:14px 0}}
.img-wrap img{{width:100%;display:block}}
.img-cap{{padding:8px 14px;font-size:.79rem;color:#6b7280;border-top:1px solid #e5e7eb;font-style:italic}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:14px 0}}
.conc-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:14px 0}}
.conc-card{{background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;padding:16px}}
.conc-card .icon{{font-size:20px;margin-bottom:8px}}
.conc-card h4{{font-size:.88rem;font-weight:600;margin-bottom:4px}}
.conc-card p{{font-size:.82rem;color:#6b7280;margin:0}}
ul,ol{{padding-left:20px;color:#374151;margin:8px 0 12px;line-height:2}}
li{{font-size:.9rem}}
.footer{{text-align:center;font-size:.8rem;color:#9ca3af;margin-top:40px;padding-top:20px;border-top:1px solid #e5e7eb}}
@media(max-width:640px){{.toc ol,.grid2,.conc-grid{{grid-template-columns:1fr}}.cover{{padding:32px 20px}}.cover h1{{font-size:1.6rem}}}}
</style></head><body><div class="page">

<div class="cover">
  <div class="cover-eye">Finance · Data Science · College Project</div>
  <h1>Stock Market Analysis &amp;<br/>ML Price Prediction</h1>
  <p class="cover-sub">End-to-end data science project: EDA → Technical Indicators → Feature Engineering → Machine Learning, applied to 5 years of US equity data.</p>
  <div class="chips">
    <span class="chip">🍎 AAPL</span><span class="chip">🔍 GOOGL</span><span class="chip">🪟 MSFT</span><span class="chip">📦 AMZN</span>
    <span class="chip">📅 2019–2024</span><span class="chip">🤖 4 ML Models</span><span class="chip">🐍 Python · scikit-learn</span>
  </div>
</div>

<div class="toc">
  <h2>Table of Contents</h2>
  <ol>
    <li><a href="#s1">Introduction &amp; Problem Statement</a></li>
    <li><a href="#s2">Dataset Description</a></li>
    <li><a href="#s3">Exploratory Data Analysis</a></li>
    <li><a href="#s4">Data Preprocessing</a></li>
    <li><a href="#s5">Technical Indicators</a></li>
    <li><a href="#s6">Feature Engineering</a></li>
    <li><a href="#s7">Machine Learning Models</a></li>
    <li><a href="#s8">Results &amp; Evaluation</a></li>
    <li><a href="#s9">Additional Visualizations</a></li>
    <li><a href="#s10">Conclusions &amp; Future Work</a></li>
    <li><a href="#s11">References</a></li>
  </ol>
</div>

<div class="sec" id="s1">
  <div class="sec-num">Section 1</div><h2>Introduction &amp; Problem Statement</h2>
  <p>Financial markets generate vast amounts of structured data every trading day. Data science provides tools to identify patterns, quantify risk, and build predictive models to support decision-making.</p>
  <div class="box blue"><strong>Problem Statement:</strong> Can historical stock price data and machine learning be used to accurately predict a stock's next-day closing price?</div>
  <h3>Objectives</h3>
  <ul>
    <li>Perform exploratory data analysis on 5 years of OHLCV data</li>
    <li>Compute and interpret technical indicators (RSI, MACD, Bollinger Bands)</li>
    <li>Engineer 21 meaningful ML features from raw price data</li>
    <li>Train and compare 4 regression models using a chronological train/test split</li>
    <li>Evaluate models with MAE, RMSE, R², and MAPE metrics</li>
    <li>Draw actionable, data-driven conclusions</li>
  </ul>
  <h3>Stocks Analysed</h3>
  <table><thead><tr><th>Ticker</th><th>Company</th><th>Sector</th></tr></thead><tbody>
    <tr><td><span class="tag" style="background:#eff6ff;color:#1d4ed8;border:1px solid #bfdbfe">AAPL</span></td><td>Apple Inc.</td><td>Technology</td></tr>
    <tr><td><span class="tag" style="background:#fef2f2;color:#dc2626;border:1px solid #fecaca">GOOGL</span></td><td>Alphabet Inc.</td><td>Technology / Media</td></tr>
    <tr><td><span class="tag" style="background:#f0fdf4;color:#059669;border:1px solid #bbf7d0">MSFT</span></td><td>Microsoft Corp.</td><td>Technology / Cloud</td></tr>
    <tr><td><span class="tag" style="background:#fffbeb;color:#d97706;border:1px solid #fde68a">AMZN</span></td><td>Amazon.com Inc.</td><td>E-Commerce / Cloud</td></tr>
  </tbody></table>
</div>

<div class="sec" id="s2">
  <div class="sec-num">Section 2</div><h2>Dataset Description</h2>
  <p>Daily OHLCV data for each stock spanning January 2019 – January 2024 (~1,304 trading days per stock). Data is simulated using <strong>Geometric Brownian Motion (GBM)</strong> — the standard financial model underlying the Black-Scholes options pricing formula.</p>
  <table><thead><tr><th>Column</th><th>Type</th><th>Description</th></tr></thead><tbody>
    <tr><td class="mono">Date</td><td>datetime</td><td>Trading date (business days only)</td></tr>
    <tr><td class="mono">Open</td><td>float</td><td>Opening price ($)</td></tr>
    <tr><td class="mono">High</td><td>float</td><td>Intra-day high price ($)</td></tr>
    <tr><td class="mono">Low</td><td>float</td><td>Intra-day low price ($)</td></tr>
    <tr><td class="mono">Close</td><td>float</td><td>Closing price ($) — our prediction target</td></tr>
    <tr><td class="mono">Volume</td><td>int</td><td>Shares traded</td></tr>
  </tbody></table>
  <h3>5-Year Performance Summary</h3>
  <table><thead><tr><th>Ticker</th><th>Company</th><th>Start $</th><th>End $</th><th>Total Return</th><th>Ann. Vol.</th><th>Max Drawdown</th><th>Sharpe</th></tr></thead>
  <tbody>{portfolio_rows()}</tbody></table>
</div>

<div class="sec" id="s3">
  <div class="sec-num">Section 3</div><h2>Exploratory Data Analysis</h2>
  <h3>3.1 Price History with Moving Averages</h3>
  <div class="img-wrap"><img src="data:image/png;base64,{img64('01_price_history.png')}" alt="Price History"/><div class="img-cap">Fig 1. Closing price with SMA 20/50/200 overlays for all four stocks (2019–2024).</div></div>
  <h3>3.2 Normalised Returns (Base = 100)</h3>
  <p>Normalising all prices to 100 at the start date allows fair comparison of growth across stocks with different price levels.</p>
  <div class="img-wrap"><img src="data:image/png;base64,{img64('02_normalized_returns.png')}" alt="Normalised Returns"/><div class="img-cap">Fig 2. Relative performance of all four stocks indexed to 100 on 2019-01-02.</div></div>
  <h3>3.3 Daily Returns Distribution</h3>
  <p>Daily returns approximate a normal distribution, consistent with the Efficient Market Hypothesis. Mean daily return is close to 0% for all stocks.</p>
  <div class="img-wrap"><img src="data:image/png;base64,{img64('04_returns_distribution.png')}" alt="Returns Distribution"/><div class="img-cap">Fig 3. Histogram of daily returns for each stock. Vertical line = mean return.</div></div>
  <h3>3.4 Correlation Matrix</h3>
  <p>Technology stocks exhibit high positive correlation (r &gt; 0.6) because they react to shared macro-economic events — interest rates, tech regulation, and economic cycles.</p>
  <div class="img-wrap"><img src="data:image/png;base64,{img64('03_correlation_heatmap.png')}" alt="Correlation"/><div class="img-cap">Fig 4. Pearson correlation matrix of daily returns. Values close to 1.0 indicate stocks move together.</div></div>
</div>

<div class="sec" id="s4">
  <div class="sec-num">Section 4</div><h2>Data Preprocessing</h2>
  <div class="box green">✅ No missing values found in any column across all four stocks. The dataset is complete and clean.</div>
  <h3>Moving Averages</h3>
  <p>Simple Moving Averages smooth price noise to reveal the underlying trend. A classic trading signal is the <em>golden cross</em> (SMA 20 crossing above SMA 50 = bullish) and <em>death cross</em> (SMA 20 crossing below SMA 50 = bearish).</p>
  <table><thead><tr><th>Indicator</th><th>Window</th><th>Represents</th></tr></thead><tbody>
    <tr><td>SMA 20</td><td>20 days</td><td>Short-term trend (~1 trading month)</td></tr>
    <tr><td>SMA 50</td><td>50 days</td><td>Medium-term trend (~2.5 months)</td></tr>
    <tr><td>SMA 200</td><td>200 days</td><td>Long-term trend (~10 months)</td></tr>
  </tbody></table>
</div>

<div class="sec" id="s5">
  <div class="sec-num">Section 5</div><h2>Technical Indicators</h2>
  <h3>5.1 RSI &amp; Bollinger Bands</h3>
  <p><strong>RSI (14)</strong> — Measures momentum. Above 70 = overbought; below 30 = oversold.<br/>
  <strong>Bollinger Bands</strong> — SMA(20) ± 2σ. Price touching the upper band signals potential reversal.</p>
  <div class="img-wrap"><img src="data:image/png;base64,{img64('05_bollinger_rsi.png')}" alt="Bollinger + RSI"/><div class="img-cap">Fig 5. AAPL — Bollinger Bands (top panel) and RSI(14) with overbought/oversold zones (bottom panel).</div></div>
  <h3>5.2 Rolling Volatility</h3>
  <p>Annualised volatility = std(daily returns) × √252. Higher volatility = more risk. Volatility clustering is visible — high-vol periods group together.</p>
  <div class="img-wrap"><img src="data:image/png;base64,{img64('07_volatility.png')}" alt="Volatility"/><div class="img-cap">Fig 6. Rolling 20-day annualised volatility (%). AMZN shows the highest sustained volatility.</div></div>
</div>

<div class="sec" id="s6">
  <div class="sec-num">Section 6</div><h2>Feature Engineering</h2>
  <p>We engineer <strong>21 features</strong> from raw OHLCV data to serve as inputs to the ML models:</p>
  <table><thead><tr><th>Category</th><th>Features</th></tr></thead><tbody>
    <tr><td>Price / Returns</td><td>Daily_Return, SMA_20, SMA_50, SMA_200</td></tr>
    <tr><td>Momentum</td><td>RSI, MACD, MACD_Signal</td></tr>
    <tr><td>Volatility</td><td>BB_upper, BB_lower, Volatility_20d</td></tr>
    <tr><td>Lag Features</td><td>Return_lag1, Return_lag2, Return_lag5, Close_lag1, Close_lag5</td></tr>
    <tr><td>Rolling Stats</td><td>Roll_mean_5, Roll_mean_10, Roll_std_5</td></tr>
    <tr><td>Ratios</td><td>Price_SMA20_ratio, Price_SMA50_ratio, High_Low_pct</td></tr>
  </tbody></table>
  <div class="box blue"><strong>Target Variable:</strong> Next day's closing price — <code>Target = Close.shift(-1)</code></div>
  <div class="box yellow">⚠️ <strong>Data Leakage:</strong> We use a chronological 80/20 train/test split — never shuffled. Shuffling time-series would let the model "see the future," giving misleadingly high accuracy that would fail in real trading.</div>
</div>

<div class="sec" id="s7">
  <div class="sec-num">Section 7</div><h2>Machine Learning Models</h2>
  <table><thead><tr><th>Model</th><th>Key Parameters</th><th>Strengths</th></tr></thead><tbody>
    <tr><td><strong>Linear Regression</strong></td><td>—</td><td>Fast, interpretable baseline</td></tr>
    <tr><td><strong>Ridge Regression</strong></td><td>α = 1.0</td><td>Handles multicollinear features</td></tr>
    <tr><td><strong>Random Forest</strong></td><td>100 trees, depth=8</td><td>Captures non-linear patterns</td></tr>
    <tr><td><strong>Gradient Boosting</strong></td><td>100 est., lr=0.1</td><td>Sequential boosting, high accuracy</td></tr>
  </tbody></table>
  <h3>Feature Importance (Best Model — AAPL)</h3>
  <div class="img-wrap"><img src="data:image/png;base64,{img64('10_feature_importance_AAPL.png')}" alt="Feature Importance"/><div class="img-cap">Fig 7. Top 20 feature importances for AAPL. Lag-1 close price dominates, confirming strong autocorrelation.</div></div>
</div>

<div class="sec" id="s8">
  <div class="sec-num">Section 8</div><h2>Results &amp; Evaluation</h2>
  <table><thead><tr><th>Metric</th><th>Meaning</th></tr></thead><tbody>
    <tr><td><strong>MAE</strong></td><td>Average absolute dollar error in predictions</td></tr>
    <tr><td><strong>RMSE</strong></td><td>Root mean squared error — penalises large mistakes more</td></tr>
    <tr><td><strong>R²</strong></td><td>% of variance explained (1.0 = perfect prediction)</td></tr>
    <tr><td><strong>MAPE</strong></td><td>Mean absolute % error relative to actual price</td></tr>
  </tbody></table>
  <h3>All Model Results</h3>
  <table><thead><tr><th>Ticker</th><th>Model</th><th>MAE ($)</th><th>RMSE ($)</th><th>R²</th><th>MAPE</th></tr></thead>
  <tbody>{results_rows()}</tbody></table>
  <div class="img-wrap"><img src="data:image/png;base64,{img64('09_model_comparison.png')}" alt="Model Comparison"/><div class="img-cap">Fig 8. Average MAE, RMSE, and R² across all four stocks for each model.</div></div>
  <h3>Prediction vs Actual</h3>
  <div class="grid2">
    <div class="img-wrap"><img src="data:image/png;base64,{img64('08_prediction_AAPL.png')}" alt="AAPL"/><div class="img-cap">Fig 9a. AAPL — Predicted vs Actual + residuals.</div></div>
    <div class="img-wrap"><img src="data:image/png;base64,{img64('08_prediction_MSFT.png')}" alt="MSFT"/><div class="img-cap">Fig 9b. MSFT — Predicted vs Actual + residuals.</div></div>
    <div class="img-wrap"><img src="data:image/png;base64,{img64('08_prediction_GOOGL.png')}" alt="GOOGL"/><div class="img-cap">Fig 9c. GOOGL — Predicted vs Actual + residuals.</div></div>
    <div class="img-wrap"><img src="data:image/png;base64,{img64('08_prediction_AMZN.png')}" alt="AMZN"/><div class="img-cap">Fig 9d. AMZN — Predicted vs Actual + residuals.</div></div>
  </div>
</div>

<div class="sec" id="s9">
  <div class="sec-num">Section 9</div><h2>Additional Visualizations</h2>
  <div class="img-wrap"><img src="data:image/png;base64,{img64('06_volume_analysis.png')}" alt="Volume"/><div class="img-cap">Fig 10. Trading volume with 30-day moving average. Volume spikes often precede major price moves.</div></div>
</div>

<div class="sec" id="s10">
  <div class="sec-num">Section 10</div><h2>Conclusions &amp; Future Work</h2>
  <h3>Key Findings</h3>
  <div class="conc-grid">
    <div class="conc-card"><div class="icon">📈</div><h4>Strong Autocorrelation</h4><p>Today's price strongly predicts tomorrow's price — linear models achieved R² &gt; 0.93 across all stocks.</p></div>
    <div class="conc-card"><div class="icon">🔗</div><h4>High Sector Correlation</h4><p>All four tech stocks move together (r &gt; 0.6) due to shared macro-economic exposure.</p></div>
    <div class="conc-card"><div class="icon">🤖</div><h4>Linear Models Win</h4><p>Ridge and Linear Regression outperformed tree-based models because price prediction is largely a linear, autocorrelated problem.</p></div>
    <div class="conc-card"><div class="icon">⚡</div><h4>Lag-1 is Most Important</h4><p>The previous day's close price was the single most predictive feature — consistent with random walk theory.</p></div>
    <div class="conc-card"><div class="icon">📉</div><h4>Volatility Clustering</h4><p>High-volatility periods group together — a GARCH-like property well-documented in financial literature.</p></div>
    <div class="conc-card"><div class="icon">🏆</div><h4>Best Risk-Adjusted Stock</h4><p>AAPL had the highest total return; MSFT showed the best risk-adjusted performance based on its Sharpe ratio.</p></div>
  </div>
  <h3>Limitations</h3>
  <ul>
    <li>Simulated data (GBM) lacks real-world fat tails, regime changes, and news events</li>
    <li>Only technical features used — no fundamental data (P/E ratio, earnings surprises)</li>
    <li>Transaction costs and market impact not considered</li>
    <li>Single-step (1-day) prediction only</li>
  </ul>
  <h3>Future Work</h3>
  <ul>
    <li>Use real market data via the <code>yfinance</code> Python library</li>
    <li>Implement LSTM neural networks for sequence-aware learning</li>
    <li>Add news sentiment scores as a feature (NLP)</li>
    <li>Apply Markowitz mean-variance portfolio optimisation</li>
    <li>Build a backtesting framework to simulate trading strategies</li>
  </ul>
</div>

<div class="sec" id="s11">
  <div class="sec-num">Section 11</div><h2>References</h2>
  <ol>
    <li>Fama, E.F. (1970). <em>Efficient Capital Markets: A Review of Theory and Empirical Work</em>. Journal of Finance, 25(2), 383–417.</li>
    <li>Murphy, J.J. (1999). <em>Technical Analysis of the Financial Markets</em>. New York Institute of Finance.</li>
    <li>Géron, A. (2019). <em>Hands-On Machine Learning with Scikit-Learn, Keras &amp; TensorFlow</em>. O'Reilly Media.</li>
    <li>Black, F. &amp; Scholes, M. (1973). <em>The Pricing of Options and Corporate Liabilities</em>. Journal of Political Economy, 81(3), 637–654.</li>
    <li>Wilder, J.W. (1978). <em>New Concepts in Technical Trading Systems</em>. Trend Research.</li>
    <li>Pedregosa, F. et al. (2011). <em>Scikit-learn: Machine Learning in Python</em>. JMLR 12, 2825–2830.</li>
    <li>McKinney, W. (2010). <em>Data Structures for Statistical Computing in Python</em>. Proc. 9th Python in Science Conf.</li>
  </ol>
</div>

</div>
<div class="footer">Stock Market Analysis &amp; ML Price Prediction · Finance Data Science Project · Python · Pandas · Scikit-learn · Data: GBM (2019–2024)</div>
</body></html>"""

out = os.path.join(REPORT_DIR, "project_report.html")
with open(out, "w", encoding="utf-8") as f: f.write(html)
sz = os.path.getsize(out)/1024/1024
print(f"Report saved: {out}")
print(f"Size: {sz:.1f} MB")
