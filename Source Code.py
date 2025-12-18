import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold


# PARAMETERS
fx_pairs               = ["EURUSD=X", "GBPUSD=X", "GC=F"]
transaction_cost       = 0.00005
risk_per_trade         = 0.02
initial_capital        = 100000
stop_loss_atr_mult     = 0.1
target_profit_atr_mult = 1
train_window           = 20
data_period            = '3y'
max_hold               = 7


base_model = XGBClassifier(eval_metric='logloss', random_state=42)
param_dist  = {
    'n_estimators':     [50, 100, 200, 300],
    'max_depth':        [4,5, 6, 8, 10],
    'learning_rate':    [0.01, 0.03, 0.05, 0.1],
    'subsample':        [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'reg_lambda':       [0.1, 0.5, 1.0, 2.0, 5.0],
    'gamma':            [0, 0.1, 0.2, 0.5],
    'min_child_weight': [1, 3, 5, 10]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=100,               # number of random combos to try
    scoring='roc_auc',       # maximize ROC AUC
    cv=cv,
    verbose=0,
    n_jobs=-1,
    random_state=42
)



all_trades = {}
summary    = []

for pair in fx_pairs:
    print(f"Processing {pair}...")
    df = yf.download(pair,
                     interval="1D",
                     period=data_period,
                     auto_adjust=True,
                     progress=False)
    if df.empty:
        print(f" No data for {pair}, skipping.")
        continue

    df.dropna(inplace=True)
    df = df.copy()

    # Build 1D price series
    close = pd.Series(df["Close"].values.flatten(), index=df.index)
    high  = pd.Series(df["High"].values.flatten(), index=df.index)
    low   = pd.Series(df["Low"].values.flatten(), index=df.index)

    # Features
    df["log_return"] = np.log(close / close.shift(1))
    df["RSI"]        = RSIIndicator(close=close, window=14).rsi()
    df["ATR"]        = pd.concat([
                         high - low,
                         (high - close.shift(1)).abs(),
                         (low  - close.shift(1)).abs()
                       ], axis=1).max(axis=1).rolling(3).mean()
    
    df['MACD'] = MACD(close=close).macd_diff()
    bb_h = close.rolling(window=7).mean() + 2 * close.rolling(window=7).std()
    bb_l = close.rolling(window=7).mean() - 2 * close.rolling(window=7).std()
    df['BB_width'] = bb_h - bb_l
    
    
    
    

    # Target
    df["target"] = (df["log_return"].shift(-1) > 0).astype(int)

    df.dropna(inplace=True)
    df = df.copy()
    close = pd.Series(df["Close"].values.flatten(), index=df.index)
    high  = pd.Series(df["High"].values.flatten(), index=df.index)
    low   = pd.Series(df["Low"].values.flatten(), index=df.index)

    features = ['RSI', 'MACD', 'BB_width', 'ATR']
    df["pred_prob"] = np.nan
    df["signal"]    = 0
    X_all = df[features].values
    y_all = df["target"].values
   # print(" Searching best hyperparameters...")
    search.fit(X_all, y_all)
    best_params = search.best_params_
   # print(" Best params:", best_params)
    df['ATR_avg'] = df['ATR'].rolling(window=7).mean()
    df['HighVol'] = df['ATR'] > df['ATR_avg']*1.1








    # Rolling train + signal every bar
    for i in range(train_window, len(df)):
        train = df.iloc[i-train_window : i]
        model = XGBClassifier(
            **best_params,
            eval_metric='logloss',
            random_state=42
)
        model.fit(train[features], train["target"])

        idx = df.index[i]
        prob = model.predict_proba(df.loc[[idx], features])[0, 1]
        df.loc[idx, 'pred_prob'] = prob
        #df.loc[idx, 'signal'] = np.where(prob > 0.5, 1, -1)
        highvol = df['HighVol'].iloc[i]
        if prob > 0.5 and highvol  :
            df.loc[idx, 'signal'] = 1
        elif prob < 0.2  and highvol:
            df.loc[idx, 'signal'] = -1

    # Prepare entry price + TP/SL
    df["entry_price"] = np.nan
    df["tp_price"]    = np.nan
    df["sl_price"]    = np.nan
    for i in range(train_window, len(df)):
        if df["signal"].iat[i] != 0 and df["signal"].iat[i-1] != df["signal"].iat[i]:
            entry = close.iat[i]
            atr   = df["ATR"].iat[i]
            sign   = df["signal"].iat[i]
            df.loc[df.index[i], 'entry_price'] = entry
            df.loc[df.index[i], 'tp_price']    = entry + (target_profit_atr_mult * atr * sign)
            df.loc[df.index[i], 'sl_price']    = entry - (stop_loss_atr_mult * atr * sign)

    # Simulate trades until TP, SL, signal-flip, or max_hold
    capital = initial_capital
    trades = []
    i = train_window
    while i < len(df):
        sig = df["signal"].iat[i]
        if sig == 0:
            i += 1
            continue
        
        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df["cum_capital"] = initial_capital + trades_df["pnl"].cumsum()
            
        else:
            trades_df["cum_capital"] = initial_capital 
            
        all_trades[pair] = trades_df
        cc = trades_df["cum_capital"].dropna()
        
        if not cc.empty:
             Previous_cap = cc.iat[-1]
        else:
             Previous_cap = initial_capital
        prev_cap = trades_df["cum_capital"].iat[-1] if not trades_df.empty else initial_capital
             
        entry_idx   = i
        tp_price    = df["tp_price"].iat[i]
        sl_price    = df["sl_price"].iat[i]
        entry_price = close.iat[entry_idx]
        entry_atr   = df["ATR"].iat[entry_idx]
        exit_idx    = None
        exit_price  = None
        size = (risk_per_trade * prev_cap) / (stop_loss_atr_mult * entry_atr)
        
        for j in range(i+1, len(df)):
            hold_len = j - entry_idx

            # signal-flip exit
            if df["signal"].iat[j] != sig:
                exit_idx, exit_price = j, close.iat[j]
                break

            # max hold exit
            if hold_len >= max_hold:
                exit_idx, exit_price = j, close.iat[j]
                break

            # take-profit / stop-loss
            if sig > 0 and high.iat[j] >= tp_price:
                exit_idx, exit_price = j, tp_price
                break
            if sig > 0 and low.iat[j]  <= sl_price:
                exit_idx, exit_price = j, sl_price
                break
            if sig < 0 and low.iat[j]  <= tp_price:
                exit_idx, exit_price = j, tp_price
                break
            if sig < 0 and high.iat[j] >= sl_price:
                exit_idx, exit_price = j, sl_price
                break

        # default exit at last bar
        if exit_idx is None:
            exit_idx, exit_price = len(df)-1, close.iat[-1]
            
         # Summary metrics
       

             
        entry_price = close.iat[entry_idx]
       # size        = (risk_per_trade * Previous_cap) / (df["ATR"].iat[entry_idx] * stop_loss_atr_mult)
        max_loss  = (risk_per_trade * Previous_cap)+ (transaction_cost*size)
        max_profit  = (tp_price-entry_price-transaction_cost)*size*sig
        pnl         = max(size * (sig * (exit_price-entry_price- transaction_cost)) ,- max_loss)

        
        

        trades.append({
            "entry_date":    df.index[entry_idx],
            "exit_date":     df.index[exit_idx],
            "signal":        sig,
            "entry_price":   entry_price,
            "exit_price":    exit_price,
            "position_size": size,
            "pnl":           pnl,
            "Previous_cap":           Previous_cap
        })
        i = exit_idx + 1



    # Summary metrics
    cc = trades_df["cum_capital"].dropna()
    if not cc.empty:
        final_cap = cc.iat[-1]
    else:
        final_cap = initial_capital
        
    #final_cap = trades_df["cum_capital"].iat[-1].notna() if not trades_df.empty else initial_capital
    test_df = df[df['pred_prob'].notna()]
    days_span = (test_df.index[-1] - test_df.index[0]).days
    years = days_span / 365.25 if days_span > 0 else 1
    days      = (df.index[-1] - df.index[0]).days / 365.25
    trading_days_per_year = len(test_df) / years
   
    cagr      = (final_cap / initial_capital)**(1/years) - 1
    cagr_pct  = cagr * 100
    start_price = df["Close"].iloc[0]
    end_price    = df["Close"].iloc[-1]
    mkt_cagr     = (end_price / start_price)**(1/years) - 1
    mkt_cagr_pct = mkt_cagr * 100
    ann_ret   = trades_df["pnl"].mean() * (len(trades_df)/days) if not trades_df.empty else 0
    ann_vol   = trades_df["pnl"].std() * np.sqrt(len(trades_df)/days) if not trades_df.empty else np.nan
    sharpe    = ann_ret / ann_vol if ann_vol else np.nan
    accuracy = accuracy_score(test_df['target'], (test_df['pred_prob'] > 0.5).astype(int))
    roc_auc = roc_auc_score(test_df['target'], test_df['pred_prob'])
   
   
    alpha_pct    = cagr_pct - mkt_cagr_pct
    N = len(trades_df)
    # wins and losses
    wins   = trades_df.loc[trades_df['pnl'] > 0, 'pnl']
    losses = trades_df.loc[trades_df['pnl'] <= 0, 'pnl']
    # win rate
    W = len(wins) / N if N else np.nan
    # average win and average loss
    avg_win  = wins.mean()   if not wins.empty  else 0.0
    avg_loss = losses.mean() if not losses.empty else 0.0
    # expected value per trade
    EV = W * avg_win + (1 - W) * avg_loss
    
    alpha_val = alpha_pct.iloc[0]
    # win rate
    W = len(wins) / N if N else np.nan
    
    summary.append({
        "Pair":           pair,
        "Trades":         len(trades_df),
        #'Accuracy': round(accuracy, 4),
        #'ROC AUC': round(roc_auc, 4),
        "Alpha (%)": f"{alpha_val:.2f}%",
        "Sharpe Ratio":   round(sharpe, 3),
        
        
        
        "Expected Value":   round(EV, 4),
        "Final Capital": f"${final_cap:,.2f}"
    })

    # Plots
    if not trades_df.empty:
        plt.figure(figsize=(10,4))
        plt.plot(trades_df["entry_date"], trades_df["cum_capital"], marker="o")
        plt.title(f"{pair} Equity Curve")
        plt.xlabel("Date"); plt.ylabel("Capital"); plt.grid(); plt.show()

        plt.figure(figsize=(12,4))
        plt.plot(df.index, close, label="Close")
        plt.scatter(trades_df["entry_date"], trades_df["entry_price"],
                    marker="^", label="Entry")
        plt.scatter(trades_df["exit_date"], trades_df["exit_price"],
                    marker="v", label="Exit")
        plt.title(f"{pair} Trades on Price"); plt.legend(); plt.grid(); plt.show()

# Final summary and export
summary_df = pd.DataFrame(summary)
print("\nPerformance Summary:")
print(summary_df.to_markdown(index=False))

with pd.ExcelWriter("all_trades_tp_sl.xlsx", engine="openpyxl") as writer:
    for pair, tdf in all_trades.items():
        sheet = pair.replace("=", "").replace("-", "").replace(" ", "_")[:31]
        tdf.to_excel(writer, sheet_name=sheet, index=False)

print("👉 Saved all_trades_tp_sl.xlsx")
