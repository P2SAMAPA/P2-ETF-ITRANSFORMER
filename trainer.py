import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import torch
import config
import data_manager
from itransformer import train_model, predict
from sklearn.preprocessing import StandardScaler

def create_sequences(df, seq_len):
    T, num_vars = df.shape
    X, y = [], []
    for i in range(seq_len, T):
        X.append(df.iloc[i-seq_len:i].values.T)   # (num_vars, seq_len)
        y.append(df.iloc[i].values)
    return np.array(X), np.array(y)

def compute_conformal_intervals(predictions, residuals, confidence=0.9):
    abs_res = np.abs(residuals)
    q = np.quantile(abs_res, confidence)
    lower = predictions - q
    upper = predictions + q
    return lower, upper

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} (iTransformer) ===")
        combined = data_manager.prepare_combined_data(df, tickers)
        print(f"  Combined data length: {len(combined)} days")
        if combined.empty or len(combined) < max(config.WINDOWS) + config.SEQ_LEN + 10:
            print("  Insufficient combined data – falling back to returns only")
            # Fallback: use only ETF returns (no macro)
            combined = data_manager.prepare_returns_matrix(df, tickers)
            print(f"  Returns‑only data length: {len(combined)} days")
            if combined.empty or len(combined) < max(config.WINDOWS) + config.SEQ_LEN + 10:
                print("  Still insufficient, skipping universe")
                all_results[universe_name] = {"top_etfs": []}
                continue

        best_per_etf = {}
        window_results = {}

        for win in config.WINDOWS:
            if len(combined) < win + config.SEQ_LEN + 10:
                print(f"  Skipping window {win}d (need at least {win + config.SEQ_LEN + 10} days, have {len(combined)})")
                continue
            print(f"  Processing window {win}d...")
            data_win = combined.iloc[-win:]
            # Standardise
            scaler = StandardScaler()
            data_scaled = pd.DataFrame(scaler.fit_transform(data_win), index=data_win.index, columns=data_win.columns)
            # Create sequences
            X, y = create_sequences(data_scaled, config.SEQ_LEN)
            if len(X) < 20:
                print(f"    Not enough sequences (need 20, got {len(X)})")
                continue
            split = int(0.8 * len(X))
            X_train, X_cal = X[:split], X[split:]
            y_train, y_cal = y[:split], y[split:]
            num_vars = data_scaled.shape[1]
            model = train_model(X_train, y_train, config.SEQ_LEN, num_vars,
                                epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
                                lr=config.LR, device=device)
            pred_cal = predict(model, X_cal, device)
            residuals = y_cal - pred_cal
            last_seq = data_scaled.iloc[-config.SEQ_LEN:].values.T.reshape(1, num_vars, config.SEQ_LEN)
            pred_last = predict(model, last_seq, device)[0]
            lower, upper = compute_conformal_intervals(pred_last, residuals, confidence=config.CONFIDENCE)
            etf_cols = [c for c in data_scaled.columns if c not in config.MACRO_COLUMNS]
            etf_indices = [list(data_scaled.columns).index(c) for c in etf_cols]
            scores = {etf_cols[i]: upper[etf_indices[i]] for i in range(len(etf_cols))}
            window_results[win] = scores
            for etf, score in scores.items():
                if etf not in best_per_etf or score > best_per_etf[etf][0]:
                    best_per_etf[etf] = (score, win)

        if not best_per_etf:
            print("  No valid predictions – falling back to historical mean return")
            returns = data_manager.prepare_returns_matrix(df, tickers)
            for etf in tickers:
                if etf in returns.columns:
                    mean_ret = returns[etf].iloc[-252:].mean()
                    if not np.isnan(mean_ret):
                        best_per_etf[etf] = (max(mean_ret, 1e-6), 0)
            if not best_per_etf:
                all_results[universe_name] = {"top_etfs": []}
                continue

        full_scores = {ticker: {"score": score, "best_window": win} for ticker, (score, win) in best_per_etf.items()}
        sorted_etfs = sorted(best_per_etf.items(), key=lambda x: x[1][0], reverse=True)
        top_etfs = [{"ticker": ticker, "score": float(score), "best_window": win} for ticker, (score, win) in sorted_etfs[:config.TOP_N]]

        print(f"  Top 3 ETFs by conformal upper bound: {[e['ticker'] for e in top_etfs]}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "full_scores": full_scores,
            "window_results": window_results,
            "run_date": today
        }

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/itransformer_{today}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": today, "universes": all_results}, f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== iTransformer Engine complete ===")

if __name__ == "__main__":
    main()
