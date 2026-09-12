import os
import shutil
import tempfile
import threading
import pandas as pd
import numpy as np
import time
import gc
import random
import warnings
from itertools import product as iprod

import torch
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

warnings.filterwarnings("ignore")

try:
    import pytorch_lightning as pl
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATS, NHITS, PatchTST, LSTM, TFT, TiDE
    from neuralforecast.losses.pytorch import MAE as NF_MAE
    NEURAL_AVAILABLE = True

    class ModelTrainingProgressCallback(pl.Callback):
        """Her batch veya eğitim adımında anlık adım, loss ve süreyi yakalar."""
        def __init__(self, callback_fn, model_name, params_str, h_label, current_trial, total_trials, max_steps, t0):
            super().__init__()
            self.callback_fn = callback_fn
            self.model_name = model_name
            self.params_str = params_str
            self.h_label = h_label
            self.current_trial = current_trial
            self.total_trials = total_trials
            self.max_steps = max_steps
            self.t0 = t0
            self.last_emit = 0.0

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            now = time.time()
            step = trainer.global_step
            # UI performansını korumak için saniyede en fazla 3-4 kez emit et
            if (now - self.last_emit >= 0.25) or (step >= self.max_steps):
                self.last_emit = now
                loss_val = None
                try:
                    if isinstance(outputs, dict) and "loss" in outputs:
                        loss_val = round(float(outputs["loss"].item()), 4)
                except Exception:
                    pass

                elapsed = round(now - self.t0, 1)
                pct_step = round((step / max(self.max_steps, 1)) * 100, 1)
                if self.callback_fn:
                    self.callback_fn({
                        "type": "step_progress",
                        "model": self.model_name,
                        "params": self.params_str,
                        "horizon": self.h_label,
                        "current_trial": self.current_trial,
                        "total": self.total_trials,
                        "step": step,
                        "max_steps": self.max_steps,
                        "loss": loss_val,
                        "stage": f"Eğitiliyor (Adım {step}/{self.max_steps} - %{pct_step})",
                        "trial_elapsed": elapsed,
                    })
except ImportError:
    NEURAL_AVAILABLE = False


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def calc_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    try:
        r2 = r2_score(y_true, y_pred)
    except:
        r2 = -999.0
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-10))) * 100
    return round(mae, 4), round(rmse, 4), round(r2, 4), round(mape, 2)


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df["ds"] = pd.to_datetime(df["Tarih"])
    df["y"] = df["Fiyat"]
    df = df.sort_values("ds").reset_index(drop=True)
    return df[["ds", "y"]]


# ============================================================
#  PROPHET WORKER
# ============================================================

def run_prophet_grid(df_raw, grid, horizons, asset_name="Asset", completed_dict=None, callback=None, stop_event=None):
    """
    Prophet grid search calistirir.
    completed_dict: {(asset, model, params, horizon): row} - onceden tamamlananlar atlanir.
    callback(result_dict): Her sonuc icin cagirilir.
    stop_event: threading.Event, set edilirse durur.
    """
    combos = list(iprod(
        grid["changepoint_prior_scale"],
        grid["changepoint_range"],
        grid["seasonality_mode"],
        grid["use_log"],
    ))
    total = len(combos) * len(horizons)
    done = 0

    for cps, cr, sm, use_log in combos:
        if stop_event and stop_event.is_set():
            return

        params_str = f"scale={cps}, range={cr}, mode={sm}, log={use_log}"

        for h_label, h_days in horizons.items():
            if h_days >= len(df_raw) - 100:
                done += 1
                continue

            # Checkpoint Kontrolu
            key = (asset_name, "Prophet", params_str, h_label)
            if completed_dict is not None and key in completed_dict:
                done += 1
                cached = completed_dict[key]
                if callback:
                    callback({
                        "model": "Prophet",
                        "params": params_str,
                        "horizon": h_label,
                        "r2": cached["r2"],
                        "mae": cached["mae"],
                        "rmse": cached["rmse"],
                        "mape": cached["mape"],
                        "elapsed": cached.get("elapsed", 0),
                        "done": done,
                        "total": total,
                        "status": "ok",
                        "is_cached": True,
                    })
                continue

            t0 = time.time()
            try:
                seed_everything(42)
                train_df = df_raw.iloc[:-h_days].copy()
                test_df = df_raw.iloc[-h_days:].copy()
                actuals = test_df["y"].values

                tr = train_df[["ds", "y"]].copy()
                if use_log:
                    tr["y"] = np.log(tr["y"])

                if callback:
                    callback({
                        "type": "step_progress",
                        "model": "Prophet",
                        "params": params_str,
                        "horizon": h_label,
                        "current_trial": done + 1,
                        "total": total,
                        "step": 0,
                        "max_steps": 1,
                        "loss": None,
                        "stage": "L-BFGS Optimizasyonu ile Fit Ediliyor (CPU)",
                        "trial_elapsed": round(time.time() - t0, 1),
                    })

                m = Prophet(
                    daily_seasonality=False,
                    weekly_seasonality=True,
                    yearly_seasonality=True,
                    changepoint_prior_scale=cps,
                    changepoint_range=cr,
                    seasonality_mode=sm,
                    seasonality_prior_scale=1.0,
                )
                m.fit(tr)

                if callback:
                    callback({
                        "type": "step_progress",
                        "model": "Prophet",
                        "params": params_str,
                        "horizon": h_label,
                        "current_trial": done + 1,
                        "total": total,
                        "step": 1,
                        "max_steps": 1,
                        "loss": None,
                        "stage": f"Gelecek {h_days} Günlük Tahminler Hesaplanıyor",
                        "trial_elapsed": round(time.time() - t0, 1),
                    })

                fc = m.predict(m.make_future_dataframe(periods=h_days))
                preds = fc["yhat"].iloc[-h_days:].values

                if use_log:
                    preds = np.exp(preds)

                mae, rmse, r2, mape = calc_metrics(actuals, preds)
                elapsed = round(time.time() - t0, 1)

                del m, fc, tr

                done += 1
                if callback:
                    callback({
                        "model": "Prophet",
                        "params": params_str,
                        "horizon": h_label,
                        "r2": r2, "mae": mae, "rmse": rmse, "mape": mape,
                        "elapsed": elapsed,
                        "done": done, "total": total,
                        "status": "ok",
                    })
            except Exception as e:
                done += 1
                if callback:
                    callback({
                        "model": "Prophet",
                        "params": params_str,
                        "horizon": h_label,
                        "r2": -999, "mae": 999, "rmse": 999, "mape": 999,
                        "elapsed": round(time.time() - t0, 1),
                        "done": done, "total": total,
                        "status": f"HATA: {str(e)[:60]}",
                    })

            clear_memory()


# ============================================================
#  XGBOOST WORKER
# ============================================================

def create_xgb_features(df, lag_days):
    data = df.copy()
    data["log_ret"] = np.log(data["y"] / data["y"].shift(1))
    data["dayofweek"] = data["ds"].dt.dayofweek
    data["month"] = data["ds"].dt.month
    data["is_quarter_end"] = data["ds"].dt.is_quarter_end.astype(int)

    for i in range(1, lag_days + 1):
        data[f"lag_ret_{i}"] = data["log_ret"].shift(i)
    for w in [5, 10, 20]:
        data[f"ma_ret_{w}"] = data["log_ret"].rolling(w).mean()
        ma_price = data["y"].rolling(w).mean()
        data[f"dist_ma_{w}"] = (data["y"] - ma_price) / ma_price

    data["volatility"] = data["log_ret"].rolling(10).std()
    delta = data["y"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    data["rsi"] = 100 - (100 / (1 + rs))

    data = data.dropna().reset_index(drop=True)
    return data


def run_xgboost_grid(df_raw, grid, horizons, asset_name="Asset", completed_dict=None, callback=None, stop_event=None):
    combos = list(iprod(
        grid["n_estimators"],
        grid["learning_rate"],
        grid["max_depth"],
        grid["lag_days"],
        grid["use_log"],
    ))
    total = len(combos) * len(horizons)
    done = 0

    for n_est, lr, md, lag, use_log in combos:
        if stop_event and stop_event.is_set():
            return

        params_str = f"n_est={n_est}, lr={lr}, depth={md}, lag={lag}, log={use_log}"

        for h_label, h_days in horizons.items():
            if h_days >= len(df_raw) - 200:
                done += 1
                continue

            # Checkpoint Kontrolu
            key = (asset_name, "XGBoost", params_str, h_label)
            if completed_dict is not None and key in completed_dict:
                done += 1
                cached = completed_dict[key]
                if callback:
                    callback({
                        "model": "XGBoost",
                        "params": params_str,
                        "horizon": h_label,
                        "r2": cached["r2"],
                        "mae": cached["mae"],
                        "rmse": cached["rmse"],
                        "mape": cached["mape"],
                        "elapsed": cached.get("elapsed", 0),
                        "done": done,
                        "total": total,
                        "status": "ok",
                        "is_cached": True,
                    })
                continue

            t0 = time.time()
            try:
                seed_everything(42)
                train_df = df_raw.iloc[:-h_days].copy()
                test_df = df_raw.iloc[-h_days:].copy()
                actuals = test_df["y"].values

                work_df = train_df.copy()
                if use_log:
                    work_df["y"] = np.log(work_df["y"])

                data = create_xgb_features(work_df, lag)
                if len(data) < 50:
                    done += 1
                    continue

                feature_cols = [c for c in data.columns if c not in ["ds", "y", "log_ret"]]
                X = data[feature_cols].values
                y_target = data["log_ret"].values

                split = int(len(X) * 0.9)
                X_train, X_val = X[:split], X[split:]
                y_train, y_val = y_target[:split], y_target[split:]

                if callback:
                    callback({
                        "type": "step_progress",
                        "model": "XGBoost",
                        "params": params_str,
                        "horizon": h_label,
                        "current_trial": done + 1,
                        "total": total,
                        "step": 0,
                        "max_steps": n_est,
                        "loss": None,
                        "stage": f"Ağaçlar Eğitiliyor ({n_est} Ağaç, Derinlik: {md})",
                        "trial_elapsed": round(time.time() - t0, 1),
                    })

                model = XGBRegressor(
                    n_estimators=n_est,
                    learning_rate=lr,
                    max_depth=md,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.5,
                    objective="reg:squarederror",
                    tree_method="hist",
                    early_stopping_rounds=50,
                    eval_metric="mae",
                    random_state=42,
                )
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

                if callback:
                    callback({
                        "type": "step_progress",
                        "model": "XGBoost",
                        "params": params_str,
                        "horizon": h_label,
                        "current_trial": done + 1,
                        "total": total,
                        "step": n_est,
                        "max_steps": n_est,
                        "loss": None,
                        "stage": f"{h_days} Günlük Otoregresif Fiyat Projeksiyonu",
                        "trial_elapsed": round(time.time() - t0, 1),
                    })

                last_price = data["y"].iloc[-1]
                last_data = work_df.tail(100).copy()
                preds = []
                future_dates = pd.date_range(
                    start=train_df["ds"].iloc[-1] + pd.Timedelta(days=1),
                    periods=h_days,
                )
                recent_trend = data["log_ret"].tail(30).mean()

                for idx in range(h_days):
                    current_feats = create_xgb_features(last_data, lag)
                    if len(current_feats) == 0:
                        pred_ret = 0
                    else:
                        last_feat_row = current_feats[feature_cols].iloc[-1:].values
                        pred_ret = model.predict(last_feat_row)[0]

                    min_w = max(0.1, 0.5 - 0.02 * (h_days / 30.0))
                    tw = max(min_w, 1.0 - (idx / (h_days * 1.3)))
                    mr = 0.75
                    pred_ret = (pred_ret * mr) + (recent_trend * tw * (1.0 - mr))

                    next_price = last_price * np.exp(pred_ret)
                    preds.append(next_price)
                    new_row = pd.DataFrame({"ds": [future_dates[idx]], "y": [next_price]})
                    last_data = pd.concat([last_data, new_row], ignore_index=True).tail(100)
                    last_price = next_price

                preds = np.array(preds)
                if use_log:
                    preds = np.exp(preds)

                mae, rmse, r2, mape = calc_metrics(actuals, preds)
                elapsed = round(time.time() - t0, 1)

                del model
                done += 1
                if callback:
                    callback({
                        "model": "XGBoost",
                        "params": params_str,
                        "horizon": h_label,
                        "r2": r2, "mae": mae, "rmse": rmse, "mape": mape,
                        "elapsed": elapsed,
                        "done": done, "total": total,
                        "status": "ok",
                    })
            except Exception as e:
                done += 1
                if callback:
                    callback({
                        "model": "XGBoost",
                        "params": params_str,
                        "horizon": h_label,
                        "r2": -999, "mae": 999, "rmse": 999, "mape": 999,
                        "elapsed": round(time.time() - t0, 1),
                        "done": done, "total": total,
                        "status": f"HATA: {str(e)[:60]}",
                    })

            clear_memory()


# ============================================================
#  NEURAL WORKER (NBEATS, NHITS, PatchTST, LSTM)
# ============================================================

def run_neural_grid(df_raw, model_name, grid, horizons, asset_name="Asset", completed_dict=None, use_gpu=False, callback=None, stop_event=None):
    if not NEURAL_AVAILABLE:
        if callback:
            callback({
                "model": model_name,
                "params": "-",
                "horizon": "-",
                "r2": -999, "mae": 999, "rmse": 999, "mape": 999,
                "elapsed": 0,
                "done": 1, "total": 1,
                "status": "HATA: NeuralForecast yuklu degil",
            })
        return

    combos = list(iprod(
        grid["epochs"],
        grid["lr"],
        grid["use_log"],
        grid["input_factor"],
    ))
    total = len(combos) * len(horizons)
    done = 0

    accelerator = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

    for epochs, lr, use_log, isf in combos:
        if stop_event and stop_event.is_set():
            return

        params_str = f"epochs={epochs}, lr={lr}, log={use_log}, input_factor={isf}"

        for h_label, h_days in horizons.items():
            if h_days >= len(df_raw) - 200:
                done += 1
                continue

            # Checkpoint Kontrolu
            key = (asset_name, model_name, params_str, h_label)
            if completed_dict is not None and key in completed_dict:
                done += 1
                cached = completed_dict[key]
                if callback:
                    callback({
                        "model": model_name,
                        "params": params_str,
                        "horizon": h_label,
                        "r2": cached["r2"],
                        "mae": cached["mae"],
                        "rmse": cached["rmse"],
                        "mape": cached["mape"],
                        "elapsed": cached.get("elapsed", 0),
                        "done": done,
                        "total": total,
                        "status": "ok",
                        "is_cached": True,
                    })
                continue

            t0 = time.time()
            seed_everything(42)
            clear_memory()

            train_df = df_raw.iloc[:-h_days].copy()
            test_df = df_raw.iloc[-h_days:].copy()
            actuals = test_df["y"].values

            temp_root = tempfile.mkdtemp(prefix=f"pl_{model_name}_{threading.get_ident()}_")
            try:
                work_df = train_df[["ds", "y"]].copy()
                if use_log:
                    work_df["y"] = np.log(work_df["y"])
                work_df["unique_id"] = "1"

                input_size = int(h_days * isf)
                hard_limit = len(work_df) - h_days - 2
                input_size = min(input_size, hard_limit)
                input_size = max(input_size, 16)

                progress_cb = ModelTrainingProgressCallback(
                    callback_fn=callback,
                    model_name=model_name,
                    params_str=params_str,
                    h_label=h_label,
                    current_trial=done + 1,
                    total_trials=total,
                    max_steps=epochs,
                    t0=t0,
                )

                if callback:
                    callback({
                        "type": "step_progress",
                        "model": model_name,
                        "params": params_str,
                        "horizon": h_label,
                        "current_trial": done + 1,
                        "total": total,
                        "step": 0,
                        "max_steps": epochs,
                        "loss": None,
                        "stage": f"Eğitim Başlatılıyor ({accelerator.upper()})",
                        "trial_elapsed": round(time.time() - t0, 1),
                    })

                if model_name == "NBEATS":
                    nf_model = NBEATS(
                        input_size=input_size, h=h_days,
                        stack_types=["trend", "identity"],
                        n_blocks=[2, 1], n_polynomials=2,
                        mlp_units=2 * [[256, 256]],
                        learning_rate=lr, max_steps=epochs,
                        loss=NF_MAE(), scaler_type="robust",
                        accelerator=accelerator, batch_size=64,
                        callbacks=[progress_cb], devices=1,
                        default_root_dir=temp_root,
                        random_seed=42,
                    )
                elif model_name == "NHITS":
                    if input_size >= 64:
                        kernels, downsample = [64, 32, 2], [64, 32, 1]
                    elif input_size >= 32:
                        kernels, downsample = [16, 8, 1], [16, 8, 1]
                    else:
                        kernels, downsample = [4, 2, 1], [4, 2, 1]

                    nf_model = NHITS(
                        input_size=input_size, h=h_days,
                        n_pool_kernel_size=kernels,
                        n_freq_downsample=downsample,
                        mlp_units=[[64, 64], [64, 64]],
                        learning_rate=lr, max_steps=epochs,
                        loss=NF_MAE(), scaler_type="standard",
                        accelerator=accelerator, batch_size=64,
                        callbacks=[progress_cb], devices=1,
                        default_root_dir=temp_root,
                        random_seed=42,
                    )
                elif model_name == "PatchTST":
                    if input_size >= 48:
                        patch_len = 16
                    elif input_size >= 24:
                        patch_len = 8
                    else:
                        patch_len = 4
                        input_size = max(input_size, 8)

                    nf_model = PatchTST(
                        input_size=input_size, h=h_days,
                        patch_len=patch_len, stride=patch_len // 2,
                        n_heads=4, hidden_size=32, dropout=0.1,
                        learning_rate=lr, max_steps=epochs,
                        loss=NF_MAE(), scaler_type="standard",
                        accelerator=accelerator, batch_size=32,
                        callbacks=[progress_cb], devices=1,
                        default_root_dir=temp_root,
                        random_seed=42,
                    )
                elif model_name == "LSTM":
                    context_size = max(5, input_size // 4)
                    nf_model = LSTM(
                        input_size=input_size, h=h_days,
                        encoder_n_layers=1, decoder_layers=1,
                        encoder_hidden_size=64, decoder_hidden_size=64,
                        context_size=context_size,
                        learning_rate=lr, max_steps=epochs,
                        loss=NF_MAE(), scaler_type="standard",
                        accelerator=accelerator, batch_size=64,
                        inference_windows_batch_size=1,
                        callbacks=[progress_cb], devices=1,
                        default_root_dir=temp_root,
                        random_seed=42,
                    )
                elif model_name == "TFT":
                    nf_model = TFT(
                        input_size=input_size, h=h_days,
                        hidden_size=64, n_head=4,
                        learning_rate=lr, max_steps=epochs,
                        loss=NF_MAE(), scaler_type="standard",
                        accelerator=accelerator, batch_size=32,
                        callbacks=[progress_cb], devices=1,
                        default_root_dir=temp_root,
                        random_seed=42,
                    )
                elif model_name == "TiDE":
                    nf_model = TiDE(
                        input_size=input_size, h=h_days,
                        hidden_size=64, decoder_output_dim=16,
                        learning_rate=lr, max_steps=epochs,
                        loss=NF_MAE(), scaler_type="standard",
                        accelerator=accelerator, batch_size=32,
                        callbacks=[progress_cb], devices=1,
                        default_root_dir=temp_root,
                        random_seed=42,
                    )

                nf = NeuralForecast(models=[nf_model], freq="D")
                nf.fit(df=work_df, val_size=0)

                if callback:
                    callback({
                        "type": "step_progress",
                        "model": model_name,
                        "params": params_str,
                        "horizon": h_label,
                        "current_trial": done + 1,
                        "total": total,
                        "step": epochs,
                        "max_steps": epochs,
                        "loss": None,
                        "stage": "Tahminler Üretiliyor & Metrikler Hesaplanıyor",
                        "trial_elapsed": round(time.time() - t0, 1),
                    })

                forecast = nf.predict()

                col = [c for c in forecast.columns if c not in ["ds", "unique_id"]][0]
                preds = forecast[col].values

                if use_log:
                    preds = np.exp(preds)

                eval_len = min(len(preds), len(actuals))
                mae, rmse, r2, mape = calc_metrics(actuals[:eval_len], preds[:eval_len])
                elapsed = round(time.time() - t0, 1)

                del nf, nf_model, forecast
                done += 1
                if callback:
                    callback({
                        "model": model_name,
                        "params": params_str,
                        "horizon": h_label,
                        "r2": r2, "mae": mae, "rmse": rmse, "mape": mape,
                        "elapsed": elapsed,
                        "done": done, "total": total,
                        "status": "ok",
                    })
            except Exception as e:
                done += 1
                if callback:
                    callback({
                        "model": model_name,
                        "params": params_str,
                        "horizon": h_label,
                        "r2": -999.0, "mae": 999.0, "rmse": 999.0, "mape": 999.0,
                        "elapsed": round(time.time() - t0, 1),
                        "done": done, "total": total,
                        "status": f"HATA: {str(e)[:60]}",
                    })
            finally:
                shutil.rmtree(temp_root, ignore_errors=True)

            clear_memory()


def run_garch_grid(df_raw, model_name, grid, horizons, asset_name="Varlik",
                   completed_dict=None, callback=None, stop_event=None):
    """
    GARCH / EGARCH parametre grid search (CPU).
    """
    from arch import arch_model

    combos = list(iprod(
        grid["p"],
        grid["q"],
        grid["dist"],
        grid["mean"],
        grid["use_log"],
    ))

    total = len(combos) * len(horizons)
    done = 0
    vol_type = "EGARCH" if model_name == "EGARCH" else "GARCH"

    for combo_idx, (p, q, dist, mean_model, use_log) in enumerate(combos, 1):
        if stop_event and stop_event.is_set():
            return

        params_str = f"p={p}, q={q}, dist={dist}, mean={mean_model}, log={use_log}"

        for h_label, h_days in horizons.items():
            if h_days >= len(df_raw) - 50:
                done += 1
                continue

            key = (asset_name, model_name, params_str, h_label)
            if completed_dict is not None and key in completed_dict:
                done += 1
                cached = completed_dict[key]
                if callback:
                    callback({
                        "model": model_name,
                        "params": params_str,
                        "horizon": h_label,
                        "r2": cached["r2"],
                        "mae": cached["mae"],
                        "rmse": cached["rmse"],
                        "mape": cached["mape"],
                        "elapsed": cached.get("elapsed", 0),
                        "done": done,
                        "total": total,
                        "status": "ok",
                        "is_cached": True,
                    })
                continue

            t0 = time.time()
            seed_everything(42)

            try:
                train_prices = df_raw["y"].iloc[:-h_days].values
                actuals = df_raw["y"].iloc[-h_days:].values

                if use_log:
                    train_series = np.diff(np.log(np.maximum(train_prices, 1e-6))) * 100
                else:
                    train_series = np.diff(train_prices) / np.maximum(train_prices[:-1], 1e-6) * 100

                if callback:
                    callback({
                        "type": "step_progress",
                        "model": model_name,
                        "params": params_str,
                        "horizon": h_label,
                        "current_trial": done + 1,
                        "total": total,
                        "step": 0,
                        "max_steps": 1,
                        "loss": None,
                        "stage": f"Volatilite Fit Ediliyor ({vol_type}, Dağılım: {dist})",
                        "trial_elapsed": round(time.time() - t0, 1),
                    })

                if vol_type == "EGARCH":
                    am = arch_model(train_series, mean=mean_model, vol="EGARCH", p=p, o=1, q=q, dist=dist)
                    res = am.fit(disp="off", show_warning=False)
                    if callback:
                        callback({
                            "type": "step_progress",
                            "model": model_name,
                            "params": params_str,
                            "horizon": h_label,
                            "current_trial": done + 1,
                            "total": total,
                            "step": 1,
                            "max_steps": 1,
                            "loss": None,
                            "stage": f"Monte Carlo Simülasyonu ile Tahmin Üretiliyor ({h_days} Gün)",
                            "trial_elapsed": round(time.time() - t0, 1),
                        })
                    forecast_res = res.forecast(horizon=h_days, method="simulation", simulations=100)
                else:
                    am = arch_model(train_series, mean=mean_model, vol="GARCH", p=p, q=q, dist=dist)
                    res = am.fit(disp="off", show_warning=False)
                    if callback:
                        callback({
                            "type": "step_progress",
                            "model": model_name,
                            "params": params_str,
                            "horizon": h_label,
                            "current_trial": done + 1,
                            "total": total,
                            "step": 1,
                            "max_steps": 1,
                            "loss": None,
                            "stage": f"Analitik Varyans & Getiri Patikası Hesaplanıyor ({h_days} Gün)",
                            "trial_elapsed": round(time.time() - t0, 1),
                        })
                    forecast_res = res.forecast(horizon=h_days)

                fc_means = forecast_res.mean.iloc[-1].values

                last_train_price = train_prices[-1]
                preds = np.zeros(h_days)
                curr_price = last_train_price
                for step_idx in range(h_days):
                    ret_step = fc_means[step_idx] / 100.0
                    if use_log:
                        curr_price = curr_price * np.exp(ret_step)
                    else:
                        curr_price = curr_price * (1.0 + ret_step)
                    preds[step_idx] = curr_price

                eval_len = min(len(preds), len(actuals))
                mae, rmse, r2, mape = calc_metrics(actuals[:eval_len], preds[:eval_len])
                elapsed = round(time.time() - t0, 2)

                done += 1
                if callback:
                    callback({
                        "model": model_name,
                        "params": params_str,
                        "horizon": h_label,
                        "r2": r2, "mae": mae, "rmse": rmse, "mape": mape,
                        "elapsed": elapsed,
                        "done": done, "total": total,
                        "status": "ok",
                    })
            except Exception as e:
                done += 1
                if callback:
                    callback({
                        "model": model_name,
                        "params": params_str,
                        "horizon": h_label,
                        "r2": -999.0, "mae": 999.0, "rmse": 999.0, "mape": 999.0,
                        "elapsed": round(time.time() - t0, 2),
                        "done": done, "total": total,
                        "status": f"HATA: {str(e)[:60]}",
                    })

