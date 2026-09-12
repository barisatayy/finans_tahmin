import os
import pandas as pd
import numpy as np
from prophet import Prophet
from xgboost import XGBRegressor
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS, PatchTST, LSTM, TFT, TiDE
from neuralforecast.losses.pytorch import MAE
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import warnings
import random
import gc

warnings.filterwarnings("ignore")


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prepare_neural_data(df):
    nixtla_df = df.copy()
    nixtla_df["unique_id"] = "1"
    return nixtla_df


# --- VARLIĞA ÖZEL VARSAYILAN MİMARİ VE PARAMETRE AYARLARI (6-12 AY TAHMİN ODAKLI) ---
DEFAULT_ASSET_CONFIGS = {
    "USD/TRY": {
        "model": "Prophet",
        "imputation": "Önceki Değer (Forward Fill)",
        "use_log": False,
        "prophet_changepoint_prior_scale": 0.03,
        "prophet_changepoint_range": 0.90,
        "prophet_seasonality_mode": "additive",
        "prophet_seasonality_prior_scale": 1.0,
    },
    "EUR/TRY": {
        "model": "Prophet",
        "imputation": "Önceki Değer (Forward Fill)",
        "use_log": False,
        "prophet_changepoint_prior_scale": 0.03,
        "prophet_changepoint_range": 0.90,
        "prophet_seasonality_mode": "additive",
        "prophet_seasonality_prior_scale": 1.0,
    },
    "Altın (Ons)": {
        "model": "Prophet",
        "imputation": "Önceki Değer (Forward Fill)",
        "use_log": False,
        "prophet_changepoint_prior_scale": 0.08,
        "prophet_changepoint_range": 0.95,
        "prophet_seasonality_mode": "additive",
        "prophet_seasonality_prior_scale": 2.0,
    },
    "Bitcoin (USD)": {
        "model": "NBEATS",
        "imputation": "Önceki Değer (Forward Fill)",
        "use_log": True,
        "neural_epochs": 800,
        "neural_lr": 0.001,
    },
    "Apple": {
        "model": "Prophet",
        "imputation": "Lineer İnterpolasyon",
        "use_log": False,
        "prophet_changepoint_prior_scale": 0.03,
        "prophet_changepoint_range": 0.90,
        "prophet_seasonality_mode": "additive",
        "prophet_seasonality_prior_scale": 1.0,
    },
    "Tesla": {
        "model": "PatchTST",
        "imputation": "Lineer İnterpolasyon",
        "use_log": False,
        "neural_epochs": 700,
        "neural_lr": 0.001,
    },
    "THY": {
        "model": "XGBoost",
        "imputation": "Lineer İnterpolasyon",
        "use_log": True,
        "xgb_n_estimators": 1000,
        "xgb_learning_rate": 0.03,
        "xgb_lag_days": 14,
    },
}


def normalize_asset_name(name):
    """Farkli yazimlari (USD/TRY, USD_TRY, Altin (Ons), THY (TL) vb.) standart formata cevirir."""
    n = str(name).lower().replace(" ", "").replace("/", "").replace("_", "").replace("-", "")
    n = n.replace("ı", "i").replace("ğ", "g").replace("ü", "u").replace("ş", "s").replace("ö", "o").replace("ç", "c")
    if "btc" in n or "bitcoin" in n:
        return "btc"
    if "altin" in n or "gold" in n or "ons" in n:
        return "altin"
    if "thy" in n:
        return "thy"
    if "eur" in n:
        return "eur"
    if "usd" in n:
        return "usd"
    if "aapl" in n or "apple" in n:
        return "apple"
    if "tsla" in n or "tesla" in n:
        return "tesla"
    return n


def parse_params_str(params_str):
    """Grid search parametre metnini ('n_est=600, lr=0.08, depth=3...') Python sozlugune cevirir."""
    res = {}
    if not isinstance(params_str, str):
        return res
    pairs = [p.strip() for p in params_str.split(",") if "=" in p]
    for pair in pairs:
        k, v = pair.split("=", 1)
        k, v = k.strip(), v.strip()
        if v.lower() == "true":
            v = True
        elif v.lower() == "false":
            v = False
        else:
            try:
                if "." in v:
                    v = float(v)
                else:
                    v = int(v)
            except ValueError:
                pass

        if k == "log":
            res["use_log"] = v
        elif k == "scale":
            res["changepoint_prior_scale"] = v
            res["prophet_changepoint_prior_scale"] = v
        elif k == "range":
            res["changepoint_range"] = v
            res["prophet_changepoint_range"] = v
        elif k == "mode":
            res["seasonality_mode"] = v
            res["prophet_seasonality_mode"] = v
        elif k == "n_est":
            res["n_estimators"] = v
            res["xgb_n_estimators"] = v
        elif k == "lr":
            res["learning_rate"] = v
            res["xgb_learning_rate"] = v
            res["neural_lr"] = v
        elif k == "depth":
            res["max_depth"] = v
            res["xgb_max_depth"] = v
        elif k == "lag":
            res["lag_days"] = v
            res["xgb_lag_days"] = v
        elif k == "epochs":
            res["epochs"] = v
            res["neural_epochs"] = v
        elif k == "input_factor":
            res["input_factor"] = v
        else:
            res[k] = v
    return res


def get_best_model_from_csv(asset_name, forecast_months=6):
    """
    grid_search_en_iyi_modeller.csv dosyasindan secilen varlik ve ufuk icin
    en iyi model ve hiperparametreleri ceker.
    """
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grid_search_en_iyi_modeller.csv")
    if not os.path.exists(csv_path):
        return None

    try:
        df_best = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception:
        return None

    if df_best.empty or "Varlık" not in df_best.columns:
        return None

    # Ufuk etiketini belirle
    if forecast_months <= 4:
        target_horizon = "3 Ay"
    elif forecast_months <= 7:
        target_horizon = "6 Ay"
    else:
        target_horizon = "12 Ay"

    target_norm = normalize_asset_name(asset_name)

    matched_rows = []
    for _, row in df_best.iterrows():
        if normalize_asset_name(row["Varlık"]) == target_norm:
            matched_rows.append(row)

    if not matched_rows:
        return None

    df_matched = pd.DataFrame(matched_rows)
    horizon_rows = df_matched[df_matched["Tahmin Ufku"] == target_horizon]
    best_row = horizon_rows.iloc[0] if not horizon_rows.empty else df_matched.iloc[0]

    model_name = str(best_row["En İyi Model"]).strip()
    raw_params_str = str(best_row["En İyi Parametreler"]).strip()
    parsed_params = parse_params_str(raw_params_str)

    return {
        "model": model_name,
        "horizon": str(best_row["Tahmin Ufku"]),
        "params_str": raw_params_str,
        "parsed_params": parsed_params,
        "r2": best_row.get("R2 Skoru", None),
        "mape": best_row.get("MAPE (%)", None),
        "mae": best_row.get("MAE", None),
    }


def get_asset_config(asset_name, forecast_months=6):
    """
    Varliga ve tahmin ufkuna ozel en iyi mimari ve parametreleri doner.
    Varsayilan olarak grid_search_en_iyi_modeller.csv'den otomatik yukler.
    """
    fallback = {
        "model": "Prophet",
        "imputation": "Önceki Değer (Forward Fill)",
        "use_log": False,
        "prophet_changepoint_prior_scale": 0.05,
        "prophet_changepoint_range": 0.9,
        "prophet_seasonality_mode": "multiplicative",
        "prophet_seasonality_prior_scale": 1.0,
        "xgb_n_estimators": 1000,
        "xgb_learning_rate": 0.05,
        "xgb_lag_days": 7,
        "xgb_max_depth": 5,
        "neural_epochs": 500,
        "neural_lr": 0.001,
        "input_factor": 2.0,
        "is_optimized": False,
    }
    cfg = fallback.copy()
    if asset_name in DEFAULT_ASSET_CONFIGS:
        cfg.update(DEFAULT_ASSET_CONFIGS[asset_name])

    # 1. Grid Search CSV kontrolu
    best_csv = get_best_model_from_csv(asset_name, forecast_months)
    if best_csv:
        cfg["model"] = best_csv["model"]
        cfg["is_optimized"] = True
        cfg["grid_r2"] = best_csv["r2"]
        cfg["grid_mape"] = best_csv["mape"]
        cfg["grid_mae"] = best_csv["mae"]
        cfg["grid_horizon"] = best_csv["horizon"]
        cfg["grid_params_str"] = best_csv["params_str"]

        parsed = best_csv["parsed_params"]
        cfg["use_log"] = parsed.get("use_log", cfg.get("use_log", False))

        clean_name = best_csv["model"]
        if clean_name == "Prophet":
            if "changepoint_prior_scale" in parsed:
                cfg["prophet_changepoint_prior_scale"] = parsed["changepoint_prior_scale"]
            if "changepoint_range" in parsed:
                cfg["prophet_changepoint_range"] = parsed["changepoint_range"]
            if "seasonality_mode" in parsed:
                cfg["prophet_seasonality_mode"] = parsed["seasonality_mode"]
        elif clean_name == "XGBoost":
            if "n_estimators" in parsed:
                cfg["xgb_n_estimators"] = parsed["n_estimators"]
            if "learning_rate" in parsed:
                cfg["xgb_learning_rate"] = parsed["learning_rate"]
            if "max_depth" in parsed:
                cfg["xgb_max_depth"] = parsed["max_depth"]
            if "lag_days" in parsed:
                cfg["xgb_lag_days"] = parsed["lag_days"]
        elif clean_name in ["GARCH", "EGARCH"]:
            cfg["p"] = parsed.get("p", 1)
            cfg["q"] = parsed.get("q", 1)
            cfg["dist"] = parsed.get("dist", "Normal")
            cfg["mean"] = parsed.get("mean", "Constant")
        else:  # Neural (NBEATS, NHITS, PatchTST, LSTM, TFT, TiDE)
            if "epochs" in parsed:
                cfg["neural_epochs"] = parsed["epochs"]
            if "lr" in parsed:
                cfg["neural_lr"] = parsed["lr"]
            if "input_factor" in parsed:
                cfg["input_factor"] = parsed["input_factor"]

        cfg.update(parsed)

    return cfg


def calculate_metrics(y_true, y_pred):
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    try:
        r2 = r2_score(y_true, y_pred)
        r2 = max(-1.0, min(1.0, r2))
    except:
        r2 = 0

    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-10))) * 100
    mape = min(mape, 200.0)

    if len(y_true) > 1:
        diff_true = np.diff(y_true)
        diff_pred = np.diff(y_pred)
        da = np.mean(np.sign(diff_true) == np.sign(diff_pred)) * 100
    else:
        da = 0

    return mae, rmse, r2, mape, da


# --- PROPHET ---
def train_predict_prophet(df, horizon_days, params):
    seed_everything(42)
    clear_memory()

    scale = float(
        params.get(
            "changepoint_prior_scale",
            params.get("prophet_changepoint_prior_scale", 0.05),
        )
    )
    range_val = float(
        params.get(
            "changepoint_range",
            params.get("prophet_changepoint_range", 0.9),
        )
    )
    mode = str(
        params.get(
            "seasonality_mode",
            params.get("prophet_seasonality_mode", "additive"),
        )
    )
    prior_scale = float(
        params.get(
            "seasonality_prior_scale",
            params.get("prophet_seasonality_prior_scale", 1.0),
        )
    )
    use_log = bool(params.get("use_log", False))

    tr = df[["ds", "y"]].copy()
    if use_log:
        tr["y"] = np.log(tr["y"])

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=scale,
        changepoint_range=range_val,
        seasonality_mode=mode,
        seasonality_prior_scale=prior_scale,
    )
    model.fit(tr)
    future = model.make_future_dataframe(periods=horizon_days)
    forecast = model.predict(future)

    if use_log:
        forecast["yhat"] = np.exp(forecast["yhat"])
        if "yhat_upper" in forecast.columns:
            forecast["yhat_upper"] = np.exp(forecast["yhat_upper"])
        if "yhat_lower" in forecast.columns:
            forecast["yhat_lower"] = np.exp(forecast["yhat_lower"])

    return forecast, model


# --- XGBOOST ---
def create_xgb_features(df, lag_days=7):
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


def train_predict_xgboost(df, horizon_days, params, use_gpu=False):
    seed_everything(42)
    clear_memory()

    n_est = int(params.get("n_estimators", params.get("xgb_n_estimators", 1000)))
    lr = float(params.get("learning_rate", params.get("xgb_learning_rate", 0.05)))
    md = int(params.get("max_depth", params.get("xgb_max_depth", 5)))
    lag = int(params.get("lag_days", params.get("xgb_lag_days", 14)))
    use_log = bool(params.get("use_log", False))

    work_df = df[["ds", "y"]].copy()
    if use_log:
        work_df["y"] = np.log(work_df["y"])

    data = create_xgb_features(work_df, lag)
    if len(data) < 30:
        raise ValueError("XGBoost için en az 30 günlük veri gerekli.")

    feature_cols = [c for c in data.columns if c not in ["ds", "y", "log_ret"]]
    X = data[feature_cols].values
    y = data["log_ret"].values

    split = int(len(X) * 0.9)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

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
    if use_gpu and torch.cuda.is_available():
        try:
            model.set_params(device="cuda")
        except Exception:
            pass

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    if use_gpu and torch.cuda.is_available():
        try:
            model.set_params(device="cpu")
        except Exception:
            pass
    model._feature_names = feature_cols

    last_price = data["y"].iloc[-1]
    last_data = work_df.tail(100).copy()
    preds = []
    future_dates = pd.date_range(
        start=df["ds"].iloc[-1] + pd.Timedelta(days=1), periods=horizon_days
    )

    recent_trend = data["log_ret"].tail(30).mean()

    for idx in range(horizon_days):
        current_feats = create_xgb_features(last_data, lag)
        if len(current_feats) == 0:
            pred_ret = 0
        else:
            last_feat_row = current_feats[feature_cols].iloc[-1:].values
            pred_ret = model.predict(last_feat_row)[0]

        min_w = max(0.1, 0.5 - 0.02 * (horizon_days / 30.0))
        tw = max(min_w, 1.0 - (idx / (horizon_days * 1.3)))
        mr = 0.75
        pred_ret = (pred_ret * mr) + (recent_trend * tw * (1.0 - mr))

        next_price = last_price * np.exp(pred_ret)
        preds.append(next_price)

        new_row = pd.DataFrame({"ds": [future_dates[idx]], "y": [next_price]})
        last_data = pd.concat([last_data, new_row], ignore_index=True).tail(100)
        last_price = next_price

    if use_log:
        final_preds = np.exp(preds)
    else:
        final_preds = preds

    return pd.DataFrame({"ds": future_dates, "yhat": final_preds}), model


# --- NEURAL FORECAST ---
def get_model_instance(model_name, input_size, horizon_days, lr, epochs, accelerator):

    # --- NBEATS ---
    if model_name == "NBEATS":
        return NBEATS(
            input_size=input_size,
            h=horizon_days,
            stack_types=["trend", "identity"],
            n_blocks=[2, 1],
            n_polynomials=2,
            mlp_units=2 * [[256, 256]],
            learning_rate=lr,
            max_steps=epochs,
            loss=MAE(),
            scaler_type="robust",
            accelerator=accelerator,
            batch_size=64,
            random_seed=42,
        )

    # --- NHITS ---
    elif model_name == "NHITS":
        if input_size >= 64:
            kernels, downsample = [64, 32, 2], [64, 32, 1]
        elif input_size >= 32:
            kernels, downsample = [16, 8, 1], [16, 8, 1]
        else:
            kernels, downsample = [4, 2, 1], [4, 2, 1]

        return NHITS(
            input_size=input_size,
            h=horizon_days,
            n_pool_kernel_size=kernels,
            n_freq_downsample=downsample,
            mlp_units=[[64, 64], [64, 64]],
            learning_rate=lr,
            max_steps=epochs,
            loss=MAE(),
            scaler_type="standard",
            accelerator=accelerator,
            batch_size=64,
            random_seed=42,
        )

    # --- PatchTST ---
    elif model_name == "PatchTST":
        if input_size >= 48:
            patch_len = 16
        elif input_size >= 24:
            patch_len = 8
        else:
            patch_len = 4
            input_size = max(input_size, 8)

        return PatchTST(
            input_size=input_size,
            h=horizon_days,
            patch_len=patch_len,
            stride=patch_len // 2,
            n_heads=4,
            hidden_size=32,
            dropout=0.1,
            learning_rate=lr,
            max_steps=epochs,
            loss=MAE(),
            scaler_type="standard",
            accelerator=accelerator,
            batch_size=32,
            random_seed=42,
        )

    # --- LSTM ---
    elif model_name == "LSTM":
        context_size = max(5, input_size // 4)
        return LSTM(
            input_size=input_size,
            h=horizon_days,
            encoder_n_layers=1,
            decoder_layers=1,
            encoder_hidden_size=64,
            decoder_hidden_size=64,
            context_size=context_size,
            learning_rate=lr,
            max_steps=epochs,
            loss=MAE(),
            scaler_type="standard",
            accelerator=accelerator,
            batch_size=64,
            inference_windows_batch_size=1,
            random_seed=42,
        )
    elif model_name == "TFT":
        return TFT(
            input_size=input_size,
            h=horizon_days,
            hidden_size=64,
            n_head=4,
            learning_rate=lr,
            max_steps=epochs,
            loss=MAE(),
            scaler_type="standard",
            accelerator=accelerator,
            batch_size=32,
            random_seed=42,
        )
    elif model_name == "TiDE":
        return TiDE(
            input_size=input_size,
            h=horizon_days,
            hidden_size=64,
            decoder_output_dim=16,
            learning_rate=lr,
            max_steps=epochs,
            loss=MAE(),
            scaler_type="standard",
            accelerator=accelerator,
            batch_size=32,
            random_seed=42,
        )
    else:
        raise ValueError(f"Bilinmeyen model: {model_name}")


def train_predict_garch(df, horizon_days, model_name, params):
    """
    GARCH / EGARCH ekonometrik zaman serisi tahmini.
    """
    from arch import arch_model

    p = params.get("p", 1)
    q = params.get("q", 1)
    dist = params.get("dist", "Normal")
    mean_model = params.get("mean", "Constant")
    use_log = params.get("use_log", False)
    vol_type = "EGARCH" if model_name == "EGARCH" else "GARCH"

    train_prices = df["y"].values

    if use_log:
        train_series = np.diff(np.log(np.maximum(train_prices, 1e-6))) * 100
    else:
        train_series = np.diff(train_prices) / np.maximum(train_prices[:-1], 1e-6) * 100

    train_series = np.nan_to_num(train_series, nan=0.0, posinf=1.0, neginf=-1.0)
    hist_mean = float(np.nanmean(train_series)) if len(train_series) > 0 else 0.05
    hist_var = float(np.nanvar(train_series)) if len(train_series) > 0 else 0.01
    if hist_var <= 0:
        hist_var = 0.01

    fc_means = None
    fc_vars = None

    if vol_type == "EGARCH":
        am = arch_model(train_series, mean=mean_model, vol="EGARCH", p=p, o=1, q=q, dist=dist)
        res = am.fit(disp="off", show_warning=False)
        try:
            forecast_res = res.forecast(horizon=horizon_days, method="simulation", simulations=100)
            fc_means = forecast_res.mean.iloc[-1].values
            fc_vars = forecast_res.variance.iloc[-1].values
            if np.isnan(fc_means).any() or np.isinf(fc_means).any():
                fc_an = res.forecast(horizon=horizon_days)
                fc_means = fc_an.mean.iloc[-1].values
        except Exception:
            fc_an = res.forecast(horizon=horizon_days)
            fc_means = fc_an.mean.iloc[-1].values
            fc_vars = np.full(horizon_days, hist_var)
    else:
        am = arch_model(train_series, mean=mean_model, vol="GARCH", p=p, q=q, dist=dist)
        res = am.fit(disp="off", show_warning=False)
        forecast_res = res.forecast(horizon=horizon_days)
        fc_means = forecast_res.mean.iloc[-1].values
        fc_vars = forecast_res.variance.iloc[-1].values

    if fc_means is None or len(fc_means) < horizon_days:
        fc_means = np.full(horizon_days, hist_mean)
    if fc_vars is None or len(fc_vars) < horizon_days:
        fc_vars = np.full(horizon_days, hist_var)

    fc_means = np.where(np.isnan(fc_means) | np.isinf(fc_means), hist_mean, fc_means)
    fc_vars = np.where(np.isnan(fc_vars) | np.isinf(fc_vars) | (fc_vars < 0), hist_var, fc_vars)

    last_date = pd.to_datetime(df["ds"].iloc[-1])
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")

    last_price = float(train_prices[-1])
    preds = np.zeros(horizon_days)
    upper_bounds = np.zeros(horizon_days)
    lower_bounds = np.zeros(horizon_days)

    curr_price = last_price
    for i in range(horizon_days):
        ret_step = float(fc_means[i]) / 100.0
        ret_step = np.clip(ret_step, -0.5, 0.5)
        raw_var = max(0.0, float(fc_vars[i]))
        sigma_step = np.sqrt(raw_var) / 100.0
        sigma_step = np.clip(sigma_step, 0.0, 0.5)

        if use_log:
            curr_price = curr_price * np.exp(ret_step)
            upper_bounds[i] = curr_price * np.exp(1.96 * sigma_step)
            lower_bounds[i] = max(0.01, curr_price * np.exp(-1.96 * sigma_step))
        else:
            mult = max(0.01, 1.0 + ret_step)
            curr_price = curr_price * mult
            upper_bounds[i] = curr_price * (1.0 + 1.96 * sigma_step)
            lower_bounds[i] = max(0.01, curr_price * (1.0 - 1.96 * sigma_step))

        # Fiyatın asla 0, eksi veya NaN kalmaması için emniyet koruması
        if np.isnan(curr_price) or curr_price <= 0:
            curr_price = last_price * (1.0 + (i + 1) * (hist_mean / 100.0))

        preds[i] = curr_price

    forecast_df = pd.DataFrame({
        "ds": future_dates,
        "yhat": preds,
        "yhat_upper": upper_bounds,
        "yhat_lower": lower_bounds,
    })
    return forecast_df, None


def train_predict_neural(df, horizon_days, model_name, params, use_gpu=False):
    seed_everything(42)
    clear_memory()

    epochs = int(params.get("epochs", params.get("neural_epochs", 300)))
    lr = float(params.get("lr", params.get("neural_lr", 0.001)))
    isf = float(params.get("input_factor", 1.8))
    use_log = bool(params.get("use_log", False))

    work_df = df[["ds", "y"]].copy()
    if use_log:
        work_df["y"] = np.log(work_df["y"])
    work_df["unique_id"] = "1"

    accelerator = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

    input_size = int(horizon_days * isf)
    hard_limit_input = len(work_df) - horizon_days - 2
    if hard_limit_input < 10:
        raise RuntimeError(
            f"Veri boyutu ({len(df)}) yetersiz. Lütfen tahmin süresini düşürün."
        )

    current_input_size = min(input_size, hard_limit_input)
    current_input_size = max(current_input_size, 16)

    max_retries = 3
    attempt = 0
    last_error = None

    while attempt < max_retries:
        try:
            clear_memory()

            model_instance = get_model_instance(
                model_name, current_input_size, horizon_days, lr, epochs, accelerator
            )

            nf = NeuralForecast(models=[model_instance], freq="D")
            nf.fit(df=work_df, val_size=0)
            forecast = nf.predict()

            del nf
            del model_instance
            clear_memory()

            col = [c for c in forecast.columns if c not in ["ds", "unique_id"]][0]
            forecast = forecast.rename(columns={col: "yhat"})
            if use_log:
                forecast["yhat"] = np.exp(forecast["yhat"])
            return forecast[["ds", "yhat"]].reset_index(drop=True), None

        except Exception as e:
            last_error = e
            error_str = str(e).lower()

            if (
                "pop from empty list" in error_str
                or "size" in error_str
                or "memory" in error_str
            ):
                attempt += 1
                new_size = int(current_input_size * 0.8)

                if new_size < 10:
                    break

                current_input_size = new_size
                print(
                    f"Hata alındı, input_size {current_input_size}'e düşürülüp tekrar deneniyor..."
                )
            else:
                raise e

    clear_memory()
    raise RuntimeError(
        f"Model eğitimi {max_retries} kez denendi ancak başarısız oldu.\n"
        f"Son Hata: {last_error}\n"
        "Lütfen daha kısa bir tahmin süresi seçin."
    )


# --- VARLIĞA ÖZEL TAHMİN MİMARİSİ ALTYAPISI ---
ASSET_SPECIFIC_HANDLERS = {}


def predict_for_asset(
    selected_asset, model_option, train_df, horizon_days, params, use_gpu=False
):
    """
    Seçilen varlık için özel bir mimari handler'ı tanımlanmışsa onu çalıştırır,
    aksi halde standart model tahmin fonksiyonlarına yönlendirir.
    """
    if selected_asset in ASSET_SPECIFIC_HANDLERS:
        handler = ASSET_SPECIFIC_HANDLERS[selected_asset]
        return handler(train_df, horizon_days, params, use_gpu)

    clean_model_name = model_option.split(" ")[0]
    if clean_model_name == "Prophet":
        return train_predict_prophet(train_df, horizon_days, params)
    elif clean_model_name == "XGBoost":
        return train_predict_xgboost(train_df, horizon_days, params, use_gpu)
    elif clean_model_name in ["GARCH", "EGARCH"]:
        return train_predict_garch(train_df, horizon_days, clean_model_name, params)
    else:  # Neural (NBEATS, NHITS, PatchTST, LSTM, TFT, TiDE)
        return train_predict_neural(
            train_df, horizon_days, clean_model_name, params, use_gpu
        )
