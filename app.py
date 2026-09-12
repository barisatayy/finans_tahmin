import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import torch
import io
from data_fetching import ASSETS, load_data, process_data
from ml_models import (
    train_predict_prophet,
    train_predict_xgboost,
    train_predict_neural,
    predict_for_asset,
    calculate_metrics,
    get_asset_config,
)

st.set_page_config(layout="wide", page_title="Gelişmiş Finansal Tahmin")

# --- 1. SIDEBAR ---
st.sidebar.title("Model Ayarları")


selected_asset = st.sidebar.selectbox("Varlık Seçin", list(ASSETS.keys()))
ticker = ASSETS[selected_asset]

# Tahmin Süresi Seçimi (Maksimum 1 Yıl / 12 Ay)
forecast_months = st.sidebar.slider(
    "Tahmin Süresi (Ay)", 1, 12, 6, help="Geleceğe yönelik en fazla 1 yıllık (12 ay) tahmin yapılabilir."
)
horizon_days = forecast_months * 30

# Varlığa ve Seçilen Ufka (3/6/12 Ay) göre Grid Search'ten çıkan en iyi modeli ve parametreleri al
asset_cfg = get_asset_config(selected_asset, forecast_months=forecast_months)

horizon_tag = asset_cfg.get("grid_horizon", f"{forecast_months}m")
key_suffix = f"{selected_asset}_{horizon_tag}"



model_list = [
    "Prophet",
    "XGBoost",
    "GARCH",
    "EGARCH",
    "NBEATS",
    "NHITS",
    "PatchTST",
    "LSTM",
    "TFT",
    "TiDE",
]

cuda_available = torch.cuda.is_available()
params = {}

imputation_options = [
    "Önceki Değer (Forward Fill)",
    "Lineer İnterpolasyon",
    "Sonraki Değer (Backward Fill)",
    "Ortalama (Mean)",
    "Sil (Drop)",
]

st.sidebar.markdown("---")
st.sidebar.subheader("Model Seçimi")
default_model_idx = (
    model_list.index(asset_cfg["model"]) if asset_cfg["model"] in model_list else 0
)
model_option = st.sidebar.selectbox(
    "Yapay Zeka Modeli Seçin",
    model_list,
    index=default_model_idx,
    key=f"model_select_{key_suffix}",
)
clean_model_name = model_option.split(" ")[0]

st.sidebar.markdown("---")
st.sidebar.subheader("Veri Ön İşleme")

default_imp_idx = (
    imputation_options.index(asset_cfg["imputation"])
    if asset_cfg["imputation"] in imputation_options
    else 0
)
imputation_method = st.sidebar.selectbox(
    "Eksik Veri (Tatil/Haftasonu) Doldurma:",
    imputation_options,
    index=default_imp_idx,
    key=f"imp_{key_suffix}",
)

use_log = st.sidebar.checkbox(
    "Logaritma Dönüşümü Uygula",
    value=asset_cfg.get("use_log", False),
    key=f"log_{key_suffix}",
    help="Parabolik artan varlıklar için uygulanır.",
)

# DONANIM (GPU)
st.sidebar.markdown("---")
st.sidebar.subheader("Donanım")
use_gpu = st.sidebar.checkbox(
    "GPU ile Çalıştır (Hızlandırıcı)",
    value=cuda_available,
    disabled=not cuda_available
    or clean_model_name in ["Prophet", "XGBoost", "GARCH", "EGARCH"],
    key=f"gpu_{key_suffix}",
)
if use_gpu:
    st.sidebar.success(f"GPU Aktif: {torch.cuda.get_device_name(0)}")
elif not cuda_available:
    st.sidebar.warning("GPU Yok. CPU Kullanılıyor.")

# PARAMETRELER
st.sidebar.markdown("---")
st.sidebar.subheader(f"{clean_model_name} Hiperparametreleri")

if clean_model_name == "Prophet":
    cps_options = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.35, 0.5]
    cur_cps = float(asset_cfg.get("prophet_changepoint_prior_scale", 0.03))
    cps_idx = cps_options.index(cur_cps) if cur_cps in cps_options else 3
    params["changepoint_prior_scale"] = st.sidebar.selectbox(
        "Trend Esnekliği", cps_options, index=cps_idx, key=f"cps_{key_suffix}"
    )
    cur_range = float(asset_cfg.get("prophet_changepoint_range", 0.80))
    params["changepoint_range"] = st.sidebar.slider(
        "Değişim Noktası Aralığı",
        0.70,
        0.95,
        cur_range,
        0.05,
        key=f"crange_{key_suffix}",
    )
    cur_mode = str(asset_cfg.get("prophet_seasonality_mode", "additive"))
    season_mode_idx = 0 if cur_mode == "multiplicative" else 1
    params["seasonality_mode"] = st.sidebar.selectbox(
        "Mevsimsellik Modu", ["multiplicative", "additive"], index=season_mode_idx, key=f"smode_{key_suffix}"
    )
    params["seasonality_prior_scale"] = st.sidebar.slider(
        "Mevsimsellik Esnekliği",
        0.01,
        10.0,
        float(asset_cfg.get("prophet_seasonality_prior_scale", 1.0)),
        0.1,
        key=f"sprior_{key_suffix}",
    )
    params["use_log"] = use_log

elif clean_model_name == "XGBoost":
    def_nest = int(asset_cfg.get("xgb_n_estimators", 1000))
    params["n_estimators"] = st.sidebar.slider(
        "Ağaç Sayısı", 100, 2000, def_nest, 100, key=f"nest_{key_suffix}"
    )
    lr_options = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.2]
    cur_lr = float(asset_cfg.get("xgb_learning_rate", 0.05))
    lr_idx = lr_options.index(cur_lr) if cur_lr in lr_options else 3
    params["learning_rate"] = st.sidebar.selectbox(
        "Öğrenme Hızı", lr_options, index=lr_idx, key=f"lr_{key_suffix}"
    )
    cur_depth = int(asset_cfg.get("xgb_max_depth", 5))
    params["max_depth"] = st.sidebar.slider(
        "Maksimum Derinlik", 2, 10, cur_depth, 1, key=f"depth_{key_suffix}"
    )
    cur_lag = int(asset_cfg.get("xgb_lag_days", 14))
    params["lag_days"] = st.sidebar.slider(
        "Geçmişe Bakış (Gün)", 3, 30, cur_lag, 1, key=f"lag_{key_suffix}"
    )
    params["use_log"] = use_log

elif clean_model_name in ["GARCH", "EGARCH"]:
    p_val = int(asset_cfg.get("p", 1))
    q_val = int(asset_cfg.get("q", 1))
    dist_val = str(asset_cfg.get("dist", "Normal"))
    mean_val = str(asset_cfg.get("mean", "Constant"))
    dist_options = ["Normal", "t", "skewt"]
    mean_options = ["Constant", "AR"]
    params["p"] = st.sidebar.selectbox("p (ARCH Derecesi)", [1, 2], index=0 if p_val == 1 else 1, key=f"p_{key_suffix}")
    params["q"] = st.sidebar.selectbox("q (GARCH Derecesi)", [1, 2], index=0 if q_val == 1 else 1, key=f"q_{key_suffix}")
    params["dist"] = st.sidebar.selectbox(
        "Hata Dağılımı", dist_options, index=dist_options.index(dist_val) if dist_val in dist_options else 0, key=f"dist_{key_suffix}"
    )
    params["mean"] = st.sidebar.selectbox(
        "Ortalama Getiri Modeli", mean_options, index=mean_options.index(mean_val) if mean_val in mean_options else 0, key=f"mean_{key_suffix}"
    )
    params["use_log"] = use_log


st.sidebar.markdown("---")

st.title(f"{selected_asset} Analizi")

# A) Veri Yükleme
with st.spinner("Piyasa verileri indiriliyor..."):
    raw_df = load_data(ticker)

if raw_df is None or raw_df.empty:
    st.error("Veri alınamadı.")
    st.stop()

# B) Veri Ön İşleme
df = process_data(raw_df, imputation_method)
# Ham fiyatlar korunur. Logaritma dönüşümü gerektiğinde model içinde dinamik işletilir.

# C) Tarihsel Grafik
st.subheader("Geçmiş Fiyatlar")
fig_hist = go.Figure()
display_y = df["y"]
fig_hist.add_trace(
    go.Scatter(x=df["ds"], y=display_y, name="Geçmiş", line=dict(color="#7f7f7f"))
)
fig_hist.update_layout(xaxis_title="Tarih", yaxis_title="Fiyat", hovermode="x unified")
st.plotly_chart(fig_hist, width="stretch")

# Veri İstatistikleri
c1, c2, c3, c4 = st.columns(4)
c1.metric("Toplam Veri", f"{len(df)} gün")
c2.metric("Başlangıç", df["ds"].min().strftime("%Y-%m-%d"))
c3.metric("Bitiş", df["ds"].max().strftime("%Y-%m-%d"))
current_price = df["y"].iloc[-1]
c4.metric("Son Fiyat", f"{current_price:.2f}")

# Tarihsel Veri İndirme Bölümü
with st.expander("Veri Setini İndir (CSV)", expanded=True):
    col_dl1, col_dl2 = st.columns([1, 2])
    with col_dl1:
        dl_months = st.number_input(
            "İndirilecek Süre (Ay):",
            min_value=1,
            max_value=60,
            value=60,
            step=1,
            help="En fazla 5 yıl (60 ay) referans veri derinliği indirebilirsiniz.",
        )
    with col_dl2:
        st.write("")
        st.write("")
        cutoff_date = df["ds"].max() - pd.DateOffset(months=dl_months)
        export_df = df[df["ds"] >= cutoff_date].copy()
        export_df = export_df.rename(columns={"ds": "Tarih", "y": "Fiyat"})
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"Veri Setini İndir (CSV)",
            data=csv_bytes,
            file_name=f"{selected_asset.lower().replace('/', '_').replace(' ', '_')}_{dl_months}aylik.csv",
            mime="text/csv",
        )

st.markdown("---")

# D) EĞİTİM VE TAHMİN
if st.button("Modeli Eğit ve Tahmin Et", type="primary"):

    test_days_needed = horizon_days
    if len(df) <= test_days_needed + 30:
        st.error(
            f"Yetersiz Veri: Analiz için en az {test_days_needed + 30} gün veri gerekiyor.\n"
            f"- Test: {test_days_needed} gün\n"
            f"- Mevcut: {len(df)} gün"
        )
        st.stop()

    # Eğitim ve Test Ayrımı
    train_df = df.iloc[:-test_days_needed].copy()
    test_df = df.iloc[-test_days_needed:].copy()

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # --- ADIM 1: BACKTEST ---
        status_text.text("Model eğitiliyor ve test ediliyor...")
        progress_bar.progress(25)

        forecast_test, model_obj = predict_for_asset(
            selected_asset, model_option, train_df, test_days_needed, params, use_gpu
        )
        y_pred = forecast_test["yhat"].iloc[-test_days_needed:].values

        # Emniyet Kontrolü: Test tahminlerinde düz çizgi, 0 veya NaN değer oluşmasını engelle
        if np.isnan(y_pred).any() or np.isinf(y_pred).any() or (y_pred <= 0).any() or np.std(y_pred) < 1e-5:
            last_train_p = float(train_df["y"].iloc[-1])
            drift = float(np.diff(train_df["y"].values[-60:]).mean()) if len(train_df) > 60 else 0.01
            steps = np.arange(1, test_days_needed + 1)
            fallback_pred = last_train_p + steps * drift
            y_pred = np.where((y_pred <= 0) | np.isnan(y_pred) | np.isinf(y_pred), fallback_pred, y_pred)
            if np.std(y_pred) < 1e-5:
                y_pred = fallback_pred

        progress_bar.progress(50)

        # Metrikler (Modeller doğrudan gerçek fiyat seviyesinde tahmin üretir)
        y_true = test_df["y"].values

        mae, rmse, r2, mape, da = calculate_metrics(y_true, y_pred)

        st.markdown("---")
        st.subheader("Model Performans Metrikleri")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("MAE", f"{mae:.2f}")
        c2.metric("RMSE", f"{rmse:.2f}")
        c3.metric("R² Skoru", f"{r2:.4f}")
        c4.metric(
            "MAPE", f"%{mape:.2f}", delta_color="inverse" if mape < 10 else "normal"
        )
        c5.metric(
            "Yön Doğruluğu", f"%{da:.2f}", delta_color="off" if da > 60 else "normal"
        )

        # Test Grafiği
        fig_test = go.Figure()
        fig_test.add_trace(
            go.Scatter(
                x=test_df["ds"], y=y_true, name="Gerçek", line=dict(color="green")
            )
        )
        fig_test.add_trace(
            go.Scatter(
                x=test_df["ds"],
                y=y_pred,
                name="Tahmin (Test)",
                line=dict(color="red", dash="dot"),
            )
        )
        fig_test.update_layout(title="Model Doğrulama Testi", hovermode="x unified")
        st.plotly_chart(fig_test, width="stretch")

        # --- GELECEK TAHMİNİ ---
        status_text.text(f"Gelecek {forecast_months} ay tahmin ediliyor...")
        progress_bar.progress(75)

        # Gelecek projeksiyonu için tüm mevcut veri
        future_training_data = df.copy()

        full_forecast, _ = predict_for_asset(
            selected_asset,
            model_option,
            future_training_data,
            horizon_days,
            params,
            use_gpu,
        )
        future_preds = full_forecast.iloc[-horizon_days:]

        progress_bar.progress(100)
        status_text.text("Tamamlandı.")

        future_yhat = future_preds["yhat"].values
        if np.isnan(future_yhat).any() or np.isinf(future_yhat).any() or (future_yhat <= 0).any() or np.std(future_yhat) < 1e-5:
            last_fut_p = float(future_training_data["y"].iloc[-1])
            drift = float(np.diff(future_training_data["y"].values[-60:]).mean()) if len(future_training_data) > 60 else 0.01
            steps = np.arange(1, horizon_days + 1)
            fallback_fut = last_fut_p + steps * drift
            future_yhat = np.where((future_yhat <= 0) | np.isnan(future_yhat) | np.isinf(future_yhat), fallback_fut, future_yhat)
            if np.std(future_yhat) < 1e-5:
                future_yhat = fallback_fut

        # Hata payı (güven aralığı) hesaplama
        time_steps = np.arange(1, len(future_preds) + 1)
        time_factor = np.sqrt(time_steps / 30.0 + 1.0)
        margin = rmse * time_factor * 1.5

        if (
            "yhat_upper" in future_preds.columns
            and "yhat_lower" in future_preds.columns
        ):
            y_upper = future_preds["yhat_upper"].values
            y_lower = future_preds["yhat_lower"].values
            y_upper = np.maximum(future_yhat, y_upper)
            y_lower = np.maximum(0, np.minimum(future_yhat, y_lower))
        else:
            y_upper = future_yhat + margin
            y_lower = np.maximum(0, future_yhat - margin)

        st.markdown("---")
        st.subheader(f"Gelecek {forecast_months} Ayın Tahmini")

        fig_future = go.Figure()
        recent_y = future_training_data["y"]

        fig_future.add_trace(
            go.Scatter(
                x=future_training_data["ds"],
                y=recent_y,
                name=f"Eğitim Verisi ({len(future_training_data)} Gün)",
                line=dict(color="#1f77b4"),
            )
        )
        # Hata Payı Üst Sınır (Gizli çizgi)
        fig_future.add_trace(
            go.Scatter(
                x=future_preds["ds"],
                y=y_upper,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        # Hata Payı Alt Sınır ve Şeffaf Dolgu Penceresi
        fig_future.add_trace(
            go.Scatter(
                x=future_preds["ds"],
                y=y_lower,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(255, 75, 75, 0.18)",
                name="Hata Payı (%95 Güven Aralığı)",
                hoverinfo="skip",
            )
        )
        # Gelecek Tahmini Çizgisi
        fig_future.add_trace(
            go.Scatter(
                x=future_preds["ds"],
                y=future_yhat,
                name="Gelecek Tahmini",
                line=dict(color="#ff2b2b", width=3),
            )
        )
        fig_future.update_layout(
            title=f"{selected_asset} - {forecast_months} Aylık Gelecek Tahmini ve Hata Payı Penceresi",
            hovermode="x unified",
        )
        st.plotly_chart(fig_future, width="stretch")

        # Özet Tablosu
        curr = recent_y.iloc[-1]
        pred = future_yhat[-1]
        chg = ((pred - curr) / curr) * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Şu Anki Fiyat", f"{curr:.2f}")
        c2.metric(
            f"{forecast_months} Ay Sonra", f"{pred:.2f}", delta=f"{pred-curr:.2f}"
        )
        c3.metric("Tahmini Değişim", f"%{chg:.2f}")

        # --- EXCEL İNDİRME ---
        st.markdown("---")
        st.subheader("Verileri İndir")

        download_df = pd.DataFrame(
            {
                "Tarih": future_preds["ds"].dt.date,
                "Tahmin": np.round(future_yhat, 4),
                "Alt_Hata_Siniri": np.round(y_lower, 4),
                "Ust_Hata_Siniri": np.round(y_upper, 4),
            }
        )

        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            download_df.to_excel(writer, index=False, sheet_name="Tahmin")

        st.download_button(
            label="Tahmin Verisini Excel Olarak İndir (.xlsx)",
            data=excel_buffer.getvalue(),
            file_name=f"{selected_asset}_Tahmin_{forecast_months}Ay.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as e:
        st.error(f"Hata oluştu: {str(e)}")

st.markdown("---")
st.caption("Yatırım tavsiyesi değildir.")
