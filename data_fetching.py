import streamlit as st
import yfinance as yf
import pandas as pd

ASSETS = {
    "USD/TRY": "TRY=X",
    "EUR/TRY": "EURTRY=X",
    "Altın (Ons)": "GC=F",
    "Bitcoin (USD)": "BTC-USD",
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "THY": "THYAO.IS",
}

LOCAL_FILE_MAP = {
    "USD/TRY": "usd_try_60aylik.csv",
    "EUR/TRY": "eur_try_60aylik.csv",
    "Altın (Ons)": "altın_(ons)_60aylik.csv",
    "Bitcoin (USD)": "bitcoin_(usd)_60aylik.csv",
    "Apple": "apple_60aylik.csv",
    "Tesla": "tesla_60aylik.csv",
    "THY": "thy_60aylik.csv",
}


def load_local_data(asset_name):
    """datas/ klasorundeki grid search 60 aylik referans CSV dosyasini yukler."""
    import os
    datas_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datas")
    
    # Isim normalizasyonu ve esleme (Bitcoin (USD) icindeki 'USD'den dolayi once bitcoin kontrol edilmeli)
    an_lower = str(asset_name).lower()
    if "btc" in an_lower or "bitcoin" in an_lower:
        filename = "bitcoin_(usd)_60aylik.csv"
    elif "alt" in an_lower or "ons" in an_lower or "gold" in an_lower:
        filename = "altın_(ons)_60aylik.csv"
    elif "eur" in an_lower:
        filename = "eur_try_60aylik.csv"
    elif "usd" in an_lower:
        filename = "usd_try_60aylik.csv"
    elif "thy" in an_lower:
        filename = "thy_60aylik.csv"
    elif "appl" in an_lower or "apple" in an_lower:
        filename = "apple_60aylik.csv"
    elif "tsla" in an_lower or "tesla" in an_lower:
        filename = "tesla_60aylik.csv"
    else:
        filename = LOCAL_FILE_MAP.get(asset_name)

    if not filename:
        return None
    file_path = os.path.join(datas_dir, filename)
    if not os.path.exists(file_path):
        return None

    df = pd.read_csv(file_path)
    df["ds"] = pd.to_datetime(df["Tarih"])
    df["y"] = pd.to_numeric(df["Fiyat"], errors="coerce")
    df = df.sort_values("ds").dropna().reset_index(drop=True)
    return df[["ds", "y"]]


@st.cache_data
def load_data(ticker):
    # Grid Search 60 aylik (5 yillik) referans veri derinligi ile birebir esitlenmistir
    data = yf.download(ticker, period="5y")
    data.reset_index(inplace=True)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    target = "Adj Close" if "Adj Close" in data.columns else "Close"
    df = data[["Date", target]].rename(columns={"Date": "ds", target: "y"})

    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)

    full_range = pd.date_range(df["ds"].min(), df["ds"].max(), freq="D")
    df = df.set_index("ds").reindex(full_range).reset_index()
    df.columns = ["ds", "y"]

    # Tam 60 aylik (en fazla 1827 gun) referans cercevesi
    if len(df) > 1827:
        df = df.tail(1827).reset_index(drop=True)

    return df


def process_data(df, method):
    df = df.copy()

    if method == "Önceki Değer (Forward Fill)":
        df["y"] = df["y"].ffill()
    elif method == "Sonraki Değer (Backward Fill)":
        df["y"] = df["y"].bfill()
    elif method == "Lineer İnterpolasyon":
        df["y"] = df["y"].interpolate()
    elif method == "Ortalama (Mean)":
        df["y"] = df["y"].fillna(df["y"].mean())
    elif method == "Sil (Drop)":
        df = df.dropna()

    return df.dropna()
