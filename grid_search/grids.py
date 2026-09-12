"""
Parametre grid tanimlari - Tum modeller icin.
Her model icin anlamli araliklar, ~%10 atlamalar.
"""

# ============================================================
#  VARLIK VE VERİ DOSYASI HARİTASI
# ============================================================

ASSETS = {
    "USD/TRY": "usd_try_60aylik.csv",
    "EUR/TRY": "eur_try_60aylik.csv",
    "Altin (Ons)": "altın_(ons)_60aylik.csv",
    "Bitcoin (USD)": "bitcoin_(usd)_60aylik.csv",
    "Apple (USD)": "apple_60aylik.csv",
    "Tesla (USD)": "tesla_60aylik.csv",
    "THY (TL)": "thy_60aylik.csv",
}

# ============================================================
#  TAHMİN UFUKLARİ
# ============================================================

HORIZONS = {
    "3 Ay": 90,
    "6 Ay": 180,
    "12 Ay": 360,
}

# ============================================================
#  PROPHET PARAMETRELERİ
# ============================================================
# changepoint_prior_scale: Trend esnekligi (log olcek)
#   Dusuk (0.005): Kati, dogrusal trend. Gürültüye duyarsiz.
#   Yuksek (0.4): Esnek, her dalgayi takip eder. Overfitting riski.
# changepoint_range: Verinin yuzde kacinda kirilma aranir
#   0.80: Sadece ilk %80. Son %20'deki trend degisimlerini kacirabilir.
#   0.95: Neredeyse tum veri. Son trendleri yakalar ama gürültüye hassas.
# seasonality_mode: Mevsimsellik türü
#   additive: Sabit mevsimsellik. Düz trendlerde iyi.
#   multiplicative: Oransal mevsimsellik. Yükselen/düşen trendlerde iyi.
# use_log: Fiyata log donusumu. Üstel trendlerde (TRY gibi) faydalı.

PROPHET_GRID = {
    "changepoint_prior_scale": [0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.4],
    "changepoint_range": [0.80, 0.90, 0.95],
    "seasonality_mode": ["additive", "multiplicative"],
    "use_log": [False, True],
}

# ============================================================
#  XGBOOST PARAMETRELERİ
# ============================================================
# n_estimators: Agac sayisi.
#   Az (300): Hizli, underfitting riski.
#   Cok (1000): Yavas, daha iyi generalizasyon ama overfitting riski.
# learning_rate: Ogrenme hizi (log olcek).
#   Dusuk (0.01): Hassas, n_estimators'in yüksek olmasi lazim.
#   Yuksek (0.08): Hizli yakinlasma, kararsizlik riski.
# max_depth: Agac derinligi.
#   Az (3): Basit model, bias yuksek.
#   Cok (7): Karmasik model, variance yuksek.
# lag_days: Gecikme penceresi.
#   Az (7): Kisa hafiza, hizli tepki.
#   Cok (21): Uzun hafiza, yavas tepki.
# use_log: Log donusumu.

XGBOOST_GRID = {
    "n_estimators": [300, 600, 1000],
    "learning_rate": [0.01, 0.03, 0.08],
    "max_depth": [3, 5, 7],
    "lag_days": [7, 14, 21],
    "use_log": [False, True],
}

# ============================================================
#  NEURAL MODEL PARAMETRELERİ (NBEATS, NHITS, PatchTST, LSTM, TFT, TiDE)
# ============================================================
# epochs (max_steps): Egitim adim sayisi.
#   Az (150): Hizli, underfitting riski.
#   Cok (300): Daha iyi ogrenme, overfitting riski.
# lr: Ogrenme hizi.
#   Dusuk (0.0005): Yavaş ama stabil yakinsama.
#   Yuksek (0.003): Hizli ama patlama riski.
# use_log: Log donusumu.
# input_factor: input_size = horizon * factor.
#   Dusuk (1.5): Kisa bakis penceresi.
#   Yuksek (2.5): Uzun bakis penceresi, daha fazla tarihsel bilgi.

NEURAL_MODELS = ["NBEATS", "NHITS", "PatchTST", "LSTM", "TFT", "TiDE"]

NEURAL_GRID = {
    "epochs": [150, 300],
    "lr": [0.0005, 0.001, 0.003],
    "use_log": [False, True],
    "input_factor": [1.5, 2.5],
}

# ============================================================
#  EKONOMETRİK MODEL PARAMETRELERİ (GARCH, EGARCH)
# ============================================================
# p: ARCH gecikme derecesi (şok etkisi)
# q: GARCH gecikme derecesi (volatilite kalıcılığı)
# dist: Hata dağılımı (Normal, Student-t, Skewed Student-t)
# mean: Ortalama getiri modeli (Constant, AR-1)
# use_log: Logaritmik getiri dönüşümü

GARCH_MODELS = ["GARCH", "EGARCH"]

GARCH_GRID = {
    "p": [1, 2],
    "q": [1, 2],
    "dist": ["Normal", "t", "skewt"],
    "mean": ["Constant", "AR"],
    "use_log": [False, True],
}


def count_combos():
    """Her model icin kombinasyon sayisini hesaplar."""
    from itertools import product as iprod

    prophet_n = len(list(iprod(*PROPHET_GRID.values())))
    xgb_n = len(list(iprod(*XGBOOST_GRID.values())))
    neural_n = len(list(iprod(*NEURAL_GRID.values())))
    garch_n = len(list(iprod(*GARCH_GRID.values())))

    h = len(HORIZONS)
    res = {
        "Prophet": prophet_n * h,
        "XGBoost": xgb_n * h,
        "GARCH": garch_n * h,
        "EGARCH": garch_n * h,
    }
    for nm in NEURAL_MODELS:
        res[nm] = neural_n * h
    return res


if __name__ == "__main__":
    counts = count_combos()
    total = sum(counts.values())
    print("Kombinasyon sayilari (varlik basina):")
    for m, c in counts.items():
        print(f"  {m}: {c}")
    print(f"  TOPLAM: {total}")
    print(f"  7 varlik: {total * 7}")

