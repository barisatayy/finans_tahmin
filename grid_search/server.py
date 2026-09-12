"""
Flask + SocketIO backend server.
Worker'lari thread pool'da calistirir, sonuclari WebSocket ile frontend'e iletir.
Ayni anda Prophet/XGBoost CPU'da, Neural modeller GPU'da calisir.
"""

import os
import sys
import json
import time
import threading
from datetime import datetime
from collections import defaultdict

import psutil
import torch
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO

# grid_search paketinden importlar
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grids import (
    ASSETS,
    HORIZONS,
    PROPHET_GRID,
    XGBOOST_GRID,
    NEURAL_GRID,
    NEURAL_MODELS,
    GARCH_MODELS,
    GARCH_GRID,
    count_combos,
)
from worker import (
    load_data,
    run_prophet_grid,
    run_xgboost_grid,
    run_neural_grid,
    run_garch_grid,
)

app = Flask(__name__)
app.config["SECRET_KEY"] = "grid-search-2026"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ============================================================
#  GLOBAL STATE
# ============================================================

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datas"
)
CHECKPOINT_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "grid_search_tum_denemeler.csv",
)
BEST_CSV_OUTPUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "grid_search_en_iyi_modeller.csv",
)

# Her varlik icin durum
state = {
    "started": False,
    "start_time": None,
    "assets": {},
}

# Her varlik icin en iyi sonuclar (ufuk bazli)
best_results = {}  # { asset_name: { horizon: { r2, model, params, mae, rmse, mape } } }

# Her varlik icin tum sonuclar
all_results = []  # [ { asset, model, params, horizon, r2, mae, rmse, mape, elapsed } ]
completed_runs = {}  # { (asset, model, params, horizon): row_dict }
results_lock = threading.Lock()
checkpoint_lock = threading.Lock()

# Aktif thread'ler
active_threads = {}
stop_events = {}

# Toplam deney sayilari
COMBO_COUNTS = count_combos()
TOTAL_PER_ASSET = sum(COMBO_COUNTS.values())
TOTAL_ALL = TOTAL_PER_ASSET * len(ASSETS)

MODEL_ORDER = [
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

# GPU Eszamanlilik Limiti (RTX 3070 8 GB VRAM icin ayni anda maksimum 2 derin ogrenme modeli)
# Boylece toplam VRAM ~4.5 - 5.5 GB seviyesinde kalir; RAM'e tasma (paging / thrashing) tamamen onlenir.
GPU_CONCURRENCY_LIMIT = 2
gpu_semaphore = threading.Semaphore(GPU_CONCURRENCY_LIMIT)


def init_state():
    """Baslangic durumunu olusturur."""
    for asset_name in ASSETS:
        state["assets"][asset_name] = {
            "status": "bekliyor",  # bekliyor, calisiyor, tamamlandi
            "current_model": None,
            "done": 0,
            "total": TOTAL_PER_ASSET,
            "models": {
                m: {
                    "done": 0,
                    "total": COMBO_COUNTS[m],
                    "status": "bekliyor",
                    "current_trial": 0,
                    "current_params": "",
                    "current_horizon": "",
                    "step": 0,
                    "max_steps": 0,
                    "loss": None,
                    "stage": "Bekliyor",
                    "trial_elapsed": 0.0,
                    "logs": [],
                }
                for m in MODEL_ORDER
            },
        }
        best_results[asset_name] = {h: None for h in HORIZONS}


init_state()


def load_checkpoint():
    """Mevcut checkpoint CSV dosyasini yukler ve hafizayi gunceller."""
    if not os.path.exists(CHECKPOINT_CSV):
        with open(CHECKPOINT_CSV, "w", encoding="utf-8-sig") as f:
            f.write("asset,model,params,horizon,r2,mae,rmse,mape,elapsed,timestamp\n")
        return

    import csv

    try:
        with open(CHECKPOINT_CSV, "r", encoding="utf-8-sig", errors="ignore") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            count = 0
            for row in reader:
                if len(row) < 8:
                    continue
                try:
                    asset = row[0].strip('"')
                    model = row[1].strip('"')
                    params = row[2].strip('"')
                    horizon = row[3].strip('"')
                    r2 = float(row[4])
                    mae = float(row[5])
                    rmse = float(row[6])
                    mape = float(row[7])
                    elapsed = float(row[8]) if len(row) > 8 and row[8] else 0.0

                    row_dict = {
                        "asset": asset,
                        "model": model,
                        "params": params,
                        "horizon": horizon,
                        "r2": r2,
                        "mae": mae,
                        "rmse": rmse,
                        "mape": mape,
                        "elapsed": elapsed,
                    }

                    key = (asset, model, params, horizon)
                    if key not in completed_runs:
                        completed_runs[key] = row_dict
                        count += 1

                        with results_lock:
                            all_results.append(row_dict)

                        # Update best results
                        if asset in best_results and horizon in best_results[asset]:
                            cur_best = best_results[asset][horizon]
                            if cur_best is None or (r2 > -900 and r2 > cur_best["r2"]):
                                best_results[asset][horizon] = {
                                    "r2": r2,
                                    "model": model,
                                    "params": params,
                                    "mae": mae,
                                    "rmse": rmse,
                                    "mape": mape,
                                }

                        # Update asset state counts
                        if (
                            asset in state["assets"]
                            and model in state["assets"][asset]["models"]
                        ):
                            m_state = state["assets"][asset]["models"][model]
                            m_state["done"] += 1
                            state["assets"][asset]["done"] += 1
                            if m_state["done"] >= m_state["total"]:
                                m_state["status"] = "tamamlandi"
                                m_state["stage"] = "Tamamlandı"
                            if (
                                state["assets"][asset]["done"]
                                >= state["assets"][asset]["total"]
                            ):
                                state["assets"][asset]["status"] = "tamamlandi"
                except Exception:
                    continue
            print(f"Checkpoint yuklendi: {count} benzersiz kayit hafizaya alindi.")
    except Exception as e:
        print(f"Checkpoint yukleme hatasi: {e}")


load_checkpoint()


# ============================================================
#  CALLBACK - Worker'dan gelen sonuc
# ============================================================


def make_callback(asset_name):
    """Belirli bir varlik icin callback fonksiyonu olusturur."""

    def callback(result):
        model = result.get("model", "")
        if not model:
            return

        asset_state = state["assets"].get(asset_name)
        if not asset_state:
            return
        model_state = asset_state["models"].get(model)
        if not model_state:
            return

        # 1. Ara adim / Batch / Egitim asamasi guncellemesi
        if result.get("type") == "step_progress":
            model_state["current_trial"] = result.get(
                "current_trial", model_state.get("current_trial", 0)
            )
            model_state["current_params"] = result.get(
                "params", model_state.get("current_params", "")
            )
            model_state["current_horizon"] = result.get(
                "horizon", model_state.get("current_horizon", "")
            )
            model_state["step"] = result.get("step", 0)
            model_state["max_steps"] = result.get("max_steps", 0)
            model_state["loss"] = result.get("loss", None)
            model_state["stage"] = result.get("stage", "Eğitiliyor")
            model_state["trial_elapsed"] = result.get("trial_elapsed", 0.0)
            model_state["status"] = "calisiyor"

            socketio.emit(
                "model_step_progress",
                {
                    "asset": asset_name,
                    "model": model,
                    "step_data": result,
                    "model_state": model_state,
                },
            )
            return

        # 2. Tamamlanan deneme sonucu (normal sonuc)
        horizon = result["horizon"]
        params = result["params"]
        key = (asset_name, model, params, horizon)
        is_cached = result.get("is_cached", False)

        row = {
            "asset": asset_name,
            "model": model,
            "params": params,
            "horizon": horizon,
            "r2": result["r2"],
            "mae": result["mae"],
            "rmse": result["rmse"],
            "mape": result["mape"],
            "elapsed": result["elapsed"],
        }

        with results_lock:
            all_results.append(row)

        if not is_cached:
            completed_runs[key] = row
            with checkpoint_lock:
                try:
                    with open(CHECKPOINT_CSV, "a", encoding="utf-8-sig") as f:
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        f.write(
                            f'"{asset_name}","{model}","{params}","{horizon}",{row["r2"]},{row["mae"]},{row["rmse"]},{row["mape"]},{row["elapsed"]},"{ts}"\n'
                        )
                        f.flush()
                except Exception as e:
                    print(f"CSV yazma hatasi: {e}")

        # Modele ozel log ekle
        time_str = datetime.now().strftime("%H:%M:%S")
        if result["r2"] <= -900:
            log_line = f"[{time_str}] Deneme #{result['done']}/{result['total']} ({horizon}): Başarısız/Hata ({result['elapsed']}s)"
        else:
            log_line = f"[{time_str}] Deneme #{result['done']}/{result['total']} ({horizon}): R²: {result['r2']:.4f} | MAE: {result['mae']:.2f} | MAPE: %{result['mape']:.2f} ({result['elapsed']}s)"

        if "logs" not in model_state:
            model_state["logs"] = []
        model_state["logs"].append(log_line)
        if len(model_state["logs"]) > 100:
            model_state["logs"] = model_state["logs"][-100:]

        # Asset state guncelle
        model_state["done"] = result["done"]
        model_state["stage"] = (
            "Tamamlandı"
            if result["done"] >= result["total"]
            else "Sıradaki Denemeye Geçiliyor"
        )
        model_state["current_trial"] = result["done"]
        model_state["current_params"] = params
        model_state["current_horizon"] = horizon
        model_state["step"] = model_state.get("max_steps", 0)

        asset_state["done"] = sum(m["done"] for m in asset_state["models"].values())
        asset_state["current_model"] = model

        # En iyi skor guncelle
        if result["r2"] > -900 and horizon in best_results[asset_name]:
            current_best = best_results[asset_name][horizon]
            if current_best is None or result["r2"] > current_best["r2"]:
                best_results[asset_name][horizon] = {
                    "r2": result["r2"],
                    "model": model,
                    "params": result["params"],
                    "mae": result["mae"],
                    "rmse": result["rmse"],
                    "mape": result["mape"],
                }
                if not is_cached:
                    save_best_csv()

        # Frontend'e gonder
        socketio.emit(
            "result",
            {
                "asset": asset_name,
                "result": result,
                "best": best_results[asset_name],
                "asset_state": asset_state,
                "total_done": sum(a["done"] for a in state["assets"].values()),
                "total_all": TOTAL_ALL,
                "elapsed_total": (
                    round(time.time() - state["start_time"], 1)
                    if state["start_time"]
                    else 0
                ),
            },
        )

        socketio.emit(
            "model_step_progress",
            {
                "asset": asset_name,
                "model": model,
                "step_data": {
                    "type": "trial_completed",
                    "model": model,
                    "horizon": horizon,
                    "params": params,
                    "current_trial": result["done"],
                    "total": result["total"],
                    "step": model_state["max_steps"],
                    "max_steps": model_state["max_steps"],
                    "stage": model_state["stage"],
                    "log_line": log_line,
                },
                "model_state": model_state,
            },
        )

    return callback


# ============================================================
#  VARLIK İÇİN MODELLERİ ÇALIŞTIR
# ============================================================


def run_asset_prophet(asset_name, csv_file, stop_event):
    """Bir varlik icin Prophet grid search calistirir."""
    csv_path = os.path.join(DATA_DIR, csv_file)
    df = load_data(csv_path)
    callback = make_callback(asset_name)

    asset_state = state["assets"][asset_name]
    asset_state["models"]["Prophet"]["status"] = "calisiyor"
    socketio.emit(
        "model_status", {"asset": asset_name, "model": "Prophet", "status": "calisiyor"}
    )

    run_prophet_grid(
        df,
        PROPHET_GRID,
        HORIZONS,
        asset_name=asset_name,
        completed_dict=completed_runs,
        callback=callback,
        stop_event=stop_event,
    )

    asset_state["models"]["Prophet"]["status"] = "tamamlandi"
    socketio.emit(
        "model_status",
        {"asset": asset_name, "model": "Prophet", "status": "tamamlandi"},
    )


def run_asset_xgboost(asset_name, csv_file, stop_event):
    """Bir varlik icin XGBoost grid search calistirir."""
    csv_path = os.path.join(DATA_DIR, csv_file)
    df = load_data(csv_path)
    callback = make_callback(asset_name)

    asset_state = state["assets"][asset_name]
    asset_state["models"]["XGBoost"]["status"] = "calisiyor"
    socketio.emit(
        "model_status", {"asset": asset_name, "model": "XGBoost", "status": "calisiyor"}
    )

    run_xgboost_grid(
        df,
        XGBOOST_GRID,
        HORIZONS,
        asset_name=asset_name,
        completed_dict=completed_runs,
        callback=callback,
        stop_event=stop_event,
    )

    asset_state["models"]["XGBoost"]["status"] = "tamamlandi"
    socketio.emit(
        "model_status",
        {"asset": asset_name, "model": "XGBoost", "status": "tamamlandi"},
    )


def run_asset_garch(asset_name, model_name, csv_file, stop_event):
    """Bir varlik icin GARCH veya EGARCH grid search calistirir."""
    csv_path = os.path.join(DATA_DIR, csv_file)
    df = load_data(csv_path)
    callback = make_callback(asset_name)

    asset_state = state["assets"][asset_name]
    asset_state["models"][model_name]["status"] = "calisiyor"
    socketio.emit(
        "model_status",
        {"asset": asset_name, "model": model_name, "status": "calisiyor"},
    )

    run_garch_grid(
        df,
        model_name,
        GARCH_GRID,
        HORIZONS,
        asset_name=asset_name,
        completed_dict=completed_runs,
        callback=callback,
        stop_event=stop_event,
    )

    asset_state["models"][model_name]["status"] = "tamamlandi"
    socketio.emit(
        "model_status",
        {"asset": asset_name, "model": model_name, "status": "tamamlandi"},
    )


def run_single_neural_model(asset_name, model_name, csv_file, stop_event):
    """Bir varlik icin derin ogrenme modelini GPU'da calistirir (Maks 2 eszamanli GPU modeli, VRAM tasmasini onler)."""
    asset_state = state["assets"][asset_name]
    model_state = asset_state["models"][model_name]

    # Halihazirda tamamlanmissa atla
    if model_state.get("done", 0) >= model_state.get("total", 1):
        model_state["status"] = "tamamlandi"
        socketio.emit(
            "model_status",
            {"asset": asset_name, "model": model_name, "status": "tamamlandi"},
        )
        return

    # GPU kuyruguna alindigini bildir
    model_state["status"] = "sirada"
    model_state["stage"] = "GPU Sırasında Bekliyor (Maks 2 Paralel Model)"
    socketio.emit(
        "model_status", {"asset": asset_name, "model": model_name, "status": "sirada"}
    )

    with gpu_semaphore:
        if stop_event and stop_event.is_set():
            model_state["status"] = "bekliyor"
            socketio.emit(
                "model_status",
                {"asset": asset_name, "model": model_name, "status": "bekliyor"},
            )
            return

        csv_path = os.path.join(DATA_DIR, csv_file)
        df = load_data(csv_path)
        callback = make_callback(asset_name)

        model_state["status"] = "calisiyor"
        model_state["stage"] = "GPU'da Eğitiliyor"
        socketio.emit(
            "model_status",
            {"asset": asset_name, "model": model_name, "status": "calisiyor"},
        )

        try:
            run_neural_grid(
                df,
                model_name,
                NEURAL_GRID,
                HORIZONS,
                asset_name=asset_name,
                completed_dict=completed_runs,
                use_gpu=True,
                callback=callback,
                stop_event=stop_event,
            )
        finally:
            try:
                import gc

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        model_state["status"] = "tamamlandi"
        model_state["stage"] = "Tamamlandı"
        socketio.emit(
            "model_status",
            {"asset": asset_name, "model": model_name, "status": "tamamlandi"},
        )


def run_all_neural_models(asset_name, csv_file, stop_event):
    """Varlik icin derin ogrenme modellerini sirayla GPU'da guvenle calistirir."""
    csv_path = os.path.join(DATA_DIR, csv_file)
    df = load_data(csv_path)
    callback = make_callback(asset_name)

    for model_name in NEURAL_MODELS:
        if stop_event and stop_event.is_set():
            break
        asset_state = state["assets"][asset_name]
        asset_state["models"][model_name]["status"] = "calisiyor"
        socketio.emit(
            "model_status",
            {"asset": asset_name, "model": model_name, "status": "calisiyor"},
        )

        run_neural_grid(
            df,
            model_name,
            NEURAL_GRID,
            HORIZONS,
            asset_name=asset_name,
            completed_dict=completed_runs,
            use_gpu=True,
            callback=callback,
            stop_event=stop_event,
        )

        asset_state["models"][model_name]["status"] = "tamamlandi"
        socketio.emit(
            "model_status",
            {"asset": asset_name, "model": model_name, "status": "tamamlandi"},
        )


def save_best_csv():
    """Sadece her varligin 3, 6 ve 12 aylik EN IYI modellerini ve parametrelerini CSV'ye kaydeder."""
    import pandas as pd

    rows = []
    for asset_name, horizons_dict in best_results.items():
        for horizon, data in horizons_dict.items():
            if data and data.get("r2", -900) > -900:
                rows.append(
                    {
                        "Varlık": asset_name,
                        "Tahmin Ufku": horizon,
                        "En İyi Model": data["model"],
                        "R2 Skoru": data["r2"],
                        "MAE": data["mae"],
                        "RMSE": data["rmse"],
                        "MAPE (%)": data["mape"],
                        "En İyi Parametreler": data["params"],
                    }
                )
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(BEST_CSV_OUTPUT, index=False, encoding="utf-8-sig")


def process_single_asset(asset_name, csv_file, stop_event):
    """10 Modeli AYNI ANDA paralel calistirir: 4 CPU (Prophet, XGBoost, GARCH, EGARCH) + 6 GPU (NBEATS, NHITS, PatchTST, LSTM, TFT, TiDE)."""
    asset_state = state["assets"][asset_name]
    asset_state["status"] = "calisiyor"
    socketio.emit("asset_start", {"asset": asset_name})

    threads = []
    # 1. CPU Modelleri (Prophet, XGBoost, GARCH, EGARCH)
    t_prophet = threading.Thread(
        target=run_asset_prophet, args=(asset_name, csv_file, stop_event), daemon=True
    )
    t_xgb = threading.Thread(
        target=run_asset_xgboost, args=(asset_name, csv_file, stop_event), daemon=True
    )
    threads.extend([t_prophet, t_xgb])

    for g_model in GARCH_MODELS:
        t_garch = threading.Thread(
            target=run_asset_garch,
            args=(asset_name, g_model, csv_file, stop_event),
            daemon=True,
        )
        threads.append(t_garch)

    # 2. 6 GPU Modeli (NBEATS, NHITS, PatchTST, LSTM, TFT, TiDE - RTX 3070 uzerinde paralel)
    for model_name in NEURAL_MODELS:
        t_neural = threading.Thread(
            target=run_single_neural_model,
            args=(asset_name, model_name, csv_file, stop_event),
            daemon=True,
        )
        threads.append(t_neural)

    # 10 Thread'i Ayni Anda Baslat
    for t in threads:
        t.start()

    # Tum modellerin bitmesini bekle
    for t in threads:
        t.join()

    asset_state["status"] = "tamamlandi"
    save_best_csv()
    socketio.emit("asset_done", {"asset": asset_name, "best": best_results[asset_name]})


# ============================================================
#  ANA BAŞLATMA METODLARI
# ============================================================


def start_all():
    """Varliklari sirayla calistirir; her varlikta tum modeller calisir."""
    if state["started"]:
        return

    state["started"] = True
    state["start_time"] = time.time()

    global stop_events
    stop_events = {a: threading.Event() for a in ASSETS}

    def orchestrator():
        for asset_name, csv_file in ASSETS.items():
            if any(ev.is_set() for ev in stop_events.values()):
                break
            process_single_asset(asset_name, csv_file, stop_events[asset_name])
            if any(ev.is_set() for ev in stop_events.values()):
                break

        state["started"] = False
        save_best_csv()
        if not any(ev.is_set() for ev in stop_events.values()):
            socketio.emit(
                "all_done",
                {
                    "elapsed": (
                        round(time.time() - state["start_time"], 1)
                        if state["start_time"]
                        else 0
                    ),
                    "total": len(all_results),
                    "best_csv": "grid_search_en_iyi_modeller.csv",
                },
            )

    main_t = threading.Thread(target=orchestrator, daemon=True)
    main_t.start()
    active_threads["main_orchestrator"] = main_t


def start_single(asset_name):
    """Sadece secilen tek bir varlik icin testleri baslatir."""
    if asset_name not in ASSETS:
        return

    state["started"] = True
    state["start_time"] = time.time()

    global stop_events
    stop_events = {asset_name: threading.Event()}

    def single_runner():
        csv_file = ASSETS[asset_name]
        process_single_asset(asset_name, csv_file, stop_events[asset_name])
        state["started"] = False
        save_best_csv()
        socketio.emit(
            "single_completed",
            {
                "asset": asset_name,
                "best": best_results[asset_name],
                "elapsed": round(time.time() - state["start_time"], 1),
            },
        )

    t = threading.Thread(target=single_runner, daemon=True)
    t.start()
    active_threads[f"single_{asset_name}"] = t


# ============================================================
#  FLASK ROUTE'LARI
# ============================================================


@app.route("/")
def index():
    return render_template("dashboard.html")


def get_system_telemetry():
    """Anlik CPU, RAM ve GPU telemetrisini okur."""
    cpu_pct = psutil.cpu_percent(interval=None)
    vm = psutil.virtual_memory()
    ram_pct = vm.percent
    ram_used_gb = round(vm.used / (1024**3), 1)
    ram_total_gb = round(vm.total / (1024**3), 1)

    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "GPU Yok"
    gpu_vram_used_gb = 0.0
    gpu_vram_total_gb = 0.0
    gpu_vram_pct = 0.0

    if gpu_available:
        try:
            reserved = torch.cuda.memory_reserved(0)
            total = torch.cuda.get_device_properties(0).total_memory
            gpu_vram_used_gb = round(reserved / (1024**3), 2)
            gpu_vram_total_gb = round(total / (1024**3), 1)
            gpu_vram_pct = round((reserved / max(total, 1)) * 100, 1)
        except Exception:
            pass

    return {
        "cpu_pct": cpu_pct,
        "ram_pct": ram_pct,
        "ram_used_gb": ram_used_gb,
        "ram_total_gb": ram_total_gb,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "gpu_vram_used_gb": gpu_vram_used_gb,
        "gpu_vram_total_gb": gpu_vram_total_gb,
        "gpu_vram_pct": gpu_vram_pct,
    }


def telemetry_worker():
    """Surekli calisip frontend'e canli donanim metrikleri gonderir."""
    # psutil cpu_percent ilk cagriyi kalibre et
    psutil.cpu_percent(interval=None)
    while True:
        try:
            metrics = get_system_telemetry()
            socketio.emit("system_metrics", metrics)
        except Exception:
            pass
        time.sleep(1.5)


# Telemetri thread'ini baslat
telemetry_thread = threading.Thread(target=telemetry_worker, daemon=True)
telemetry_thread.start()


@app.route("/api/system")
def get_system():
    return jsonify(get_system_telemetry())


@app.route("/api/state")
def get_state():
    elapsed = round(time.time() - state["start_time"], 1) if state["start_time"] else 0
    total_done = sum(a["done"] for a in state["assets"].values())

    return jsonify(
        {
            "started": state["started"],
            "elapsed": elapsed,
            "total_done": total_done,
            "total_all": TOTAL_ALL,
            "assets": state["assets"],
            "best": best_results,
            "combo_counts": COMBO_COUNTS,
            "system": get_system_telemetry(),
        }
    )


@app.route("/api/results")
def get_results():
    with results_lock:
        return jsonify(all_results[-100:])


@app.route("/api/trials/<path:asset_name>")
def get_asset_trials(asset_name):
    with results_lock:
        trials = [r for r in all_results if r["asset"] == asset_name]
        return jsonify(trials)


@app.route("/api/model_detail/<path:asset_name>/<model_name>")
def get_model_detail(asset_name, model_name):
    asset_state = state["assets"].get(asset_name, {})
    model_state = asset_state.get("models", {}).get(model_name, {})
    with results_lock:
        trials = [
            r
            for r in all_results
            if r["asset"] == asset_name and r["model"] == model_name
        ]
    logs = list(model_state.get("logs", []))
    if not logs and trials:
        logs = [
            f"[Arşiv] Bu model için checkpoint'ten {len(trials)} deneme yüklendi. Detaylar 'Tamamlanan Denemeler' sekmesindedir."
        ]
    return jsonify(
        {
            "asset": asset_name,
            "model": model_name,
            "model_state": model_state,
            "trials": trials,
            "logs": logs,
        }
    )


@socketio.on("connect")
def handle_connect():
    pass


@socketio.on("start")
def handle_start():
    start_all()
    socketio.emit(
        "started",
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "total": TOTAL_ALL,
            "assets": list(ASSETS.keys()),
        },
    )


@socketio.on("start_single")
def handle_start_single(data):
    asset_name = data.get("asset")
    if asset_name and asset_name in ASSETS:
        start_single(asset_name)
        socketio.emit(
            "single_started",
            {
                "asset": asset_name,
                "time": datetime.now().strftime("%H:%M:%S"),
                "total": TOTAL_PER_ASSET,
            },
        )


@socketio.on("stop")
def handle_stop():
    for ev in stop_events.values():
        ev.set()
    state["started"] = False
    save_results_csv()
    socketio.emit("stopped", {"total_saved": len(all_results)})


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  GRID SEARCH DASHBOARD")
    print(
        f"  Toplam {TOTAL_ALL} deney ({len(ASSETS)} varlik x {TOTAL_PER_ASSET} deney/varlik)"
    )
    print(f"  Tarayicida ac: http://localhost:5050")
    print("=" * 60)
    socketio.run(
        app, host="0.0.0.0", port=5050, debug=False, allow_unsafe_werkzeug=True
    )
