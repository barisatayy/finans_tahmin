/**
 * Grid Search Dashboard - Frontend Application Logic
 * Real-time WebSocket Client, Multi-Asset State Management, Live Metrics
 */

// ============================================================
//  GLOBAL STATE & CONSTANTS
// ============================================================

const socket = io();

let currentAsset = "USD/TRY";
let isRunning = false;
let startTime = null;
let timerInterval = null;

// Store logs per asset for instant switching
const assetLogs = {
    "USD/TRY": [],
    "EUR/TRY": [],
    "Altin (Ons)": [],
    "Bitcoin (USD)": [],
    "Apple (USD)": [],
    "Tesla (USD)": [],
    "THY (TL)": []
};

// Store latest state from server
let appState = {
    total_done: 0,
    total_all: 7182,
    assets: {},
    best: {}
};

const recentResults = [];
const assetTrials = {};
let currentViewMode = "live";

// ============================================================
//  INITIALIZATION
// ============================================================

document.addEventListener("DOMContentLoaded", () => {
    fetchInitialState();
    setupSocketListeners();
});

function fetchInitialState() {
    fetch("/api/state")
        .then(res => res.json())
        .then(data => {
            if (data.assets) {
                appState = data;
                updateOverallProgress(data.total_done, data.total_all, data.elapsed);
                if (data.started) {
                    isRunning = true;
                    setButtonsRunningState(true);
                    startTime = Date.now() - (data.elapsed * 1000);
                    startTimer();
                } else {
                    setButtonsRunningState(false);
                }
                if (data.system) {
                    updateSystemTelemetry(data.system);
                }
                renderActiveAsset();
            }
        })
        .catch(err => console.log("State fetch error:", err));
}

function setButtonsRunningState(running) {
    const btnAll = document.getElementById("btn-start-all");
    const btnSingle = document.getElementById("btn-start-single");
    const btnStop = document.getElementById("btn-stop");

    if (btnAll) btnAll.disabled = running;
    if (btnSingle) btnSingle.disabled = running;
    if (btnStop) btnStop.disabled = !running;
}

// ============================================================
//  SOCKET EVENT LISTENERS
// ============================================================

function setupSocketListeners() {
    socket.on("connect", () => {
        console.log("WebSocket bağlantısı kuruldu.");
    });

    socket.on("system_metrics", data => {
        updateSystemTelemetry(data);
    });

    socket.on("started", data => {
        isRunning = true;
        setButtonsRunningState(true);
        document.getElementById("start-time-display").innerText = data.time || new Date().toLocaleTimeString();

        startTime = Date.now();
        startTimer();

        appendLogEntry("USD/TRY", "system", `Tüm varlıklar sırayla başlatıldı. Toplam ${data.total} deney yürütülüyor.`);
    });

    socket.on("single_started", data => {
        isRunning = true;
        setButtonsRunningState(true);
        document.getElementById("start-time-display").innerText = data.time || new Date().toLocaleTimeString();

        startTime = Date.now();
        startTimer();

        appendLogEntry(data.asset, "system", `[TEK TEST] Sadece ${data.asset} için testler başlatıldı. (${data.total} deney)`);
    });

    socket.on("result", data => {
        const { asset, result, best, asset_state, total_done, total_all, elapsed_total } = data;

        // 1. Update overall progress
        updateOverallProgress(total_done, total_all, elapsed_total);

        // 2. Save and append log
        appendResultLog(asset, result);

        // 3. Update memory state
        if (!appState.assets) appState.assets = {};
        appState.assets[asset] = asset_state;
        if (!appState.best) appState.best = {};
        appState.best[asset] = best;

        // 4. Update asset trials cache
        if (!assetTrials[asset]) assetTrials[asset] = [];
        assetTrials[asset].push(result);

        // 5. Update tab progress mini badge & status dot
        updateTabVisuals(asset, asset_state);

        // 6. If this is the active asset, update right-panel cards & model breakdown
        if (asset === currentAsset) {
            renderBestCards(best);
            renderModelBreakdown(asset_state);
            if (currentViewMode === "params") {
                filterAssetTrialsTable();
            }
        }

        // 7. Update recent table
        addRecentResultRow(asset, result);
    });

    socket.on("model_status", data => {
        const { asset, model, status } = data;
        if (appState.assets && appState.assets[asset] && appState.assets[asset].models[model]) {
            appState.assets[asset].models[model].status = status;
            if (asset === currentAsset) {
                renderModelBreakdown(appState.assets[asset]);
            }
        }
        // If modal open for this model, update status badge
        if (activeModalModel === model && currentAsset === asset) {
            const statusEl = document.getElementById("md-status-badge");
            if (statusEl) {
                statusEl.innerText = status === "calisiyor" ? "Çalışıyor" : (status === "sirada" ? "Sırada" : (status === "tamamlandi" ? "Tamamlandı" : "Bekliyor"));
                statusEl.className = "badge " + (status === "calisiyor" ? "badge-accent" : (status === "sirada" ? "badge-warning" : (status === "tamamlandi" ? "badge-success" : "")));
            }
        }
    });

    socket.on("model_step_progress", data => {
        const { asset, model, step_data, model_state } = data;

        // Memory state update
        if (appState.assets && appState.assets[asset] && appState.assets[asset].models[model]) {
            Object.assign(appState.assets[asset].models[model], model_state);
        }

        // If modal open for this model & asset, update live progress
        if (activeModalModel === model && currentAsset === asset) {
            updateModelModalLiveElements(model_state, step_data);
        }
    });

    socket.on("asset_start", data => {
        const dot = document.getElementById(`dot-${data.asset}`);
        if (dot) {
            dot.className = "tab-status-dot active-running";
        }
        appendLogEntry(data.asset, "system", `[BAŞLADI] ${data.asset} için testler başladı. Tüm modeller eşzamanlı çalışıyor...`);
        switchAssetTab(data.asset);
    });

    socket.on("asset_done", data => {
        const dot = document.getElementById(`dot-${data.asset}`);
        if (dot) {
            dot.className = "tab-status-dot completed";
        }

        let summaryText = `[TAMAMLANDI] ${data.asset} için tüm testler bitti! En iyi modeller grid_search_en_iyi_modeller.csv'ye kaydedildi.\n`;
        if (data.best) {
            for (const [hKey, hData] of Object.entries(data.best)) {
                if (hData) {
                    summaryText += `  • ${hKey}: ${hData.model} (R²=${hData.r2}) - ${hData.params}\n`;
                }
            }
        }

        appendLogEntry(data.asset, "system", summaryText);
        showToast(`${data.asset} için en iyi modeller kaydedildi.`, "success");
    });

    socket.on("single_completed", data => {
        isRunning = false;
        clearInterval(timerInterval);
        setButtonsRunningState(false);

        showAssetCompletedModal(data);
        showToast(`${data.asset} testi tamamlandı! (${data.elapsed}s)`, "success");
    });

    socket.on("all_done", data => {
        isRunning = false;
        clearInterval(timerInterval);
        setButtonsRunningState(false);

        showAllDoneModal(data);
        showToast("Tüm grid search başarıyla tamamlandı!", "success");
    });

    socket.on("stopped", data => {
        isRunning = false;
        clearInterval(timerInterval);
        setButtonsRunningState(false);
        appendLogEntry(currentAsset, "system", `[DURDURULDU] Kullanıcı tarafından durduruldu. En iyi sonuçlar kaydedildi.`);
        showToast("İşlem durduruldu. En iyi modeller CSV'ye kaydedildi.", "warning");
    });
}

// ============================================================
//  USER ACTIONS (START / STOP / TABS)
// ============================================================

function startAllAssets() {
    showToast("Tüm varlıklar için optimizasyon başlatıldı...", "info");
    socket.emit("start");
}

function startSingleAsset() {
    showToast(`${currentAsset} testi başlatıldı...`, "info");
    socket.emit("start_single", { asset: currentAsset });
}

function stopGridSearch() {
    showCustomConfirm(
        "Grid Search'ü Durdur",
        "İşlemi durdurmak istediğinize emin misiniz? Şimdiye kadar bulunan en iyi modeller CSV dosyasına kaydedilecektir.",
        () => {
            socket.emit("stop");
        }
    );
}

function switchAssetTab(assetName) {
    currentAsset = assetName;

    // Update active tab button styling
    document.querySelectorAll(".tab-btn").forEach(btn => {
        btn.classList.toggle("active", btn.dataset.asset === assetName);
    });

    // Update titles and single test button name
    document.getElementById("current-asset-title").innerText = assetName;
    document.getElementById("best-asset-title").innerText = assetName;

    const singleBtnName = document.getElementById("btn-single-name");
    if (singleBtnName) singleBtnName.innerText = assetName;

    const paramsAssetName = document.getElementById("view-params-asset-name");
    if (paramsAssetName) paramsAssetName.innerText = assetName;

    const tableAssetHeader = document.getElementById("table-asset-header-title");
    if (tableAssetHeader) tableAssetHeader.innerText = assetName;

    // Render components for selected asset
    renderActiveAsset();

    if (currentViewMode === "params") {
        loadAssetTrials(currentAsset);
    }
}

function switchViewMode(mode) {
    currentViewMode = mode;
    const btnLive = document.getElementById("btn-view-live");
    const btnParams = document.getElementById("btn-view-params");
    const secLive = document.getElementById("section-live-view");
    const secParams = document.getElementById("section-params-view");

    if (mode === "live") {
        if (btnLive) btnLive.classList.add("active");
        if (btnParams) btnParams.classList.remove("active");
        if (secLive) secLive.style.display = "grid";
        if (secParams) secParams.style.display = "none";
    } else {
        if (btnLive) btnLive.classList.remove("active");
        if (btnParams) btnParams.classList.add("active");
        if (secLive) secLive.style.display = "none";
        if (secParams) secParams.style.display = "block";
        loadAssetTrials(currentAsset);
    }
}

function loadAssetTrials(assetName) {
    fetch(`/api/trials/${encodeURIComponent(assetName)}`)
        .then(res => res.json())
        .then(data => {
            assetTrials[assetName] = data || [];
            filterAssetTrialsTable();
        })
        .catch(err => {
            console.error("Trials fetch error:", err);
            filterAssetTrialsTable();
        });
}

let currentSortColumn = "r2";
let currentSortDirection = "desc"; // "asc" or "desc"

function sortTableByColumn(columnKey) {
    if (currentSortColumn === columnKey) {
        currentSortDirection = currentSortDirection === "asc" ? "desc" : "asc";
    } else {
        currentSortColumn = columnKey;
        // Default direction: r2 desc, all other metrics/fields default to asc
        currentSortDirection = columnKey === "r2" ? "desc" : "asc";
    }
    filterAssetTrialsTable();
}

function filterAssetTrialsTable() {
    const trials = assetTrials[currentAsset] || [];
    const modelFilter = document.getElementById("table-filter-model").value;
    const horizonFilter = document.getElementById("table-filter-horizon").value;
    const searchFilter = (document.getElementById("table-search-param").value || "").toLowerCase().trim();

    const filtered = trials.filter(t => {
        if (modelFilter !== "ALL" && t.model !== modelFilter) return false;
        if (horizonFilter !== "ALL" && t.horizon !== horizonFilter) return false;
        if (searchFilter && !String(t.params).toLowerCase().includes(searchFilter)) return false;
        return true;
    });

    // Update sort header icons
    const sortKeys = ["model", "horizon", "r2", "mae", "rmse", "mape", "elapsed", "params"];
    sortKeys.forEach(key => {
        const iconEl = document.getElementById(`sort-icon-${key}`);
        if (iconEl) {
            if (currentSortColumn === key) {
                iconEl.innerHTML = currentSortDirection === "asc" ? "&#9650;" : "&#9660;";
            } else {
                iconEl.innerHTML = "";
            }
        }
    });

    // Sort by active column and direction
    filtered.sort((a, b) => {
        let valA = a[currentSortColumn];
        let valB = b[currentSortColumn];

        if (["r2", "mae", "rmse", "mape", "elapsed"].includes(currentSortColumn)) {
            valA = Number(valA);
            valB = Number(valB);
            if (isNaN(valA)) valA = currentSortDirection === "asc" ? 999999 : -999999;
            if (isNaN(valB)) valB = currentSortDirection === "asc" ? 999999 : -999999;
            return currentSortDirection === "asc" ? valA - valB : valB - valA;
        } else {
            valA = String(valA || "").toLowerCase();
            valB = String(valB || "").toLowerCase();
            if (valA < valB) return currentSortDirection === "asc" ? -1 : 1;
            if (valA > valB) return currentSortDirection === "asc" ? 1 : -1;
            return 0;
        }
    });

    const tbody = document.getElementById("asset-all-trials-tbody");
    const counterBadge = document.getElementById("table-records-counter");
    if (counterBadge) counterBadge.innerText = `${filtered.length} Kayıt`;

    if (!tbody) return;

    if (filtered.length === 0) {
        tbody.innerHTML = `<tr class="empty-row"><td colspan="8">Filtre kriterlerine uygun kayıt bulunamadı (${currentAsset}).</td></tr>`;
        return;
    }

    tbody.innerHTML = filtered.map(t => {
        const r2Class = t.r2 >= 0.85 ? "r2-high" : (t.r2 >= 0.5 ? "r2-mid" : "r2-low");
        const r2Display = t.r2 <= -900 ? "Hata" : Number(t.r2).toFixed(4);
        const maeDisplay = Number(t.mae) >= 999 ? "-" : Number(t.mae).toFixed(2);
        const rmseDisplay = Number(t.rmse) >= 999 ? "-" : Number(t.rmse).toFixed(2);
        const mapeDisplay = Number(t.mape) >= 999 ? "-" : `%${Number(t.mape).toFixed(2)}`;

        return `
            <tr>
                <td><span class="badge badge-outline">${t.model}</span></td>
                <td><b>${t.horizon}</b></td>
                <td><span class="r2-badge ${r2Class}">${r2Display}</span></td>
                <td>${maeDisplay}</td>
                <td>${rmseDisplay}</td>
                <td>${mapeDisplay}</td>
                <td>${t.elapsed || 0}s</td>
                <td><span class="params-code-cell">${t.params}</span></td>
            </tr>
        `;
    }).join("");
}

function clearCurrentLog() {
    assetLogs[currentAsset] = [];
    document.getElementById("log-terminal").innerHTML = `
        <div class="log-entry system-msg">
            <span class="log-time">[${new Date().toLocaleTimeString()}]</span> Log akışı temizlendi.
        </div>
    `;
}

// ============================================================
//  RENDER FUNCTIONS
// ============================================================

function renderActiveAsset() {
    const assetState = (appState.assets && appState.assets[currentAsset]) || null;
    const bestMetrics = (appState.best && appState.best[currentAsset]) || null;

    // 1. Render Log Terminal
    const terminal = document.getElementById("log-terminal");
    const logs = assetLogs[currentAsset];

    if (!logs || logs.length === 0) {
        terminal.innerHTML = `
            <div class="log-entry system-msg">
                <span class="log-time">[00:00:00]</span> ${currentAsset} için test akışı bekleniyor...
            </div>
        `;
    } else {
        terminal.innerHTML = logs.join("");
        scrollTerminalToBottom();
    }

    // 2. Render Model Breakdown
    renderModelBreakdown(assetState);

    // 3. Render Best Cards
    renderBestCards(bestMetrics);
}

function renderModelBreakdown(assetState) {
    const container = document.getElementById("models-progress-container");
    const statusBadge = document.getElementById("asset-overall-status");

    if (!assetState) {
        statusBadge.innerText = "Bekliyor";
        statusBadge.className = "badge";
        container.innerHTML = `<div style="color: var(--text-muted); font-size: 12px; padding: 10px;">Model bilgisi yükleniyor...</div>`;
        return;
    }

    statusBadge.innerText = assetState.status === "calisiyor" ? "Çalışıyor" : (assetState.status === "tamamlandi" ? "Tamamlandı" : "Bekliyor");
    statusBadge.className = "badge " + (assetState.status === "calisiyor" ? "badge-accent" : "");

    const models = assetState.models || {};
    let html = "";

    for (const [modelName, info] of Object.entries(models)) {
        const pct = info.total > 0 ? Math.round((info.done / info.total) * 100) : 0;
        let tagClass = "tag-waiting";
        let tagText = "Bekliyor";

        if (info.status === "calisiyor") {
            tagClass = "tag-running";
            tagText = "Çalışıyor";
        } else if (info.status === "sirada") {
            tagClass = "tag-queued";
            tagText = "Sırada";
        } else if (info.status === "tamamlandi" || pct >= 100) {
            tagClass = "tag-done";
            tagText = "Bitti";
        }

        html += `
            <div class="model-progress-item clickable-model-card" onclick="openModelDetailModal('${modelName}')" title="${modelName} anlık eğitim aşaması ve loglarını görmek için tıklayın">
                <div class="model-item-header">
                    <div style="display: flex; align-items: center;">
                        <span class="model-item-name">${modelName}</span>
                        <span class="model-item-inspect-pill">&#128065; Canlı Detay</span>
                    </div>
                    <span class="model-item-tag ${tagClass}">${tagText}</span>
                </div>
                <div class="model-bar-wrapper">
                    <div class="model-bar-fill ${pct >= 100 ? 'done' : ''}" style="width: ${pct}%;"></div>
                </div>
                <div class="model-item-meta">
                    <span>${info.done} / ${info.total}</span>
                    <span>%${pct}</span>
                </div>
            </div>
        `;
    }

    container.innerHTML = html;
}

function renderBestCards(bestMetrics) {
    const horizons = [
        { key: "3 Ay", idSuffix: "3m" },
        { key: "6 Ay", idSuffix: "6m" },
        { key: "12 Ay", idSuffix: "12m" }
    ];

    horizons.forEach(h => {
        const r2El = document.getElementById(`best-r2-${h.idSuffix}`);
        const modelEl = document.getElementById(`best-model-${h.idSuffix}`);
        const maeEl = document.getElementById(`best-mae-${h.idSuffix}`);
        const rmseEl = document.getElementById(`best-rmse-${h.idSuffix}`);
        const mapeEl = document.getElementById(`best-mape-${h.idSuffix}`);
        const paramsEl = document.getElementById(`best-params-${h.idSuffix}`);
        const boxEl = document.getElementById(`best-box-${h.idSuffix}`);

        const data = bestMetrics ? bestMetrics[h.key] : null;

        if (data && data.r2 !== undefined && data.r2 > -900) {
            const oldR2 = parseFloat(r2El.innerText);
            if (isNaN(oldR2) || data.r2 > oldR2) {
                // Flash glow animation on new high score
                boxEl.classList.add("glow-update");
                setTimeout(() => boxEl.classList.remove("glow-update"), 1200);
            }

            r2El.innerText = data.r2.toFixed(4);
            modelEl.innerText = data.model;
            maeEl.innerText = data.mae ? data.mae.toFixed(3) : "--";
            rmseEl.innerText = data.rmse ? data.rmse.toFixed(3) : "--";
            mapeEl.innerText = data.mape ? `%${data.mape.toFixed(1)}` : "--";
            paramsEl.innerText = data.params;
        } else {
            r2El.innerText = "--";
            modelEl.innerText = "Henüz test edilmedi";
            maeEl.innerText = "--";
            rmseEl.innerText = "--";
            mapeEl.innerText = "--";
            paramsEl.innerText = "--";
        }
    });
}

// ============================================================
//  LOGGING & LOG FORMATTING
// ============================================================

function appendResultLog(asset, result) {
    const timeStr = new Date().toLocaleTimeString();
    let badgeClass = "badge-prophet";
    if (result.model === "XGBoost") badgeClass = "badge-xgboost";
    else if (["NBEATS", "NHITS", "PatchTST", "LSTM"].includes(result.model)) badgeClass = "badge-neural";

    let r2Class = "log-r2";
    if (result.r2 >= 0.80) r2Class += " high";
    else if (result.r2 >= 0.30) r2Class += " medium";
    else if (result.r2 < 0) r2Class += " negative";

    const r2Display = result.r2 > -100 ? result.r2.toFixed(4) : "HATA";

    const logHtml = `
        <div class="log-entry">
            <span class="log-time">[${timeStr}]</span>
            <span class="log-model-badge ${badgeClass}">${result.model}</span>
            <span style="color: var(--text-muted); font-size: 11px;">[${result.horizon}]</span>
            <span class="${r2Class}">R²=${r2Display}</span>
            <span style="color: var(--text-secondary); font-size: 11px;">MAE=${result.mae} RMSE=${result.rmse} MAPE=%${result.mape}</span>
            <span style="color: #64748b; font-size: 10px;">(${result.elapsed}s)</span>
            <div style="color: #38bdf8; font-size: 11px; margin-left: 20px;">${result.params}</div>
        </div>
    `;

    if (!assetLogs[asset]) assetLogs[asset] = [];
    assetLogs[asset].push(logHtml);

    // If currently looking at this asset, append directly to DOM
    if (asset === currentAsset) {
        const terminal = document.getElementById("log-terminal");
        terminal.insertAdjacentHTML("beforeend", logHtml);
        scrollTerminalToBottom();
    }
}

function appendLogEntry(asset, type, msg) {
    const timeStr = new Date().toLocaleTimeString();
    const logHtml = `
        <div class="log-entry system-msg">
            <span class="log-time">[${timeStr}]</span> ${msg}
        </div>
    `;
    if (!assetLogs[asset]) assetLogs[asset] = [];
    assetLogs[asset].push(logHtml);

    if (asset === currentAsset) {
        const terminal = document.getElementById("log-terminal");
        terminal.insertAdjacentHTML("beforeend", logHtml);
        scrollTerminalToBottom();
    }
}

function scrollTerminalToBottom() {
    const chk = document.getElementById("autoscroll-chk");
    if (chk && chk.checked) {
        const terminal = document.getElementById("log-terminal");
        terminal.scrollTop = terminal.scrollHeight;
    }
}

// ============================================================
//  RECENT RESULTS TABLE
// ============================================================

function addRecentResultRow(asset, result) {
    const tbody = document.getElementById("recent-table-body");
    const emptyRow = tbody.querySelector(".empty-row");
    if (emptyRow) emptyRow.remove();

    let r2Color = "#94a3b8";
    if (result.r2 >= 0.8) r2Color = "#facc15";
    else if (result.r2 >= 0.4) r2Color = "#4ade80";
    else if (result.r2 < 0) r2Color = "#f87171";

    const tr = document.createElement("tr");
    tr.innerHTML = `
        <td style="font-weight: 600; color: #fff;">${asset}</td>
        <td>${result.model}</td>
        <td>${result.horizon}</td>
        <td style="color: ${r2Color}; font-weight: 700;">${result.r2.toFixed(4)}</td>
        <td>${result.mae}</td>
        <td>${result.elapsed}s</td>
    `;

    tbody.insertBefore(tr, tbody.firstChild);

    // Keep only last 12 rows
    while (tbody.children.length > 12) {
        tbody.removeChild(tbody.lastChild);
    }
}

// ============================================================
//  PROGRESS & ESTIMATED TIME CALCULATION
// ============================================================

function updateOverallProgress(done, total, elapsedSeconds) {
    const pct = total > 0 ? ((done / total) * 100).toFixed(1) : "0.0";

    document.getElementById("overall-percentage").innerText = `${pct}%`;
    document.getElementById("overall-counts").innerText = `${done.toLocaleString()} / ${total.toLocaleString()}`;
    document.getElementById("overall-progress-bar").style.width = `${pct}%`;
}

function updateTabVisuals(asset, assetState) {
    const mini = document.getElementById(`mini-progress-${asset}`);
    const dot = document.getElementById(`dot-${asset}`);

    if (assetState && mini) {
        const pct = assetState.total > 0 ? Math.round((assetState.done / assetState.total) * 100) : 0;
        mini.innerText = `%${pct}`;

        if (dot) {
            if (pct >= 100 || assetState.status === "tamamlandi") {
                dot.className = "tab-status-dot completed";
            } else if (assetState.status === "calisiyor" || pct > 0) {
                dot.className = "tab-status-dot active-running";
            }
        }
    }
}

function startTimer() {
    if (timerInterval) clearInterval(timerInterval);
    timerInterval = setInterval(() => {
        if (!isRunning || !startTime) return;
        const elapsedSec = Math.floor((Date.now() - startTime) / 1000);
        document.getElementById("elapsed-time-display").innerText = formatSeconds(elapsedSec);
    }, 1000);
}

function formatSeconds(sec) {
    if (sec < 0 || isNaN(sec)) return "00:00:00";
    const h = Math.floor(sec / 3600).toString().padStart(2, '0');
    const m = Math.floor((sec % 3600) / 60).toString().padStart(2, '0');
    const s = Math.floor(sec % 60).toString().padStart(2, '0');
    return `${h}:${m}:${s}`;
}

function updateSystemTelemetry(data) {
    if (!data) return;

    // CPU
    const cpuPct = data.cpu_pct !== undefined ? data.cpu_pct.toFixed(1) : "0.0";
    const cpuEl = document.getElementById("cpu-val-display");
    const cpuBar = document.getElementById("cpu-bar-fill");
    if (cpuEl) cpuEl.innerText = `%${cpuPct}`;
    if (cpuBar) cpuBar.style.width = `${Math.min(100, Math.max(0, data.cpu_pct || 0))}%`;

    // RAM
    const ramPct = data.ram_pct !== undefined ? data.ram_pct.toFixed(1) : "0.0";
    const ramEl = document.getElementById("ram-val-display");
    const ramBar = document.getElementById("ram-bar-fill");
    const ramSub = document.getElementById("ram-subtext-display");
    if (ramEl) ramEl.innerText = `%${ramPct}`;
    if (ramBar) ramBar.style.width = `${Math.min(100, Math.max(0, data.ram_pct || 0))}%`;
    if (ramSub) ramSub.innerText = `${data.ram_used_gb || 0} GB / ${data.ram_total_gb || 16} GB`;

    // GPU VRAM
    const gpuEl = document.getElementById("gpu-val-display");
    const gpuBar = document.getElementById("gpu-bar-fill");
    const gpuSub = document.getElementById("gpu-subtext-display");

    if (data.gpu_available) {
        const gpuPct = data.gpu_vram_pct !== undefined ? data.gpu_vram_pct.toFixed(1) : "0.0";
        if (gpuEl) gpuEl.innerText = `%${gpuPct}`;
        if (gpuBar) gpuBar.style.width = `${Math.min(100, Math.max(0, data.gpu_vram_pct || 0))}%`;
        if (gpuSub) gpuSub.innerText = `${data.gpu_name || 'GPU'} • ${data.gpu_vram_used_gb || 0} GB / ${data.gpu_vram_total_gb || 8} GB`;
    } else {
        if (gpuEl) gpuEl.innerText = "Pasif";
        if (gpuBar) gpuBar.style.width = "0%";
        if (gpuSub) gpuSub.innerText = "GPU Algılanamadı (CPU Modu)";
    }
}

// ============================================================
//  CUSTOM MODERN MODAL & TOAST NOTIFICATION HELPERS
// ============================================================

let currentModalConfirmCallback = null;

function showCustomModal({ title, subtitle, iconType = "success", iconSymbol = "&#10003;", bodyHtml = "", buttonsHtml = "" }) {
    const overlay = document.getElementById("custom-modal-overlay");
    const titleEl = document.getElementById("modal-title");
    const subtitleEl = document.getElementById("modal-subtitle");
    const iconBadge = document.getElementById("modal-icon-badge");
    const bodyEl = document.getElementById("modal-body");
    const footerEl = document.getElementById("modal-footer");

    if (!overlay) return;

    if (titleEl) titleEl.innerText = title;
    if (subtitleEl) subtitleEl.innerHTML = subtitle;

    if (iconBadge) {
        iconBadge.className = `modal-icon-badge ${iconType}`;
        iconBadge.innerHTML = iconSymbol;
    }

    if (bodyEl) bodyEl.innerHTML = bodyHtml;
    if (footerEl) {
        if (buttonsHtml) {
            footerEl.innerHTML = buttonsHtml;
        } else {
            footerEl.innerHTML = `<button class="btn btn-primary modal-action-btn" onclick="closeCustomModal()">Tamam</button>`;
        }
    }

    overlay.classList.add("active");
}

function closeCustomModal() {
    const overlay = document.getElementById("custom-modal-overlay");
    if (overlay) {
        overlay.classList.remove("active");
    }
    currentModalConfirmCallback = null;
}

function handleModalBackdropClick(event) {
    if (event.target.id === "custom-modal-overlay") {
        closeCustomModal();
    }
}

function showCustomConfirm(title, message, onConfirm) {
    currentModalConfirmCallback = onConfirm;

    const bodyHtml = `
        <div style="font-size: 14px; color: #cbd5e1; line-height: 1.6; padding: 6px 0;">
            ${message}
        </div>
    `;

    const buttonsHtml = `
        <button class="btn modal-btn-secondary modal-action-btn" onclick="closeCustomModal()">İptal</button>
        <button class="btn btn-danger modal-action-btn" onclick="executeModalConfirm()">Durdur</button>
    `;

    showCustomModal({
        title: title,
        subtitle: "Onay Gerekiyor",
        iconType: "danger",
        iconSymbol: "&#9888;",
        bodyHtml: bodyHtml,
        buttonsHtml: buttonsHtml
    });
}

function executeModalConfirm() {
    if (typeof currentModalConfirmCallback === "function") {
        currentModalConfirmCallback();
    }
    closeCustomModal();
}

function showAssetCompletedModal(data) {
    const asset = data.asset || "Varlık";
    const elapsed = data.elapsed || 0;

    let cardsHtml = "";
    if (data.best && Object.keys(data.best).length > 0) {
        cardsHtml += `<div style="font-size: 13px; font-weight: 600; color: #94a3b8; margin-bottom: 4px;">Ufuk Bazında En İyi Modeller:</div>`;
        for (const [hKey, hData] of Object.entries(data.best)) {
            if (!hData) continue;
            const r2Val = hData.r2 !== undefined ? (typeof hData.r2 === 'number' ? hData.r2.toFixed(4) : hData.r2) : "-";
            const paramsStr = hData.params || "Varsayılan Parametreler";
            
            cardsHtml += `
                <div class="modal-best-card">
                    <div class="modal-best-card-header">
                        <span class="modal-horizon-tag">${hKey}</span>
                        <span class="modal-r2-pill">R² = ${r2Val}</span>
                    </div>
                    <div class="modal-model-title">${hData.model || "Bilinmiyor"}</div>
                    <div class="modal-params-box">${paramsStr}</div>
                </div>
            `;
        }
    } else {
        cardsHtml = `<div style="padding: 12px; color: #94a3b8;">Henüz model kaydı bulunamadı.</div>`;
    }

    cardsHtml += `
        <div style="font-size: 12px; color: #64748b; margin-top: 8px; display: flex; align-items: center; gap: 6px;">
            <span style="color: #10b981;">&#10003;</span>
            <span>Tüm en iyi modeller <b>grid_search_en_iyi_modeller.csv</b> dosyasına kaydedildi.</span>
        </div>
    `;

    showCustomModal({
        title: `${asset} Optimizasyonu Tamamlandı!`,
        subtitle: `Geçen Süre: ${elapsed} saniye &bull; 1.026 Deneme`,
        iconType: "success",
        iconSymbol: "&#10003;",
        bodyHtml: cardsHtml
    });
}

function showAllDoneModal(data) {
    let summaryHtml = `
        <div style="font-size: 14px; color: #e2e8f0; line-height: 1.6;">
            Tüm varlıklar (USD/TRY, EUR/TRY, Altın, Bitcoin, Apple, Tesla, THY) için toplam <b>7.182</b> model kombinasyonu test edildi.
        </div>
        <div style="margin-top: 14px; padding: 14px; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; color: #10b981; font-size: 13px;">
            <b>Kaydedilen Dosya:</b> <code>grid_search_en_iyi_modeller.csv</code><br>
            Tüm varlıkların 1 Ay, 3 Ay ve 6 Ay için en yüksek R² skoruna sahip en iyi model ve hiperparametreleri başarıyla diske yazıldı.
        </div>
    `;

    showCustomModal({
        title: "Tüm Grid Search Başarıyla Tamamlandı!",
        subtitle: "7 Varlık &bull; 6 Model &bull; 3 Ufuk &bull; Tam Kapsam",
        iconType: "success",
        iconSymbol: "&#9733;",
        bodyHtml: summaryHtml,
        buttonsHtml: `<button class="btn btn-primary modal-action-btn" onclick="closeCustomModal()">Harika!</button>`
    });
}

function showToast(message, type = "info", duration = 4000) {
    const container = document.getElementById("toast-container");
    if (!container) return;

    let iconSymbol = "&#8505;";
    if (type === "success") iconSymbol = "&#10003;";
    if (type === "warning") iconSymbol = "&#9888;";
    if (type === "danger") iconSymbol = "&#10007;";

    const toast = document.createElement("div");
    toast.className = `toast-item ${type}`;
    toast.innerHTML = `
        <span class="toast-icon">${iconSymbol}</span>
        <span>${message}</span>
    `;

    container.appendChild(toast);

    // Trigger animation
    setTimeout(() => {
        toast.classList.add("show");
    }, 10);

    // Auto remove
    setTimeout(() => {
        toast.classList.remove("show");
        setTimeout(() => {
            if (toast.parentElement) toast.parentElement.removeChild(toast);
        }, 300);
    }, duration);
}

// ============================================================
//  MODEL DETAIL & LIVE PROGRESS MODAL CONTROLLER
// ============================================================

let activeModalModel = null;
let activeModalTab = "logs";
const modelModalLogs = {};

function openModelDetailModal(modelName) {
    activeModalModel = modelName;
    const modal = document.getElementById("model-detail-modal-overlay");
    if (!modal) return;

    // Set titles & badges
    const titleEl = document.getElementById("md-title");
    if (titleEl) titleEl.innerText = `${modelName} Canlı Takip & Detay`;

    const assetBadge = document.getElementById("md-asset-badge");
    if (assetBadge) assetBadge.innerText = currentAsset;

    const isNeural = ["NBEATS", "NHITS", "PatchTST", "LSTM", "TFT", "TiDE"].includes(modelName);
    const devBadge = document.getElementById("md-device-badge");
    if (devBadge) {
        devBadge.innerText = isNeural ? "NVIDIA RTX 3070 (CUDA GPU)" : "AMD Ryzen 5 5600 (CPU)";
        devBadge.className = isNeural ? "badge badge-accent" : "badge";
    }

    const iconEl = document.getElementById("md-model-icon");
    if (iconEl) {
        iconEl.innerHTML = isNeural ? "&#9889;" : "&#9881;";
    }

    // Reset default view tab to logs
    switchModelModalTab("logs");

    // Fetch initial model detail
    fetch(`/api/model_detail/${encodeURIComponent(currentAsset)}/${encodeURIComponent(modelName)}`)
        .then(res => res.json())
        .then(data => {
            renderInitialModelDetail(data);
        })
        .catch(err => {
            console.error("Model detail fetch error:", err);
        });

    modal.classList.add("active");
    document.body.style.overflow = "hidden";
}

function closeModelDetailModal() {
    activeModalModel = null;
    const modal = document.getElementById("model-detail-modal-overlay");
    if (modal) {
        modal.classList.remove("active");
    }
    document.body.style.overflow = "";
}

function handleModelModalBackdropClick(event) {
    if (event.target && event.target.id === "model-detail-modal-overlay") {
        closeModelDetailModal();
    }
}

function switchModelModalTab(tabKey) {
    activeModalTab = tabKey;
    const btnLogs = document.getElementById("md-tab-logs-btn");
    const btnTrials = document.getElementById("md-tab-trials-btn");
    const tabLogs = document.getElementById("md-tab-logs");
    const tabTrials = document.getElementById("md-tab-trials");

    if (tabKey === "logs") {
        if (btnLogs) btnLogs.classList.add("active");
        if (btnTrials) btnTrials.classList.remove("active");
        if (tabLogs) tabLogs.style.display = "flex";
        if (tabTrials) tabTrials.style.display = "none";
    } else {
        if (btnLogs) btnLogs.classList.remove("active");
        if (btnTrials) btnTrials.classList.add("active");
        if (tabLogs) tabLogs.style.display = "none";
        if (tabTrials) tabTrials.style.display = "flex";
    }
}

function clearModelModalLogs() {
    if (!activeModalModel) return;
    const key = `${currentAsset}_${activeModalModel}`;
    modelModalLogs[key] = [];
    const body = document.getElementById("md-terminal-body");
    if (body) {
        body.innerHTML = `
            <div class="log-entry system-msg">
                <span class="log-time">[${new Date().toLocaleTimeString()}]</span> Model logları temizlendi.
            </div>
        `;
    }
}

function renderInitialModelDetail(data) {
    const { model_state, trials, logs, model, asset } = data;
    if (activeModalModel !== model || currentAsset !== asset) return;

    // 1. Overall stats
    const done = (model_state && model_state.done) || 0;
    const total = (model_state && model_state.total) || 1;
    const pct = Math.round((done / total) * 100);

    const doneEl = document.getElementById("md-done-count");
    if (doneEl) doneEl.innerText = done;

    const totalEl = document.getElementById("md-total-count");
    if (totalEl) totalEl.innerText = total;

    const pctBadge = document.getElementById("md-pct-badge");
    if (pctBadge) pctBadge.innerText = `%${pct}`;

    const overallBar = document.getElementById("md-overall-bar");
    if (overallBar) overallBar.style.width = `${pct}%`;

    // Status badge
    const statusEl = document.getElementById("md-status-badge");
    if (statusEl) {
        const isRunning = model_state && model_state.status === "calisiyor";
        const isQueued = model_state && model_state.status === "sirada";
        const isDone = (model_state && model_state.status === "tamamlandi") || done >= total;
        statusEl.innerText = isRunning ? "Çalışıyor" : (isQueued ? "Sırada" : (isDone ? "Tamamlandı" : "Bekliyor"));
        statusEl.className = "badge " + (isRunning ? "badge-accent" : (isQueued ? "badge-warning" : (isDone ? "badge-success" : "")));
    }

    // 2. Active Stage & Step Bar
    updateModelModalLiveElements(model_state, null);

    // 3. Render Logs
    const termBody = document.getElementById("md-terminal-body");
    const key = `${asset}_${model}`;
    const logList = (logs && logs.length > 0) ? logs : (modelModalLogs[key] || []);
    modelModalLogs[key] = logList;

    if (termBody) {
        if (logList.length === 0) {
            termBody.innerHTML = `
                <div class="log-entry system-msg">
                    <span class="log-time">[${new Date().toLocaleTimeString()}]</span> Henüz log kaydı yok. Model eğitim adımları ve deneme sonuçları burada canlı akacaktır.
                </div>
            `;
        } else {
            termBody.innerHTML = logList.map(line => `<div class="log-entry">${line}</div>`).join("");
            scrollModelTerminalToBottom();
        }
    }

    // 4. Render Trials Table
    renderModelTrialsTable(trials || []);
}

function updateModelModalLiveElements(model_state, step_data) {
    if (!model_state && !step_data) return;

    const stageEl = document.getElementById("md-stage-text");
    const stepBar = document.getElementById("md-step-bar");
    const stepVal = document.getElementById("md-step-val");
    const lossVal = document.getElementById("md-loss-val");
    const elapsedVal = document.getElementById("md-elapsed-val");
    const horizonVal = document.getElementById("md-horizon-val");
    const paramsVal = document.getElementById("md-params-val");

    const stage = (step_data && step_data.stage) || (model_state && model_state.stage) || "Bekliyor";
    if (stageEl) stageEl.innerText = stage;

    const step = (step_data && step_data.step !== undefined) ? step_data.step : (model_state ? model_state.step : 0);
    const maxSteps = (step_data && step_data.max_steps) || (model_state && model_state.max_steps) || 1;
    const stepPct = maxSteps > 0 ? Math.min(100, Math.round((step / maxSteps) * 100)) : 0;

    if (stepBar) stepBar.style.width = `${stepPct}%`;
    if (stepVal) stepVal.innerText = maxSteps > 1 ? `${step} / ${maxSteps} (%${stepPct})` : `${step} / ${maxSteps}`;
    
    const loss = (step_data && step_data.loss !== undefined && step_data.loss !== null) ? step_data.loss : (model_state ? model_state.loss : null);
    if (lossVal) lossVal.innerText = (loss !== null && loss !== undefined) ? loss : "--";

    const elapsed = (step_data && step_data.trial_elapsed !== undefined) ? step_data.trial_elapsed : (model_state ? model_state.trial_elapsed : 0);
    if (elapsedVal) elapsedVal.innerText = `${elapsed}s`;

    const horizon = (step_data && step_data.horizon) || (model_state && model_state.current_horizon) || "--";
    if (horizonVal) horizonVal.innerText = horizon;

    const params = (step_data && step_data.params) || (model_state && model_state.current_params) || "--";
    if (paramsVal) paramsVal.innerText = params;

    // Append log line if provided in step_data
    if (step_data && step_data.log_line) {
        appendLineToModelTerminal(step_data.log_line);
    }
}

function appendLineToModelTerminal(line) {
    if (!activeModalModel) return;
    const termBody = document.getElementById("md-terminal-body");
    if (!termBody) return;

    const key = `${currentAsset}_${activeModalModel}`;
    if (!modelModalLogs[key]) modelModalLogs[key] = [];
    modelModalLogs[key].push(line);

    const div = document.createElement("div");
    div.className = "log-entry";
    div.innerText = line;
    termBody.appendChild(div);

    scrollModelTerminalToBottom();
}

function scrollModelTerminalToBottom() {
    const chk = document.getElementById("md-autoscroll-chk");
    if (chk && !chk.checked) return;
    const termBody = document.getElementById("md-terminal-body");
    if (termBody) {
        termBody.scrollTop = termBody.scrollHeight;
    }
}

function renderModelTrialsTable(trials) {
    const tbody = document.getElementById("md-trials-tbody");
    const countEl = document.getElementById("md-trials-count");
    if (!tbody) return;

    if (countEl) countEl.innerText = trials.length;

    if (!trials || trials.length === 0) {
        tbody.innerHTML = `<tr class="empty-row"><td colspan="8">Bu model için henüz tamamlanmış deneme kaydı yok.</td></tr>`;
        return;
    }

    tbody.innerHTML = trials.map((t, idx) => {
        const r2Class = t.r2 >= 0.85 ? "r2-high" : (t.r2 >= 0.5 ? "r2-mid" : "r2-low");
        const r2Display = t.r2 <= -900 ? "Hata" : Number(t.r2).toFixed(4);
        const maeDisplay = Number(t.mae) >= 999 ? "-" : Number(t.mae).toFixed(2);
        const rmseDisplay = Number(t.rmse) >= 999 ? "-" : Number(t.rmse).toFixed(2);
        const mapeDisplay = Number(t.mape) >= 999 ? "-" : `%${Number(t.mape).toFixed(2)}`;

        return `
            <tr>
                <td><span class="badge badge-outline">#${idx + 1}</span></td>
                <td><b>${t.horizon}</b></td>
                <td><span class="r2-badge ${r2Class}">${r2Display}</span></td>
                <td>${maeDisplay}</td>
                <td>${rmseDisplay}</td>
                <td>${mapeDisplay}</td>
                <td>${t.elapsed || 0}s</td>
                <td><span class="params-code-cell">${t.params}</span></td>
            </tr>
        `;
    }).join("");
}

// ESC KEY LISTENER FOR ALL MODALS
document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
        if (activeModalModel) {
            closeModelDetailModal();
        } else {
            closeCustomModal();
        }
    }
});


