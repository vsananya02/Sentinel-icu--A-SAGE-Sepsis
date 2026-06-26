
# SEPSISAI DAY 19 - main.py  
# ALL BUGS FIXED:
#   ✅ 26-feature sub-scaler (was causing 500 on /predict)
#   ✅ GRU rebound rate stored as fraction
#   ✅ Attention labels use clinical time, not array index

import os, gc, json, pickle, time, threading
from datetime import datetime
from collections import defaultdict, deque
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

#  Paths
SAVE_DIR = '/content/drive/MyDrive/SepsisAI/'

# ── Constants — must match training exactly ───────────────────────
SEQ_LEN     = 24
GRU_SEQ_LEN = 48
MC_SAMPLES  = 20
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.3

# 30 features the scaler was fit on (MODEL_FEATURES_V5, exact order)
MODEL_FEATURES_V5 = [
    'hr','rr','sbp','dbp','o2_sat','lactate','wbc','creatinine','temp',
    'hrr_ratio','bp_ratio','lactate_risk','qsofa_proxy',
    'MAP','shock_risk','lactate_map_interact','sofa_proxy',
    'hour_sin','hour_cos','is_night',
    'temp_dysregulation','hypothermia_flag','fever_flag',
    'wbc_trend','hr_variability','low_hrv_flag',
    'creatinine_delta6','MAP_delta6','lactate_delta6','sofa_delta3',
]

# 26 features the BiLSTM actually uses (leaky 4 removed)
LEAKY = {'sofa_proxy','sofa_delta3','qsofa_proxy','shock_risk'}
LSTM_FEATURES = [f for f in MODEL_FEATURES_V5 if f not in LEAKY]

# Indices of the 26 LSTM features inside the 30-feature scaler
LSTM_INDICES = [MODEL_FEATURES_V5.index(f) for f in LSTM_FEATURES]

REBOUND_VITALS = ['MAP','hr','sbp','o2_sat','lactate','rr','temp','wbc','creatinine','dbp']
FLUID_FEATURES = ['MAP','hr','sbp','lactate','creatinine','hour_sin','hour_cos','is_night']

device = torch.device('cpu')   # CPU for stable API serving


# ═══════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURES  (identical to training)
# ═══════════════════════════════════════════════════════════════════
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, 64)
        self.v = nn.Linear(64, 1, bias=False)

    def forward(self, lstm_out):
        energy  = torch.tanh(self.W(lstm_out))
        scores  = self.v(energy)
        weights = torch.softmax(scores, dim=1)
        context = (weights * lstm_out).sum(dim=1)
        return context, weights.squeeze(-1)


class SepsisLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, 64), nn.LayerNorm(64),
            nn.GELU(), nn.Dropout(dropout * 0.5)
        )
        self.lstm = nn.LSTM(
            input_size=64, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.attention  = TemporalAttention(hidden_size * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128), nn.LayerNorm(128),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 32), nn.GELU(),
            nn.Dropout(dropout * 0.5), nn.Linear(32, 1)
        )

    def forward(self, x, return_attention=False):
        projected       = self.input_proj(x)
        lstm_out, _     = self.lstm(projected)
        lstm_out        = self.layer_norm(lstm_out)
        context, attn_w = self.attention(lstm_out)
        logits          = self.classifier(context).squeeze(-1)
        if return_attention:
            return logits, attn_w
        return logits

    def predict_with_uncertainty(self, x, n_samples=20):
        self.train()
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                preds.append(torch.sigmoid(self(x)).cpu().numpy())
        self.eval()
        preds = np.stack(preds, axis=0)
        return preds.mean(axis=0), preds.std(axis=0)


class ReboundGRU(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, 32), nn.LayerNorm(32), nn.GELU()
        )
        self.gru = nn.GRU(
            input_size=32, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.attention  = TemporalAttention(hidden_size * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 32), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(32, 1)
        )

    def forward(self, x, return_attention=False):
        x_proj    = self.input_proj(x)
        gru_out,_ = self.gru(x_proj)
        gru_out   = self.layer_norm(gru_out)
        context, attn_weights = self.attention(gru_out)
        logits    = self.classifier(context).squeeze(-1)
        if return_attention:
            return logits, attn_weights
        return logits


# ═══════════════════════════════════════════════════════════════════
# MODEL REGISTRY  (loads once at startup, shared across all requests)
# ═══════════════════════════════════════════════════════════════════
class ModelRegistry:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def load(self):
        if self._loaded:
            return
        print("[ModelRegistry] Loading all clinical layers...")
        t0 = time.time()

        # ── Layer 1: BiLSTM ───────────────────────────────────────
        ckpt = torch.load(
            os.path.join(SAVE_DIR, 'day18_lstm_layer1.pt'),
            map_location='cpu', weights_only=False
        )
        self.lstm_model = SepsisLSTM(len(LSTM_FEATURES), HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
        self.lstm_model.load_state_dict(ckpt['model_state_dict'])
        self.lstm_model.eval()
        self.threshold  = float(ckpt['threshold'])
        self.auroc_lstm = float(ckpt['auroc_test'])

        # Full 30-feature scaler (from checkpoint)
        scaler_30 = StandardScaler()
        scaler_30.mean_           = ckpt['scaler_mean']    # shape (30,)
        scaler_30.scale_          = ckpt['scaler_std']     # shape (30,)
        scaler_30.var_            = scaler_30.scale_ ** 2
        scaler_30.n_features_in_  = 30
        scaler_30.n_samples_seen_ = 100_000

        # ── BUG FIX: 26-feature sub-scaler ──────────────────────
        # Extract exact mean/scale for the 26 LSTM features only.
        # Avoids "expects 30 features but got 26" error.
        self.scaler_26 = StandardScaler()
        self.scaler_26.mean_           = scaler_30.mean_[LSTM_INDICES]
        self.scaler_26.scale_          = scaler_30.scale_[LSTM_INDICES]
        self.scaler_26.var_            = self.scaler_26.scale_ ** 2
        self.scaler_26.n_features_in_  = 26
        self.scaler_26.n_samples_seen_ = 100_000
        del ckpt; gc.collect()
        print(f"  ✓ BiLSTM loaded  AUROC {self.auroc_lstm:.4f}")
        print(f"  ✓ scaler_26 built (26 LSTM features, 4 leaky excluded)")

        # ── Layer 2: GRU Rebound ──────────────────────────────────
        gckpt = torch.load(
            os.path.join(SAVE_DIR, 'day18_gru_rebound.pt'),
            map_location='cpu', weights_only=False
        )
        self.gru_model = ReboundGRU(len(REBOUND_VITALS), 64, 2)
        self.gru_model.load_state_dict(gckpt['model_state_dict'])
        self.gru_model.eval()
        self.auroc_gru = float(gckpt['auroc_test'])
        del gckpt; gc.collect()
        print(f"  ✓ GRU loaded  AUROC {self.auroc_gru:.4f}")

        # Rebound scaler from cached meta
        meta_path = os.path.join(SAVE_DIR, '_r5_meta.pkl')
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            self.scaler_rebound = meta['scaler_rebound']
            print("  ✓ scaler_rebound loaded from _r5_meta.pkl")
        else:
            self.scaler_rebound = None
            print("  ⚠ scaler_rebound missing — GRU layer will return 0.0")

        # ── Layer 3: Fluid XGBoost ────────────────────────────────
        with open(os.path.join(SAVE_DIR, 'day18_fluid_real.pkl'), 'rb') as f:
            self.fluid_model = pickle.load(f)
        print("  ✓ Fluid XGBoost loaded")

        # ── Layer 4: Lactate lookup ───────────────────────────────
        lac_path = os.path.join(SAVE_DIR, 'day18_lactate_clearance.csv')
        lac_df   = pd.read_csv(lac_path)
        self.lac_lookup = lac_df.set_index('stay_id')[
            ['lactate_status', 'clearance_pct']
        ].to_dict('index')
        print(f"  ✓ Lactate clearance loaded ({len(lac_df):,} patients)")

        self._loaded = True
        print(f"[ModelRegistry] Ready in {time.time()-t0:.1f}s")

registry = ModelRegistry()


# ═══════════════════════════════════════════════════════════════════
# ALERT FATIGUE ENGINE
# ═══════════════════════════════════════════════════════════════════
class AlertFatigueEngine:
    """
    Three suppression rules:
    1. Cooldown: no re-alert within 4h unless risk rose >15 points
    2. Confidence gate: suppress if MC uncertainty >= 10%
    3. Signal specificity: must have >=1 named clinical signal
    """
    def __init__(self):
        self.last_alert_time = {}
        self.last_alert_risk = {}
        self.alert_log       = defaultdict(list)

    def should_fire(self, stay_id, risk_pct, uncertainty_pct, signals):
        # Rule 2 — confidence gate
        if uncertainty_pct >= 10.0:
            return False, f"Suppressed: uncertainty {uncertainty_pct:.1f}% ≥ 10% threshold"

        # Rule 3 — signal specificity
        named = [s for s in signals if "Subtle" not in s and "attention" not in s]
        if len(named) < 1:
            return False, "Suppressed: no specific clinical signals"

        # Rule 1 — cooldown
        now = time.time()
        if stay_id in self.last_alert_time:
            hours_since = (now - self.last_alert_time[stay_id]) / 3600
            risk_delta  = risk_pct - self.last_alert_risk.get(stay_id, 0)
            if hours_since < 4.0 and risk_delta < 15.0:
                return False, (f"Suppressed: last alert {hours_since:.1f}h ago, "
                               f"risk change only +{risk_delta:.1f}pp")

        self.last_alert_time[stay_id] = now
        self.last_alert_risk[stay_id] = risk_pct
        self.alert_log[stay_id].append({
            'timestamp':       datetime.now().isoformat(),
            'risk_pct':        risk_pct,
            'uncertainty_pct': uncertainty_pct,
            'signals':         signals
        })
        return True, "Alert fired"

    def get_history(self, stay_id):
        return self.alert_log.get(stay_id, [])

alert_engine = AlertFatigueEngine()


# ═══════════════════════════════════════════════════════════════════
# PREDICTION MONITOR  (MLOps drift detection foundation)
# ═══════════════════════════════════════════════════════════════════
class PredictionMonitor:
    def __init__(self, window=1000):
        self.predictions   = deque(maxlen=window)
        self.uncertainties = deque(maxlen=window)
        self._lock         = threading.Lock()

    def log(self, risk_pct, uncertainty_pct):
        with self._lock:
            self.predictions.append(risk_pct)
            self.uncertainties.append(uncertainty_pct)

    def get_stats(self):
        with self._lock:
            if len(self.predictions) < 10:
                return {"status": "insufficient_data", "n": len(self.predictions)}
            preds = np.array(self.predictions)
            uncs  = np.array(self.uncertainties)
            return {
                "n_predictions":    len(preds),
                "risk_mean":        float(np.mean(preds)),
                "risk_std":         float(np.std(preds)),
                "high_risk_rate":   float((preds >= 75).mean()),
                "uncertainty_mean": float(np.mean(uncs)),
                "uncertainty_p95":  float(np.percentile(uncs, 95)),
                "drift_flag":       bool(np.mean(uncs) > 15.0),
                "status":           "ok"
            }

monitor = PredictionMonitor()


# ═══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING SERVICE
# Mirrors Cell 1 exactly — same formulas, same column names.
# Takes raw vitals dict list → scaled (SEQ_LEN, 26) numpy array.
# ═══════════════════════════════════════════════════════════════════
def engineer_features(vitals_list: list) -> np.ndarray:
    df = pd.DataFrame(vitals_list)

    # Core derived features
    df['MAP']                 = (df['sbp'] + 2 * df['dbp']) / 3
    df['hrr_ratio']           = df['hr'] / (df['rr'] + 0.01)
    df['bp_ratio']            = df['sbp'] / (df['dbp'] + 0.01)
    df['lactate_risk']        = (df['lactate'] > 2.0).astype(int)
    df['lactate_map_interact']= df['lactate'] * (100 - df['MAP']) / 100
    df['hour_sin']            = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_cos']            = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    df['is_night']            = ((df['hour_of_day'] >= 22) |
                                 (df['hour_of_day'] <= 6)).astype(int)
    df['temp_dysregulation']  = (df['temp'] - 37.0).abs()
    df['hypothermia_flag']    = (df['temp'] < 36.0).astype(int)
    df['fever_flag']          = (df['temp'] > 38.3).astype(int)

    # Trend features
    n = len(df)
    df['wbc_trend']          = df['wbc'].diff(min(3, n-1)).fillna(0)
    df['hr_variability']     = df['hr'].rolling(min(4, n), min_periods=2).std().fillna(0)
    df['low_hrv_flag']       = (df['hr_variability'] < 3.0).astype(int)
    df['creatinine_delta6']  = df['creatinine'].diff(min(6, n-1)).fillna(0)
    df['MAP_delta6']         = df['MAP'].diff(min(6, n-1)).fillna(0)
    df['lactate_delta6']     = df['lactate'].diff(min(6, n-1)).fillna(0)

    feats = df[LSTM_FEATURES].values.astype(np.float32)

    # ── BUG FIX: use scaler_26, not scaler_30 ────────────────────
    feats_scaled = registry.scaler_26.transform(feats)

    # Pad or truncate to SEQ_LEN
    n = len(feats_scaled)
    if n < SEQ_LEN:
        pad          = np.zeros((SEQ_LEN - n, 26), dtype=np.float32)
        feats_scaled = np.vstack([pad, feats_scaled])
    else:
        feats_scaled = feats_scaled[-SEQ_LEN:]

    return feats_scaled   # (24, 26)


def compute_signals(v: dict) -> list:
    """Extract named clinical signals from a single vital-sign dict."""
    signals = []
    map_val = (v['sbp'] + 2 * v['dbp']) / 3
    if v['temp'] > 38.3:
        signals.append(f"Fever {v['temp']:.1f}°C ↑ (inflammatory)")
    if v['temp'] < 36.0:
        signals.append(f"Hypothermia {v['temp']:.1f}°C ↓ (danger)")
    if v['creatinine'] > 1.5:
        signals.append(f"Creatinine {v['creatinine']:.1f} ↑ (renal injury)")
    if map_val < 70:
        signals.append(f"MAP {map_val:.0f} mmHg ↓ (hypotension)")
    if v['lactate'] > 2.0:
        signals.append(f"Lactate {v['lactate']:.1f} ↑ (hypoperfusion)")
    if v['wbc'] > 12 or v['wbc'] < 4:
        signals.append(f"WBC {v['wbc']:.1f} (immune dysregulation)")
    if not signals:
        signals.append("Subtle temporal pattern — see attention chart")
    return signals


def interpret_risk(risk_pct: float):
    if risk_pct >= 75:
        return "CRITICAL", "🔴", "INITIATE SEPSIS PROTOCOL — Blood cultures + Antibiotics NOW"
    elif risk_pct >= 55:
        return "ALERT", "🟠", "ESCALATE MONITORING — Repeat lactate in 2h, review antibiotics"
    elif risk_pct >= 35:
        return "WATCH", "🟡", "INCREASED VIGILANCE — Recheck vitals hourly, alert attending"
    else:
        return "MONITOR", "🟢", "ROUTINE MONITORING — Continue standard ICU protocol"


def run_gru_rebound(vitals_list: list) -> float:
    """Run GRU rebound on raw vitals. Returns probability 0-100."""
    if registry.scaler_rebound is None:
        return 0.0
    try:
        df_r     = pd.DataFrame(vitals_list)
        df_r['MAP'] = (df_r['sbp'] + 2 * df_r['dbp']) / 3
        rb_raw   = df_r[REBOUND_VITALS].values.astype(np.float32)
        rb_sc    = registry.scaler_rebound.transform(rb_raw)
        n        = len(rb_sc)
        if n < GRU_SEQ_LEN:
            pad   = np.zeros((GRU_SEQ_LEN - n, 10), dtype=np.float32)
            rb_seq = np.vstack([pad, rb_sc])
        else:
            rb_seq = rb_sc[-GRU_SEQ_LEN:]
        with torch.no_grad():
            logit = registry.gru_model(
                torch.FloatTensor(rb_seq).unsqueeze(0)
            )
            return float(torch.sigmoid(logit).item()) * 100
    except Exception as e:
        print(f"  ⚠ GRU rebound error: {e}")
        return 0.0


def run_fluid(v: dict) -> float:
    """Run fluid responsiveness XGBoost on current vitals."""
    map_val  = (v['sbp'] + 2 * v['dbp']) / 3
    hour_sin = np.sin(2 * np.pi * v['hour_of_day'] / 24)
    hour_cos = np.cos(2 * np.pi * v['hour_of_day'] / 24)
    is_night = int(v['hour_of_day'] >= 22 or v['hour_of_day'] <= 6)
    fluid_in = pd.DataFrame([{
        'MAP': map_val, 'hr': v['hr'], 'sbp': v['sbp'],
        'lactate': v['lactate'], 'creatinine': v['creatinine'],
        'hour_sin': hour_sin, 'hour_cos': hour_cos, 'is_night': is_night
    }])
    return float(registry.fluid_model.predict_proba(
        fluid_in[FLUID_FEATURES])[:, 1][0]) * 100


def run_lactate(stay_id: str):
    """Look up lactate status from pre-computed table."""
    row = registry.lac_lookup.get(stay_id, {})
    return (
        row.get('lactate_status', 'UNKNOWN'),
        float(row.get('clearance_pct', 0.0))
    )


# ═══════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS
# ═══════════════════════════════════════════════════════════════════
class VitalHour(BaseModel):
    hr:          float = Field(..., ge=20,  le=300,  description="Heart rate (bpm)")
    rr:          float = Field(..., ge=4,   le=60,   description="Respiratory rate (br/min)")
    sbp:         float = Field(..., ge=40,  le=300,  description="Systolic BP (mmHg)")
    dbp:         float = Field(..., ge=20,  le=200,  description="Diastolic BP (mmHg)")
    o2_sat:      float = Field(..., ge=50,  le=100,  description="SpO2 (%)")
    lactate:     float = Field(..., ge=0,   le=30,   description="Serum lactate (mmol/L)")
    wbc:         float = Field(..., ge=0,   le=100,  description="WBC (×10³/µL)")
    creatinine:  float = Field(..., ge=0,   le=20,   description="Creatinine (mg/dL)")
    temp:        float = Field(..., ge=30,  le=43,   description="Temperature (°C)")
    hour_of_day: int   = Field(..., ge=0,   le=23,   description="Hour of day (0-23)")

class SingleHourRequest(BaseModel):
    stay_id: str
    vitals:  VitalHour

class SequenceRequest(BaseModel):
    stay_id:    str
    vitals_24h: List[VitalHour] = Field(..., min_length=2, max_length=24)

class ClinicalReport(BaseModel):
    stay_id:              str
    timestamp:            str
    sepsis_risk_pct:      float
    uncertainty_pct:      float
    alert_tier:           str
    alert_icon:           str
    recommendation:       str
    top_signals:          List[str]
    rebound_risk_pct:     float
    fluid_responsive_pct: float
    lactate_status:       str
    lactate_clearance:    float
    alert_fired:          bool
    alert_reason:         str
    inference_ms:         float


# ═══════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════════
app = FastAPI(
    title="SepsisAI Clinical Intelligence API",
    description=(
        "Four-layer sepsis detection: BiLSTM (AUROC 0.9796) + GRU Rebound (0.8702) "
        "+ Fluid XGBoost (0.8295) + Lactate Clearance. "
        "Trained on 91,791 MIMIC-IV ICU patients."
    ),
    version="1.0.0-day19"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    registry.load()


# ── /health ───────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status":        "operational",
        "model_loaded":  registry._loaded,
        "lstm_auroc":    registry.auroc_lstm if registry._loaded else None,
        "gru_auroc":     registry.auroc_gru  if registry._loaded else None,
        "threshold":     registry.threshold  if registry._loaded else None,
        "training_data": "MIMIC-IV v3.1 · 91,791 ICU patients",
        "timestamp":     datetime.now().isoformat(),
        "monitoring":    monitor.get_stats()
    }


# ── /predict  (single-hour, fast path) ───────────────────────────
@app.post("/predict", response_model=ClinicalReport)
async def predict_single(request: SingleHourRequest):
    if not registry._loaded:
        raise HTTPException(503, "Models still loading. Retry in 30s.")

    t0 = time.time()
    v  = request.vitals.dict()

    # Repeat current vital 24 times (pad to full sequence)
    feats = engineer_features([v] * SEQ_LEN)
    X     = torch.FloatTensor(feats).unsqueeze(0)

    mean, std = registry.lstm_model.predict_with_uncertainty(X, MC_SAMPLES)
    risk_pct  = float(mean[0]) * 100
    unc_pct   = float(std[0])  * 100

    signals           = compute_signals(v)
    tier, icon, rec   = interpret_risk(risk_pct)
    fluid_pct         = run_fluid(v)
    lac_status, lac_c = run_lactate(request.stay_id)
    fired, reason     = alert_engine.should_fire(
        request.stay_id, risk_pct, unc_pct, signals
    )
    monitor.log(risk_pct, unc_pct)

    return ClinicalReport(
        stay_id=request.stay_id,
        timestamp=datetime.now().isoformat(),
        sepsis_risk_pct=round(risk_pct, 1),
        uncertainty_pct=round(unc_pct, 1),
        alert_tier=tier, alert_icon=icon, recommendation=rec,
        top_signals=signals[:3],
        rebound_risk_pct=0.0,   # needs sequence — use /predict/sequence
        fluid_responsive_pct=round(fluid_pct, 1),
        lactate_status=lac_status,
        lactate_clearance=round(lac_c, 1),
        alert_fired=fired, alert_reason=reason,
        inference_ms=round((time.time()-t0)*1000, 1)
    )


# ── /predict/sequence  (full 24h BiLSTM + all 4 layers) ──────────
@app.post("/predict/sequence", response_model=ClinicalReport)
async def predict_sequence(request: SequenceRequest):
    if not registry._loaded:
        raise HTTPException(503, "Models still loading. Retry in 30s.")

    t0          = time.time()
    vitals_list = [v.dict() for v in request.vitals_24h]
    last        = vitals_list[-1]

    # Layer 1: BiLSTM + attention
    feats = engineer_features(vitals_list)
    X     = torch.FloatTensor(feats).unsqueeze(0)
    mean, std = registry.lstm_model.predict_with_uncertainty(X, MC_SAMPLES)
    risk_pct  = float(mean[0]) * 100
    unc_pct   = float(std[0])  * 100

    registry.lstm_model.eval()
    with torch.no_grad():
        _, attn     = registry.lstm_model(X, return_attention=True)
    attn_vals       = attn.squeeze(0).cpu().numpy()          # (24,)
    peak_idx        = int(np.argmax(attn_vals))
    hours_before    = (SEQ_LEN - 1) - peak_idx               # clinical time offset

    # Layer 2: GRU Rebound
    rebound_pct = run_gru_rebound(vitals_list)

    # Layers 3 + 4
    fluid_pct         = run_fluid(last)
    lac_status, lac_c = run_lactate(request.stay_id)

    # Signals — include attention lead time as first signal if meaningful
    signals = compute_signals(last)
    if hours_before > 0:
        signals.insert(0, f"Attention peak {hours_before}h before current time")

    tier, icon, rec = interpret_risk(risk_pct)
    fired, reason   = alert_engine.should_fire(
        request.stay_id, risk_pct, unc_pct, signals
    )
    monitor.log(risk_pct, unc_pct)

    return ClinicalReport(
        stay_id=request.stay_id,
        timestamp=datetime.now().isoformat(),
        sepsis_risk_pct=round(risk_pct, 1),
        uncertainty_pct=round(unc_pct, 1),
        alert_tier=tier, alert_icon=icon, recommendation=rec,
        top_signals=signals[:3],
        rebound_risk_pct=round(rebound_pct, 1),
        fluid_responsive_pct=round(fluid_pct, 1),
        lactate_status=lac_status,
        lactate_clearance=round(lac_c, 1),
        alert_fired=fired, alert_reason=reason,
        inference_ms=round((time.time()-t0)*1000, 1)
    )


# ── /patient/{stay_id}/history ────────────────────────────────────
@app.get("/patient/{stay_id}/history")
async def patient_history(stay_id: str):
    return {
        "stay_id":     stay_id,
        "alert_count": len(alert_engine.get_history(stay_id)),
        "alerts":      alert_engine.get_history(stay_id),
        "monitoring":  monitor.get_stats()
    }
