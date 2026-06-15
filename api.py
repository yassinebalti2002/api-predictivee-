"""
API Maintenance Prédictive — v3.1.0
=====================================
POST /fleet           → Dashboard complet (vue parc + détail capteurs)
POST /predict         → Prédiction capteur unique
POST /predict-rul     → RUL capteur unique
POST /predict-batch   → Lot de capteurs
GET  /health          → Statut API
GET  /docs            → Swagger interactif
"""

import json
import joblib
import numpy as np
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn


class PrettyJSONResponse(JSONResponse):
    """Réponse JSON indentée avec accents préservés et sans valeurs nulles."""
    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            separators=(",", ": "),
        ).encode("utf-8")

warnings.filterwarnings("ignore")

# ── Chargement des modèles ─────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).parent / "models"

print("Chargement des modèles...")

scaler      = joblib.load(MODEL_DIR / "scaler_v3.pkl")
pca         = joblib.load(MODEL_DIR / "pca_v3.pkl")
features    = joblib.load(MODEL_DIR / "features_v3.pkl")
thresholds  = joblib.load(MODEL_DIR / "threshold_v3.pkl")
model_if    = joblib.load(MODEL_DIR / "model_if_v3.pkl")
model_lof   = joblib.load(MODEL_DIR / "model_lof_v3.pkl")
model_ocsvm = joblib.load(MODEL_DIR / "model_ocsvm_v3.pkl")
model_ecod  = joblib.load(MODEL_DIR / "model_ecod_v3.pkl")

try:
    model_hbos  = joblib.load(MODEL_DIR / "model_hbos_v3.pkl")
    model_copod = joblib.load(MODEL_DIR / "model_copod_v3.pkl")
    ENSEMBLE_6 = True
except Exception:
    ENSEMBLE_6 = False

try:
    scaler_rul   = joblib.load(MODEL_DIR / "scaler_rul_v1.pkl")
    features_rul = joblib.load(MODEL_DIR / "features_rul_v1.pkl")
    model_rul    = joblib.load(MODEL_DIR / "model_rul_v1.pkl")
    RUL_OK = True
except Exception:
    RUL_OK = False

MODELS_NAMES = "IF-LOF-OCSVM-ECOD" + ("-HBOS-COPOD" if ENSEMBLE_6 else "")
print(f"Modèles chargés — {MODELS_NAMES} — RUL: {RUL_OK}")

# ── Application ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="API Maintenance Prédictive",
    description="Détection d'anomalies et estimation RUL pour capteurs IoT industriels",
    version="3.1.0",
    default_response_class=PrettyJSONResponse,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Ordre des features (31) ────────────────────────────────────────────────────
FEATURES_ORDER = [
    "temp_mean", "temp_std", "temp_trend", "temp_cur",
    "vib_z_mean", "vib_z_std", "vib_z_rms_w", "vib_z_kurt", "vib_z_crest", "vib_z_cur",
    "vib_x_mean", "vib_x_std", "vib_x_rms_w", "vib_x_kurt",
    "vib_y_mean", "vib_y_std", "vib_y_rms_w", "vib_y_kurt",
    "vib_total", "health_score",
    "acc_p2p", "acc_z2p", "acc_crest", "acc_rms",
    "current_mean",
    "delta_vib", "delta_temp", "vib_entropy", "fft_ratio",
    "vib_asym_xy", "vib_asym_xz",
]

# ══════════════════════════════════════════════════════════════════════════════
# SCHÉMAS DE REQUÊTE
# ══════════════════════════════════════════════════════════════════════════════

class SensorFeatures(BaseModel):
    """31 features extraites d'une fenêtre de mesures capteur."""
    temp_mean:    float = Field(..., description="Température moyenne (°C)")
    temp_std:     float = Field(..., description="Écart-type température")
    temp_trend:   float = Field(0.0,   description="Tendance température")
    temp_cur:     float = Field(..., description="Température actuelle (°C)")

    vib_z_mean:   float = Field(..., description="Vibration Z moyenne (mg)")
    vib_z_std:    float = Field(..., description="Vibration Z écart-type")
    vib_z_rms_w:  float = Field(..., description="Vibration Z RMS pondéré")
    vib_z_kurt:   float = Field(0.0,   description="Kurtosis vibration Z")
    vib_z_crest:  float = Field(1.0,   description="Facteur de crête Z")
    vib_z_cur:    float = Field(..., description="Vibration Z actuelle")

    vib_x_mean:   float = Field(..., description="Vibration X moyenne (mg)")
    vib_x_std:    float = Field(0.0,   description="Vibration X écart-type")
    vib_x_rms_w:  float = Field(..., description="Vibration X RMS")
    vib_x_kurt:   float = Field(0.0,   description="Kurtosis vibration X")

    vib_y_mean:   float = Field(..., description="Vibration Y moyenne (mg)")
    vib_y_std:    float = Field(0.0,   description="Vibration Y écart-type")
    vib_y_rms_w:  float = Field(..., description="Vibration Y RMS")
    vib_y_kurt:   float = Field(0.0,   description="Kurtosis vibration Y")

    vib_total:    float = Field(..., description="Vibration totale combinée (mg)")
    health_score: float = Field(100.0, description="Score santé brut [0-100]")

    acc_p2p:      float = Field(..., description="Accélération peak-to-peak")
    acc_z2p:      float = Field(..., description="Accélération zero-to-peak")
    acc_crest:    float = Field(1.0,   description="Facteur de crête accélération")
    acc_rms:      float = Field(..., description="Accélération RMS")

    current_mean: float = Field(..., description="Courant moyen (A)")

    delta_vib:    float = Field(0.0, description="Variation vibration intra-fenêtre")
    delta_temp:   float = Field(0.0, description="Variation température intra-fenêtre")
    vib_entropy:  float = Field(0.0, description="Entropie signal vibration")
    fft_ratio:    float = Field(0.0, description="Ratio FFT (périodicité anormale)")
    vib_asym_xy:  float = Field(0.0, description="Asymétrie vibration X/Y")
    vib_asym_xz:  float = Field(0.0, description="Asymétrie vibration X/Z")


class CapteurInput(BaseModel):
    """Un capteur avec sa lecture courante et son historique optionnel pour les graphiques."""
    id: str = Field(..., description="Identifiant unique du capteur (ex: 8f7f2f7e)")
    features: SensorFeatures = Field(..., description="Lecture courante — 31 features")
    historique: Optional[List[SensorFeatures]] = Field(
        None,
        description="Dernières N lectures pour alimenter les graphiques (optionnel, max 50)"
    )


class FleetRequest(BaseModel):
    """Requête flotte : envoie tous les capteurs actifs d'un coup."""
    capteurs: List[CapteurInput] = Field(..., description="Liste des capteurs actifs")
    total_installes: Optional[int] = Field(
        None,
        description="Nombre total de capteurs installés (si différent des actifs envoyés)"
    )


class RULRequest(BaseModel):
    features: dict = Field(..., description="Dictionnaire {feature_name: valeur}")


# ══════════════════════════════════════════════════════════════════════════════
# SCHÉMAS DE RÉPONSE
# ══════════════════════════════════════════════════════════════════════════════

class MesuresCapteur(BaseModel):
    """Mesures brutes affichées dans le panneau capteur du dashboard."""
    temperature_c:    float
    vib_z_rms_mg:    float
    vib_x_rms_mg:    float
    vib_y_rms_mg:    float
    vib_totale_mg:   float
    kurtosis_z:      float
    degradation_pct: float
    tendance_vib:    str


class RULDetail(BaseModel):
    """Remaining Useful Life avec statut et dégradation."""
    heures:          float
    jours:           float
    statut:          str
    degradation_pct: float
    alerte:          str   # OK / STABLE / ATTENTION / CRITIQUE


class CapteurResult(BaseModel):
    """Résultat complet pour un capteur — alimente toutes les zones du dashboard."""
    id:              str
    anomalie:        bool
    score_anomalie:  float = Field(..., description="[0-1] plus haut = plus anormal")
    score_sante:     float = Field(..., description="[0-100] plus bas = plus dégradé")
    niveau_alerte:   str   = Field(..., description="OK / WARNING / CRITICAL")
    niveau_risque:   str   = Field(..., description="FAIBLE / MODÉRÉ / HAUTE")
    votes:           Dict[str, str] = Field(..., description="Vote de chaque modèle")
    vote_resume:     str   = Field(..., description="Ex: 3/4 votes · 75%")
    iteration:       int   = Field(..., description="Numéro d'itération dans la requête")
    rul:             Optional[RULDetail] = None
    mesures:         MesuresCapteur

    # Données pour les graphiques (présentes si historique fourni en entrée)
    historique_scores:       Optional[List[float]] = Field(None, description="Historique scores anomalie pour le graphique")
    historique_temperatures: Optional[List[float]] = Field(None, description="Historique températures")
    historique_vib_z:        Optional[List[float]] = Field(None, description="Historique vibrations axe Z")
    historique_vib_x:        Optional[List[float]] = Field(None, description="Historique vibrations axe X")
    historique_vib_y:        Optional[List[float]] = Field(None, description="Historique vibrations axe Y")


class VueGlobale(BaseModel):
    """Statistiques agrégées du parc — barre du haut du dashboard."""
    capteurs_actifs:   int
    total_installes:   int
    en_anomalie:       int
    taux_anomalie_pct: float
    sante_moyenne:     float
    rul_minimum:       Dict[str, Any] = Field(..., description="{valeur_h, capteur_id}")
    temp_max_parc:     Dict[str, Any] = Field(..., description="{valeur_c, capteur_id}")
    vib_max_parc:      Dict[str, Any] = Field(..., description="{valeur_mg, capteur_id}")


class FleetResponse(BaseModel):
    """Réponse complète du dashboard — une seule requête alimente tout l'écran."""
    timestamp:   str
    modeles:     str
    version_api: str
    vue_globale: VueGlobale
    capteurs:    List[CapteurResult]


class PredictResponse(BaseModel):
    anomaly:       bool
    anomaly_score: float
    health_score:  float
    votes:         dict
    confidence:    str
    alert_level:   str


class RULResponse(BaseModel):
    rul_hours:  float
    rul_days:   float
    confidence: str
    status:     str


# ══════════════════════════════════════════════════════════════════════════════
# FONCTIONS INTERNES
# ══════════════════════════════════════════════════════════════════════════════

def _preprocess(data: SensorFeatures) -> np.ndarray:
    row = [getattr(data, f) for f in FEATURES_ORDER]
    return scaler.transform(np.array(row, dtype=float).reshape(1, -1))


def _run_models(X: np.ndarray) -> Dict[str, int]:
    votes: Dict[str, int] = {
        "isolation_forest": int(model_if.predict(X)[0]),
        "lof":              int(model_lof.predict(X)[0]),
        "ocsvm":            int(model_ocsvm.predict(X)[0]),
        "ecod":             int(model_ecod.predict(X)[0]),
    }
    if ENSEMBLE_6:
        votes["hbos"]  = int(model_hbos.predict(X)[0])
        votes["copod"] = int(model_copod.predict(X)[0])
    return votes


def _compute_rul(feat: SensorFeatures) -> Optional[RULDetail]:
    if not RUL_OK:
        return None
    try:
        row = [getattr(feat, f, 0.0) if f in FEATURES_ORDER else 0.0 for f in features_rul]
        X_s = scaler_rul.transform(np.array(row, dtype=float).reshape(1, -1))
        rul_h = max(0.0, float(model_rul.predict(X_s)[0]))
        deg   = round(min(100.0, max(0.0, (1 - rul_h / 8760) * 100)), 1)
        if rul_h < 24:
            statut, alerte = "CRITIQUE — remplacement urgent", "CRITIQUE"
        elif rul_h < 168:
            statut, alerte = "ATTENTION — maintenance dans la semaine", "ATTENTION"
        elif rul_h < 720:
            statut, alerte = "STABLE — surveillance recommandée", "STABLE"
        else:
            statut, alerte = "BON — aucune action requise", "OK"
        return RULDetail(heures=round(rul_h, 1), jours=round(rul_h / 24, 1), statut=statut,
                         degradation_pct=deg, alerte=alerte)
    except Exception:
        return None


def _niveau_risque(score_sante: float) -> str:
    if score_sante >= 70:
        return "FAIBLE"
    if score_sante >= 40:
        return "MODÉRÉ"
    return "HAUTE"


def _tendance_vib(delta_vib: float) -> str:
    if delta_vib > 2.0:
        return "Hausse"
    if delta_vib < -2.0:
        return "Baisse"
    return "Stable"


def _predict_one(capteur: CapteurInput, iteration: int) -> CapteurResult:
    feat = capteur.features
    X = _preprocess(feat)
    votes_raw = _run_models(X)

    n_anomaly = sum(1 for v in votes_raw.values() if v == -1)
    n_total   = len(votes_raw)
    anomaly   = n_anomaly >= (n_total // 2 + 1)
    score     = round(n_anomaly / n_total, 3)
    health    = round(max(0.0, min(100.0, (1.0 - score) * 100)), 1)
    deg       = round(100.0 - health, 1)

    niveau_alerte = "CRITICAL" if score > 0.75 else ("WARNING" if score > 0.45 else "OK")
    votes_readable = {k: ("ANOMALIE" if v == -1 else "normal") for k, v in votes_raw.items()}

    mesures = MesuresCapteur(
        temperature_c=round(feat.temp_cur, 1),
        vib_z_rms_mg=round(feat.vib_z_rms_w, 1),
        vib_x_rms_mg=round(feat.vib_x_rms_w, 1),
        vib_y_rms_mg=round(feat.vib_y_rms_w, 1),
        vib_totale_mg=round(feat.vib_total, 1),
        kurtosis_z=round(feat.vib_z_kurt, 2),
        degradation_pct=deg,
        tendance_vib=_tendance_vib(feat.delta_vib),
    )

    # Graphiques historiques (seulement si l'appelant envoie un historique)
    hist_scores = hist_temps = hist_vz = hist_vx = hist_vy = None
    if capteur.historique:
        hist_data = capteur.historique[-20:]
        hist_scores, hist_temps, hist_vz, hist_vx, hist_vy = [], [], [], [], []
        for h in hist_data:
            Xh = _preprocess(h)
            vh = _run_models(Xh)
            sh = round(sum(1 for v in vh.values() if v == -1) / len(vh), 3)
            hist_scores.append(sh)
            hist_temps.append(round(h.temp_cur, 1))
            hist_vz.append(round(h.vib_z_rms_w, 1))
            hist_vx.append(round(h.vib_x_rms_w, 1))
            hist_vy.append(round(h.vib_y_rms_w, 1))
        # Ajoute la lecture courante à la fin
        hist_scores.append(score)
        hist_temps.append(round(feat.temp_cur, 1))
        hist_vz.append(round(feat.vib_z_rms_w, 1))
        hist_vx.append(round(feat.vib_x_rms_w, 1))
        hist_vy.append(round(feat.vib_y_rms_w, 1))

    return CapteurResult(
        id=capteur.id,
        anomalie=anomaly,
        score_anomalie=score,
        score_sante=health,
        niveau_alerte=niveau_alerte,
        niveau_risque=_niveau_risque(health),
        votes=votes_readable,
        vote_resume=f"{n_anomaly}/{n_total} votes · {round(score * 100)}%",
        iteration=iteration,
        rul=_compute_rul(feat),
        mesures=mesures,
        historique_scores=hist_scores,
        historique_temperatures=hist_temps,
        historique_vib_z=hist_vz,
        historique_vib_x=hist_vx,
        historique_vib_y=hist_vy,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

# Données de démonstration pour les 19 capteurs du dashboard
_DEMO_CAPTEURS = [
    {"id": "8f7f2f7e",  "temp": 90.0, "vib": 140.0, "cur": 8.5,  "kurt": 4.5,  "delta_v": 12.0, "hs": 15.0},
    {"id": "0ff416d2",  "temp": 17.9, "vib": 5.5,   "cur": 2.0,  "kurt": 0.20, "delta_v": 0.1,  "hs": 91.0},
    {"id": "b2acdf45",  "temp": 18.3, "vib": 5.0,   "cur": 2.1,  "kurt": 0.22, "delta_v": 0.2,  "hs": 90.0},
    {"id": "6e0c1740",  "temp": 18.5, "vib": 6.1,   "cur": 2.3,  "kurt": 0.30, "delta_v": 0.3,  "hs": 88.0},
    {"id": "07da47b8",  "temp": 18.6, "vib": 4.8,   "cur": 2.0,  "kurt": 0.18, "delta_v": 0.1,  "hs": 92.0},
    {"id": "aa7b02a1",  "temp": 18.3, "vib": 6.2,   "cur": 2.2,  "kurt": 0.28, "delta_v": 0.4,  "hs": 89.0},
    {"id": "eb084747",  "temp": 18.4, "vib": 3.8,   "cur": 1.9,  "kurt": 0.15, "delta_v": 0.1,  "hs": 94.0},
    {"id": "2c6254af",  "temp": 17.9, "vib": 5.8,   "cur": 2.1,  "kurt": 0.25, "delta_v": 0.2,  "hs": 90.0},
    {"id": "a6a46be1",  "temp": 18.1, "vib": 5.0,   "cur": 2.0,  "kurt": 0.20, "delta_v": 0.2,  "hs": 91.0},
    {"id": "f3c91d20",  "temp": 55.0, "vib": 42.0,  "cur": 5.1,  "kurt": 2.10, "delta_v": 3.5,  "hs": 45.0},
    {"id": "c7e83b11",  "temp": 18.2, "vib": 4.9,   "cur": 2.0,  "kurt": 0.21, "delta_v": 0.1,  "hs": 92.0},
    {"id": "d94f0a3e",  "temp": 70.0, "vib": 88.0,  "cur": 7.2,  "kurt": 3.80, "delta_v": 8.0,  "hs": 22.0},
    {"id": "1b2e5c77",  "temp": 17.8, "vib": 5.2,   "cur": 2.1,  "kurt": 0.23, "delta_v": 0.2,  "hs": 91.0},
    {"id": "8a0d4f62",  "temp": 18.0, "vib": 5.6,   "cur": 2.2,  "kurt": 0.26, "delta_v": 0.3,  "hs": 90.0},
    {"id": "3f7c9a1d",  "temp": 62.0, "vib": 65.0,  "cur": 6.3,  "kurt": 3.00, "delta_v": 5.0,  "hs": 35.0},
    {"id": "e5b2083c",  "temp": 18.3, "vib": 4.7,   "cur": 1.9,  "kurt": 0.19, "delta_v": 0.1,  "hs": 93.0},
    {"id": "9c1d6e4a",  "temp": 18.1, "vib": 5.3,   "cur": 2.1,  "kurt": 0.24, "delta_v": 0.2,  "hs": 91.0},
    {"id": "7f4b2c8e",  "temp": 18.5, "vib": 5.9,   "cur": 2.3,  "kurt": 0.27, "delta_v": 0.3,  "hs": 89.0},
    {"id": "2d8e5f1b",  "temp": 18.2, "vib": 4.6,   "cur": 2.0,  "kurt": 0.18, "delta_v": 0.1,  "hs": 93.0},
]


def _demo_features(c: dict) -> SensorFeatures:
    t, v, cur, k, dv, hs = c["temp"], c["vib"], c["cur"], c["kurt"], c["delta_v"], c["hs"]
    vz = round(v * 0.70, 1)
    vx = round(v * 0.53, 1)
    vy = round(v * 0.50, 1)
    return SensorFeatures(
        temp_mean=round(t * 0.98, 1), temp_std=round(t * 0.05, 2),
        temp_trend=round(dv * 0.1, 2), temp_cur=t,
        vib_z_mean=round(vz * 0.97, 1), vib_z_std=round(vz * 0.10, 2),
        vib_z_rms_w=vz, vib_z_kurt=k, vib_z_crest=round(1.2 + k * 0.3, 2), vib_z_cur=vz,
        vib_x_mean=round(vx * 0.97, 1), vib_x_std=round(vx * 0.08, 2),
        vib_x_rms_w=vx, vib_x_kurt=round(k * 0.85, 2),
        vib_y_mean=round(vy * 0.97, 1), vib_y_std=round(vy * 0.08, 2),
        vib_y_rms_w=vy, vib_y_kurt=round(k * 0.78, 2),
        vib_total=v, health_score=hs,
        acc_p2p=round(v * 1.4, 1), acc_z2p=round(v * 0.7, 1),
        acc_crest=round(1.2 + k * 0.3, 2), acc_rms=vz,
        current_mean=cur,
        delta_vib=dv, delta_temp=round(dv * 0.3, 2),
        vib_entropy=round(0.5 + k * 0.5, 2), fft_ratio=round(k * 0.4, 2),
        vib_asym_xy=round(abs(vx - vy) / max(v, 1), 3),
        vib_asym_xz=round(abs(vx - vz) / max(v, 1), 3),
    )


@app.get("/demo", response_model=FleetResponse, response_model_exclude_none=True, tags=["Dashboard flotte"])
def demo():
    """
    **Apercu direct dans le navigateur.**

    Genere les 19 capteurs du dashboard et retourne la reponse complete sans POST.
    Ouvre http://localhost:8000/demo dans le navigateur pour voir le JSON.
    """
    req = FleetRequest(
        capteurs=[CapteurInput(id=c["id"], features=_demo_features(c)) for c in _DEMO_CAPTEURS],
        total_installes=19,
    )
    return predict_fleet(req)


@app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard flotte"])
def dashboard():
    """Dashboard HTML interactif — ouvre dans le navigateur."""
    data = demo()
    vg   = data.vue_globale
    ts   = data.timestamp
    now  = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    def badge_alerte(niveau: str) -> str:
        colors = {"CRITICAL": "#ef4444", "WARNING": "#f59e0b", "OK": "#10b981"}
        c = colors.get(niveau, "#6b7280")
        return f'<span style="background:{c};color:#fff;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700">{niveau}</span>'

    def badge_risque(niveau: str) -> str:
        colors = {"HAUTE": "#ef4444", "MODERE": "#f59e0b", "MODÉRÉ": "#f59e0b", "FAIBLE": "#10b981"}
        c = colors.get(niveau.upper().replace("É","E"), "#6b7280")
        return f'<span style="background:{c}22;color:{c};border:1px solid {c};padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700">{niveau}</span>'

    def health_bar(score: float) -> str:
        c = "#ef4444" if score < 40 else ("#f59e0b" if score < 70 else "#10b981")
        return f'''
        <div style="background:#1f2937;border-radius:4px;height:6px;margin:4px 0">
          <div style="background:{c};width:{score}%;height:6px;border-radius:4px;transition:width .3s"></div>
        </div>
        <div style="color:{c};font-size:12px;font-weight:700">{score}/100</div>'''

    def votes_html(votes: dict) -> str:
        parts = []
        for model, result in votes.items():
            short = {"isolation_forest":"IF","lof":"LOF","ocsvm":"OCSVM","ecod":"ECOD","hbos":"HBOS","copod":"COPOD"}.get(model, model.upper())
            c = "#ef4444" if result == "ANOMALIE" else "#10b981"
            parts.append(f'<span style="background:{c}22;color:{c};border:1px solid {c};padding:1px 5px;border-radius:3px;font-size:10px;font-weight:700">{short}</span>')
        return " ".join(parts)

    def sensor_card(c) -> str:
        anomalie_color = "#ef4444" if c.anomalie else "#10b981"
        anomalie_label = "ANOMALIE" if c.anomalie else "NORMAL"
        card_border    = "#ef444433" if c.anomalie else ("#f59e0b33" if c.niveau_alerte == "WARNING" else "#1f2937")
        m = c.mesures
        rul_html = ""
        if c.rul:
            rul_c = "#ef4444" if c.rul.alerte == "CRITIQUE" else ("#f59e0b" if c.rul.alerte == "ATTENTION" else "#10b981")
            rul_html = f'''
            <div style="margin-top:10px;padding:8px;background:#0a0e1a;border-radius:6px">
              <div style="color:#6b7280;font-size:10px;text-transform:uppercase;letter-spacing:1px">Remaining Useful Life</div>
              <div style="color:{rul_c};font-size:18px;font-weight:700;margin:2px 0">{c.rul.heures} h <span style="font-size:12px;color:#9ca3af">/ {c.rul.jours} j</span></div>
              <div style="color:#6b7280;font-size:11px">{c.rul.statut}</div>
              <div style="background:#1f2937;border-radius:3px;height:4px;margin-top:6px">
                <div style="background:{rul_c};width:{min(c.rul.degradation_pct,100)}%;height:4px;border-radius:3px"></div>
              </div>
              <div style="color:#6b7280;font-size:10px;margin-top:2px">Degradation {c.rul.degradation_pct}%</div>
            </div>'''

        return f'''
        <div style="background:#111827;border:1px solid {card_border};border-radius:10px;padding:16px;position:relative">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px">
            <div>
              <div style="color:#e5e7eb;font-size:14px;font-weight:700;font-family:monospace">{c.id}</div>
              <div style="color:#6b7280;font-size:11px;margin-top:2px">{c.vote_resume} &nbsp;|&nbsp; iter {c.iteration}</div>
            </div>
            <div style="display:flex;gap:6px;flex-direction:column;align-items:flex-end">
              <span style="background:{anomalie_color};color:#fff;padding:2px 10px;border-radius:4px;font-size:11px;font-weight:700">{anomalie_label}</span>
              {badge_risque(c.niveau_risque)}
            </div>
          </div>

          {health_bar(c.score_sante)}

          <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:10px">
            <div style="background:#0a0e1a;border-radius:6px;padding:6px">
              <div style="color:#6b7280;font-size:10px">TEMPERATURE</div>
              <div style="color:#60a5fa;font-size:16px;font-weight:700">{m.temperature_c} <span style="font-size:10px">°C</span></div>
            </div>
            <div style="background:#0a0e1a;border-radius:6px;padding:6px">
              <div style="color:#6b7280;font-size:10px">VIB TOTALE</div>
              <div style="color:#a78bfa;font-size:16px;font-weight:700">{m.vib_totale_mg} <span style="font-size:10px">mg</span></div>
            </div>
            <div style="background:#0a0e1a;border-radius:6px;padding:6px">
              <div style="color:#6b7280;font-size:10px">VIB Z RMS</div>
              <div style="color:#e5e7eb;font-size:14px;font-weight:600">{m.vib_z_rms_mg} mg</div>
            </div>
            <div style="background:#0a0e1a;border-radius:6px;padding:6px">
              <div style="color:#6b7280;font-size:10px">KURTOSIS Z</div>
              <div style="color:#e5e7eb;font-size:14px;font-weight:600">{m.kurtosis_z}</div>
            </div>
            <div style="background:#0a0e1a;border-radius:6px;padding:6px">
              <div style="color:#6b7280;font-size:10px">DEGRADATION</div>
              <div style="color:{"#ef4444" if m.degradation_pct>50 else "#f59e0b" if m.degradation_pct>20 else "#10b981"};font-size:14px;font-weight:600">{m.degradation_pct}%</div>
            </div>
            <div style="background:#0a0e1a;border-radius:6px;padding:6px">
              <div style="color:#6b7280;font-size:10px">TENDANCE VIB</div>
              <div style="color:#e5e7eb;font-size:13px;font-weight:600">{m.tendance_vib}</div>
            </div>
          </div>

          <div style="margin-top:10px">
            <div style="color:#6b7280;font-size:10px;margin-bottom:4px;text-transform:uppercase;letter-spacing:1px">Votes modeles</div>
            <div style="display:flex;flex-wrap:wrap;gap:4px">{votes_html(c.votes)}</div>
          </div>

          {rul_html}
        </div>'''

    # ── KPI cards ──────────────────────────────────────────────────────────────
    taux_c = "#ef4444" if vg.taux_anomalie_pct > 30 else ("#f59e0b" if vg.taux_anomalie_pct > 10 else "#10b981")
    sante_c = "#ef4444" if vg.sante_moyenne < 40 else ("#f59e0b" if vg.sante_moyenne < 70 else "#10b981")

    kpi_cards = f'''
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:24px">
      <div style="background:#111827;border:1px solid #1f2937;border-radius:10px;padding:16px">
        <div style="color:#6b7280;font-size:11px;text-transform:uppercase;letter-spacing:1px">Capteurs actifs</div>
        <div style="color:#e5e7eb;font-size:32px;font-weight:800;margin:4px 0">{vg.capteurs_actifs}</div>
        <div style="color:#6b7280;font-size:11px">sur {vg.total_installes} installes</div>
      </div>
      <div style="background:#111827;border:1px solid #ef444433;border-radius:10px;padding:16px">
        <div style="color:#6b7280;font-size:11px;text-transform:uppercase;letter-spacing:1px">En anomalie</div>
        <div style="color:#ef4444;font-size:32px;font-weight:800;margin:4px 0">{vg.en_anomalie}</div>
        <div style="color:{taux_c};font-size:11px">taux {vg.taux_anomalie_pct}%</div>
      </div>
      <div style="background:#111827;border:1px solid #1f2937;border-radius:10px;padding:16px">
        <div style="color:#6b7280;font-size:11px;text-transform:uppercase;letter-spacing:1px">Sante moyenne</div>
        <div style="color:{sante_c};font-size:32px;font-weight:800;margin:4px 0">{vg.sante_moyenne}</div>
        <div style="color:#6b7280;font-size:11px">score /100</div>
      </div>
      <div style="background:#111827;border:1px solid #1f2937;border-radius:10px;padding:16px">
        <div style="color:#6b7280;font-size:11px;text-transform:uppercase;letter-spacing:1px">RUL minimum</div>
        <div style="color:#f59e0b;font-size:28px;font-weight:800;margin:4px 0">{vg.rul_minimum.get("valeur_h","N/A")} <span style="font-size:14px">h</span></div>
        <div style="color:#6b7280;font-size:11px;font-family:monospace">{vg.rul_minimum.get("capteur_id","")}</div>
      </div>
      <div style="background:#111827;border:1px solid #1f2937;border-radius:10px;padding:16px">
        <div style="color:#6b7280;font-size:11px;text-transform:uppercase;letter-spacing:1px">Temp. max parc</div>
        <div style="color:#60a5fa;font-size:28px;font-weight:800;margin:4px 0">{vg.temp_max_parc.get("valeur_c")} <span style="font-size:14px">°C</span></div>
        <div style="color:#6b7280;font-size:11px;font-family:monospace">{vg.temp_max_parc.get("capteur_id")}</div>
      </div>
      <div style="background:#111827;border:1px solid #1f2937;border-radius:10px;padding:16px">
        <div style="color:#6b7280;font-size:11px;text-transform:uppercase;letter-spacing:1px">Vib. max parc</div>
        <div style="color:#a78bfa;font-size:28px;font-weight:800;margin:4px 0">{vg.vib_max_parc.get("valeur_mg")} <span style="font-size:14px">mg</span></div>
        <div style="color:#6b7280;font-size:11px;font-family:monospace">{vg.vib_max_parc.get("capteur_id")}</div>
      </div>
    </div>'''

    cards_html = "\n".join(sensor_card(c) for c in data.capteurs)

    html = f'''<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Maintenance Predictive — ISG BIZERTE</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0 }}
    body {{ background: #0a0e1a; color: #e5e7eb; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; min-height: 100vh }}
    header {{ background: #111827; border-bottom: 1px solid #1f2937; padding: 14px 24px; display: flex; justify-content: space-between; align-items: center }}
    .logo {{ display: flex; flex-direction: column }}
    .logo-title {{ color: #e5e7eb; font-size: 16px; font-weight: 800; letter-spacing: 1px; text-transform: uppercase }}
    .logo-title span {{ color: #3b82f6 }}
    .logo-sub {{ color: #6b7280; font-size: 11px; margin-top: 2px }}
    .header-right {{ display: flex; align-items: center; gap: 20px }}
    .models {{ color: #6b7280; font-size: 12px; font-family: monospace }}
    .models span {{ color: #60a5fa; font-weight: 700 }}
    .api-badge {{ background: #10b98122; color: #10b981; border: 1px solid #10b981; padding: 3px 10px; border-radius: 4px; font-size: 11px; font-weight: 700 }}
    .timestamp {{ color: #6b7280; font-size: 12px }}
    main {{ padding: 24px }}
    .section-title {{ color: #9ca3af; font-size: 12px; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 12px; display: flex; align-items: center; gap: 8px }}
    .section-title::after {{ content: ""; flex: 1; height: 1px; background: #1f2937 }}
    .sensors-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 14px }}
    .refresh {{ position: fixed; bottom: 24px; right: 24px; background: #3b82f6; color: #fff; border: none; padding: 10px 20px; border-radius: 8px; cursor: pointer; font-size: 13px; font-weight: 700; box-shadow: 0 4px 12px #3b82f640 }}
    .refresh:hover {{ background: #2563eb }}
  </style>
</head>
<body>
<header>
  <div class="logo">
    <div class="logo-title"><span>Maintenance Predictive</span> en temps reel</div>
    <div class="logo-sub">Systeme IoT &nbsp;·&nbsp; ISG BIZERTE &nbsp;·&nbsp; v3.1.0</div>
  </div>
  <div class="header-right">
    <div class="models">Modeles <span>{data.modeles}</span></div>
    <div class="api-badge">API 3.1.0 — OK</div>
    <div class="timestamp">{now}</div>
  </div>
</header>

<main>
  <div class="section-title">Vue globale du parc</div>
  {kpi_cards}

  <div class="section-title">Etat des capteurs ({vg.capteurs_actifs} actifs)</div>
  <div class="sensors-grid">
    {cards_html}
  </div>
</main>

<button class="refresh" onclick="location.reload()">Rafraichir</button>
</body>
</html>'''

    return HTMLResponse(content=html)


@app.get("/health", tags=["Statut"])
def health_check():
    return {
        "status": "ok",
        "version": "3.1.0",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "modeles": {
            "ensemble": MODELS_NAMES,
            "taille": 6 if ENSEMBLE_6 else 4,
            "rul_disponible": RUL_OK,
        },
        "features": {
            "anomalie": len(FEATURES_ORDER),
            "rul": len(features_rul) if RUL_OK else None,
        },
    }


@app.post("/fleet", response_model=FleetResponse, response_model_exclude_none=True, tags=["Dashboard flotte"])
def predict_fleet(req: FleetRequest):
    """
    **Endpoint principal du dashboard.**

    Analyse toute la flotte de capteurs en une seule requête et retourne :
    - `vue_globale` → barre du haut (capteurs actifs, anomalies, santé moyenne, RUL min, temp/vib max)
    - `capteurs[]` → liste complète avec score, votes, RUL, mesures et historique graphiques

    ### Entrée
    ```json
    {
      "capteurs": [
        {
          "id": "8f7f2f7e",
          "features": { ...31 features... },
          "historique": [ ...lectures précédentes (optionnel, pour graphiques)... ]
        }
      ],
      "total_installes": 20
    }
    ```

    ### Sortie (résumé)
    ```json
    {
      "timestamp": "2026-06-11T18:04:51",
      "modeles": "IF·LOF·OCSVM·ECOD",
      "vue_globale": {
        "capteurs_actifs": 19,
        "en_anomalie": 6,
        "sante_moyenne": 53.0,
        "rul_minimum": { "valeur_h": 594.1, "capteur_id": "8f7f2f7e" },
        "temp_max_parc": { "valeur_c": 18.6, "capteur_id": "07da47b8" },
        "vib_max_parc":  { "valeur_mg": 5.0, "capteur_id": "a6a46be1" }
      },
      "capteurs": [
        {
          "id": "8f7f2f7e",
          "anomalie": true,
          "score_anomalie": 0.500,
          "score_sante": 50.0,
          "niveau_alerte": "WARNING",
          "niveau_risque": "MODÉRÉ",
          "votes": { "isolation_forest": "ANOMALIE", "lof": "normal", ... },
          "vote_resume": "2/4 votes · 50%",
          "rul": { "heures": 594.1, "jours": 24.8, "alerte": "ATTENTION", ... },
          "mesures": {
            "temperature_c": 17.8,
            "vib_z_rms_mg": 3.0,
            "vib_x_rms_mg": 2.0,
            "vib_y_rms_mg": 2.0,
            "vib_totale_mg": 6.0,
            "kurtosis_z": 0.25,
            "degradation_pct": 9.8,
            "tendance_vib": "→ Stable"
          },
          "historique_scores": [0.25, 0.50, ...],
          "historique_temperatures": [17.5, 17.8, ...],
          "historique_vib_z": [2.8, 3.0, ...],
          "historique_vib_x": [2.0, 2.0, ...],
          "historique_vib_y": [1.9, 2.0, ...]
        }
      ]
    }
    ```
    """
    if not req.capteurs:
        raise HTTPException(status_code=422, detail="Liste de capteurs vide")

    def _safe_predict(args):
        i, capteur = args
        return _predict_one(capteur, iteration=i + 1)

    with ThreadPoolExecutor(max_workers=min(len(req.capteurs), 8)) as pool:
        try:
            results: List[CapteurResult] = list(pool.map(_safe_predict, enumerate(req.capteurs)))
        except Exception as e:
            raise HTTPException(status_code=422, detail=str(e))

    n_total   = len(results)
    n_anomaly = sum(1 for r in results if r.anomalie)
    sante_moy = round(sum(r.score_sante for r in results) / n_total, 1)

    rul_list = [(r.id, r.rul.heures) for r in results if r.rul]
    if rul_list:
        rul_id, rul_h = min(rul_list, key=lambda x: x[1])
        rul_min = {"valeur_h": round(rul_h, 1), "capteur_id": rul_id}
    else:
        rul_min = {"valeur_h": None, "capteur_id": None}

    temp_max_r = max(results, key=lambda r: r.mesures.temperature_c)
    vib_max_r  = max(results, key=lambda r: r.mesures.vib_totale_mg)

    vue = VueGlobale(
        capteurs_actifs=n_total,
        total_installes=req.total_installes or n_total,
        en_anomalie=n_anomaly,
        taux_anomalie_pct=round(n_anomaly / n_total * 100, 1),
        sante_moyenne=sante_moy,
        rul_minimum=rul_min,
        temp_max_parc={"valeur_c": temp_max_r.mesures.temperature_c, "capteur_id": temp_max_r.id},
        vib_max_parc={"valeur_mg": vib_max_r.mesures.vib_totale_mg,  "capteur_id": vib_max_r.id},
    )

    return FleetResponse(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        modeles=MODELS_NAMES,
        version_api="3.1.0",
        vue_globale=vue,
        capteurs=results,
    )


@app.post("/predict", response_model=PredictResponse, response_model_exclude_none=True, tags=["Capteur unique"])
def predict(data: SensorFeatures):
    """Détecte une anomalie sur un capteur unique (sans historique ni RUL)."""
    try:
        X = _preprocess(data)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Erreur prétraitement : {e}")

    votes_raw = _run_models(X)
    n_anomaly = sum(1 for v in votes_raw.values() if v == -1)
    n_total   = len(votes_raw)
    score     = round(n_anomaly / n_total, 3)
    health    = round(max(0.0, min(100.0, (1.0 - score) * 100)), 1)
    anomaly   = n_anomaly >= (n_total // 2 + 1)

    votes_readable = {k: ("ANOMALIE" if v == -1 else "normal") for k, v in votes_raw.items()}
    conf  = "HIGH"  if score > 0.75 else ("MEDIUM" if score > 0.45 else "LOW")
    alert = "CRITICAL" if score > 0.75 else ("WARNING" if score > 0.45 else "OK")

    return PredictResponse(
        anomaly=anomaly, anomaly_score=score, health_score=health,
        votes=votes_readable, confidence=conf, alert_level=alert,
    )


@app.post("/predict-rul", response_model=RULResponse, response_model_exclude_none=True, tags=["Capteur unique"])
def predict_rul(req: RULRequest):
    """Estime le Remaining Useful Life (RUL) en heures."""
    if not RUL_OK:
        raise HTTPException(status_code=503, detail="Modèle RUL non disponible")
    try:
        row = [req.features.get(f, 0.0) for f in features_rul]
        X_s = scaler_rul.transform(np.array(row, dtype=float).reshape(1, -1))
        rul_hours = max(0.0, float(model_rul.predict(X_s)[0]))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Erreur RUL : {e}")

    rul_days = round(rul_hours / 24, 1)
    if rul_hours < 24:
        status = "CRITIQUE — remplacement urgent"
    elif rul_hours < 168:
        status = "ATTENTION — maintenance dans la semaine"
    elif rul_hours < 720:
        status = "STABLE — surveillance recommandée"
    else:
        status = "BON — aucune action requise"

    return RULResponse(rul_hours=round(rul_hours, 1), rul_days=rul_days, confidence="MEDIUM", status=status)


@app.post("/predict-batch", tags=["Capteur unique"])
def predict_batch(data: List[SensorFeatures]):
    """Prédit sur plusieurs capteurs, retourne une liste de résultats."""
    results = []
    for i, item in enumerate(data):
        try:
            r = predict(item)
            results.append({"index": i, "success": True, **r.dict()})
        except Exception as e:
            results.append({"index": i, "success": False, "error": str(e)})
    return {"count": len(results), "results": results}


# ── Lancement ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
