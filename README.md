# API Maintenance Prédictive — v3.1.0

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)
![Render](https://img.shields.io/badge/Deployed-Render-46E3B7?logo=render)
![License](https://img.shields.io/badge/License-MIT-yellow)

> Système IoT de détection d'anomalies et estimation de durée de vie résiduelle (RUL) pour capteurs industriels IFM — ISG BIZERTE

---

## Démo en ligne

| Endpoint | URL |
|----------|-----|
| Dashboard visuel | [/dashboard](https://api-predictivee.onrender.com/dashboard) |
| Documentation Swagger | [/docs](https://api-predictivee.onrender.com/docs) |
| Statut API | [/health](https://api-predictivee.onrender.com/health) |
| Démo JSON (19 capteurs) | [/demo](https://api-predictivee.onrender.com/demo) |

---

## Architecture

```
Capteurs IFM (IoT)
        │
        ▼
┌───────────────────────────────────────────┐
│           API FastAPI v3.1.0              │
│                                           │
│  ┌─────────────┐   ┌──────────────────┐  │
│  │  Détection  │   │   Estimation RUL │  │
│  │  Anomalies  │   │  (GradientBoost) │  │
│  │             │   │                  │  │
│  │ IF · LOF    │   │  46 features     │  │
│  │ OCSVM · ECOD│   │  R² = 0.56       │  │
│  │ HBOS · COPOD│   │                  │  │
│  └─────────────┘   └──────────────────┘  │
│         │                   │            │
│         ▼                   ▼            │
│    Vote majoritaire    RUL en heures     │
│    (3/6 → anomalie)   + statut alerte   │
└───────────────────────────────────────────┘
        │
        ▼
   Dashboard HTML / JSON / Postman
```

---

## Endpoints

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/health` | Statut API + modèles chargés |
| `GET` | `/dashboard` | Dashboard HTML interactif |
| `GET` | `/demo` | Simulation 19 capteurs réels |
| `POST` | `/predict` | Détection anomalie capteur unique |
| `POST` | `/predict-rul` | Estimation RUL en heures |
| `POST` | `/predict-batch` | Lot de capteurs |
| `POST` | `/fleet` | Dashboard flotte complet |
| `GET` | `/docs` | Swagger UI interactif |

---

## Modèles ML

### Détection d'anomalies (ensemble de 6 modèles)

| Modèle | Type | Rôle |
|--------|------|------|
| Isolation Forest | Arbre d'isolation | Détecte les points isolés |
| LOF | Densité locale | Compare avec les voisins |
| OCSVM | SVM one-class | Frontière de décision |
| ECOD | Distribution empirique | Copule outlier detection |
| HBOS | Histogrammes | Détection par feature |
| COPOD | Copule statistique | Dépendances entre features |

**Vote majoritaire** : 3 modèles sur 6 → `anomaly: true`

### Estimation RUL

| Propriété | Valeur |
|-----------|--------|
| Algorithme | GradientBoostingRegressor |
| Features | 46 (spectrales + temporelles) |
| MAE test | ±317h |
| R² test | 0.56 |
| Entraîné sur | 12 000 échantillons |

---

## Dataset

- **Source** : 20 capteurs IFM réels — ISG BIZERTE
- **Période** : novembre 2025 → mars 2026
- **Taille** : 73 917 mesures
- **Anomalies** : 18 480 détectées (6.6% contamination)
- **Features** : 31 features par capteur (vibration XYZ, température, courant, entropie, FFT)

---

## Installation locale

### Prérequis
- Python 3.11+
- pip

### Lancement

```bash
git clone https://github.com/yassinebalti2002/api-predictivee-.git
cd api-predictivee-
pip install -r requirements.txt
python api.py
```

L'API est accessible sur `http://localhost:8000`

### Avec Docker

```bash
docker-compose up -d
```

---

## Exemple d'utilisation

### Capteur normal

```bash
curl -X POST https://api-predictivee.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "temp_mean": 18.0, "temp_std": 0.3, "temp_trend": 0.0, "temp_cur": 18.3,
    "vib_z_mean": 2.8, "vib_z_std": 0.4, "vib_z_rms_w": 3.0,
    "vib_z_kurt": 0.25, "vib_z_crest": 1.2, "vib_z_cur": 3.0,
    "vib_x_mean": 1.8, "vib_x_std": 0.2, "vib_x_rms_w": 2.0, "vib_x_kurt": 0.1,
    "vib_y_mean": 1.7, "vib_y_std": 0.2, "vib_y_rms_w": 2.0, "vib_y_kurt": 0.1,
    "vib_total": 6.0, "health_score": 92.0,
    "acc_p2p": 6.0, "acc_z2p": 3.0, "acc_crest": 1.2, "acc_rms": 3.0,
    "current_mean": 2.1,
    "delta_vib": 0.1, "delta_temp": 0.05, "vib_entropy": 0.8, "fft_ratio": 0.1,
    "vib_asym_xy": 0.05, "vib_asym_xz": 0.02
  }'
```

**Réponse :**
```json
{
  "anomaly": false,
  "anomaly_score": 0.333,
  "health_score": 66.7,
  "confidence": "LOW",
  "alert_level": "OK"
}
```

### Capteur en anomalie (temp=90°C, vib=140mg)

```json
{
  "anomaly": true,
  "anomaly_score": 0.5,
  "health_score": 50.0,
  "votes": {
    "isolation_forest": "ANOMALIE",
    "lof": "ANOMALIE",
    "ocsvm": "ANOMALIE",
    "ecod": "normal",
    "hbos": "normal",
    "copod": "normal"
  },
  "confidence": "MEDIUM",
  "alert_level": "WARNING"
}
```

---

## Structure du projet

```
api-predictivee-/
├── api.py                  # Application FastAPI principale
├── requirements.txt        # Dépendances Python
├── Dockerfile              # Image Docker
├── docker-compose.yml      # Lancement local Docker
├── render.yaml             # Configuration déploiement Render
├── test_api.py             # Tests manuels (serveur déjà lancé)
├── run_tests.py            # Tests automatiques (lance le serveur)
└── models/
    ├── model_if_v3.pkl     # Isolation Forest
    ├── model_lof_v3.pkl    # Local Outlier Factor
    ├── model_ocsvm_v3.pkl  # One-Class SVM
    ├── model_ecod_v3.pkl   # ECOD
    ├── model_hbos_v3.pkl   # HBOS
    ├── model_copod_v3.pkl  # COPOD
    ├── model_rul_v1.pkl    # GradientBoosting RUL
    ├── scaler_v3.pkl       # Normalisation features anomalie
    ├── scaler_rul_v1.pkl   # Normalisation features RUL
    ├── pca_v3.pkl          # Réduction dimensionnelle
    ├── features_v3.pkl     # Noms features anomalie
    ├── features_rul_v1.pkl # Noms features RUL
    ├── threshold_v3.pkl    # Seuils optimaux
    └── metrics_v3.csv      # Métriques d'évaluation
```

---

## Niveaux d'alerte

| Score anomalie | Niveau | Action recommandée |
|----------------|--------|--------------------|
| < 0.45 (0-2/6 votes) | `OK` | Aucune action |
| 0.45 – 0.75 (3-4/6 votes) | `WARNING` | Inspection préventive |
| > 0.75 (5-6/6 votes) | `CRITICAL` | Intervention urgente |

### Niveaux RUL

| RUL | Statut | Action |
|-----|--------|--------|
| > 720h | BON | Aucune action requise |
| 168h – 720h | STABLE | Surveillance recommandée |
| 24h – 168h | ATTENTION | Maintenance dans la semaine |
| < 24h | CRITIQUE | Remplacement urgent |

---

## Technologies

- **Backend** : FastAPI + Uvicorn
- **ML** : scikit-learn 1.5.2, PyOD, NumPy, Joblib
- **Conteneurisation** : Docker + Docker Compose
- **Déploiement** : Render.com (cloud)
- **Versioning** : GitHub

---

## Auteur

**Yassine Balti** — ISG BIZERTE  
Projet de maintenance prédictive IoT — 2025/2026
