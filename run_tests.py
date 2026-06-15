"""
Test complet de l'API — GET /health + POST /fleet + POST /predict
"""
import subprocess
import time
import requests
import json
import sys

proc = subprocess.Popen(["python", "api.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
print(f"Serveur demarré (PID {proc.pid})")
time.sleep(8)

CAPTEUR_NORMAL = {
    "temp_mean": 18.0, "temp_std": 0.3, "temp_trend": 0.0, "temp_cur": 18.3,
    "vib_z_mean": 2.8, "vib_z_std": 0.4, "vib_z_rms_w": 3.0,
    "vib_z_kurt": 0.25, "vib_z_crest": 1.2, "vib_z_cur": 3.0,
    "vib_x_mean": 1.8, "vib_x_std": 0.2, "vib_x_rms_w": 2.0, "vib_x_kurt": 0.1,
    "vib_y_mean": 1.7, "vib_y_std": 0.2, "vib_y_rms_w": 2.0, "vib_y_kurt": 0.1,
    "vib_total": 6.0, "health_score": 92.0,
    "acc_p2p": 6.0, "acc_z2p": 3.0, "acc_crest": 1.2, "acc_rms": 3.0,
    "current_mean": 2.1,
    "delta_vib": 0.1, "delta_temp": 0.05, "vib_entropy": 0.8, "fft_ratio": 0.1,
    "vib_asym_xy": 0.05, "vib_asym_xz": 0.02,
}

CAPTEUR_ANOMALIE = {
    "temp_mean": 85.0, "temp_std": 8.5, "temp_trend": 2.5, "temp_cur": 90.0,
    "vib_z_mean": 95.0, "vib_z_std": 18.0, "vib_z_rms_w": 98.0,
    "vib_z_kurt": 4.5, "vib_z_crest": 4.2, "vib_z_cur": 100.0,
    "vib_x_mean": 72.0, "vib_x_std": 15.0, "vib_x_rms_w": 74.0, "vib_x_kurt": 3.8,
    "vib_y_mean": 68.0, "vib_y_std": 14.0, "vib_y_rms_w": 70.0, "vib_y_kurt": 3.5,
    "vib_total": 140.0, "health_score": 15.0,
    "acc_p2p": 200.0, "acc_z2p": 100.0, "acc_crest": 4.2, "acc_rms": 98.0,
    "current_mean": 8.5,
    "delta_vib": 12.0, "delta_temp": 5.0, "vib_entropy": 3.8, "fft_ratio": 2.1,
    "vib_asym_xy": 1.8, "vib_asym_xz": 1.5,
}

def sep(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

# ── 1. GET /health ────────────────────────────────────────────
sep("GET /health")
try:
    r = requests.get("http://localhost:8000/health", timeout=5)
    print(f"Status: {r.status_code}")
    print(json.dumps(r.json(), indent=2, ensure_ascii=False))
except Exception as e:
    print(f"ERREUR: {e}")
    proc.terminate()
    sys.exit(1)

# ── 2. POST /predict (capteur anomalie) ──────────────────────
sep("POST /predict — capteur ANOMALIE")
try:
    r = requests.post("http://localhost:8000/predict", json=CAPTEUR_ANOMALIE, timeout=15)
    print(f"Status: {r.status_code}")
    print(json.dumps(r.json(), indent=2, ensure_ascii=False))
except Exception as e:
    print(f"ERREUR: {e}")

# ── 3. POST /predict (capteur normal) ────────────────────────
sep("POST /predict — capteur NORMAL")
try:
    r = requests.post("http://localhost:8000/predict", json=CAPTEUR_NORMAL, timeout=15)
    print(f"Status: {r.status_code}")
    print(json.dumps(r.json(), indent=2, ensure_ascii=False))
except Exception as e:
    print(f"ERREUR: {e}")

# ── 4. POST /fleet ────────────────────────────────────────────
sep("POST /fleet — flotte 2 capteurs")
fleet_payload = {
    "total_installes": 5,
    "capteurs": [
        {"id": "anomalie-01", "features": CAPTEUR_ANOMALIE},
        {"id": "normal-01",   "features": CAPTEUR_NORMAL},
    ]
}
try:
    r = requests.post("http://localhost:8000/fleet", json=fleet_payload, timeout=30)
    print(f"Status: {r.status_code}")
    data = r.json()
    print("vue_globale:", json.dumps(data.get("vue_globale"), indent=2, ensure_ascii=False))
    print("\ncapteurs:")
    for c in data.get("capteurs", []):
        tag = "ANOMALIE" if c["anomalie"] else "normal"
        rul = c.get("rul")
        rul_str = f"  RUL={rul['heures']}h alerte={rul['alerte']}" if rul else ""
        print(f"  [{c['id']}] {tag}  sante={c['score_sante']}  risque={c['niveau_risque']}  {c['vote_resume']}{rul_str}")
except Exception as e:
    print(f"ERREUR: {e}")

proc.terminate()
print("\nServeur arrêté.")
