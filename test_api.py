"""
Test de l'API Maintenance Prédictive v3.1.0
Usage: python test_api.py
"""

import requests
import json

BASE_URL = "http://localhost:8000"

# ── Features d'un capteur normal ──────────────────────────────────────────────
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

# ── Features d'un capteur en anomalie ─────────────────────────────────────────
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

# ── Requête flotte (comme le dashboard) ───────────────────────────────────────
FLEET_REQUEST = {
    "total_installes": 20,
    "capteurs": [
        {
            "id": "8f7f2f7e",
            "features": CAPTEUR_ANOMALIE,
            # historique optionnel — sans historique = pas de graphiques dans la réponse
        },
        {
            "id": "0ff416d2",
            "features": CAPTEUR_NORMAL,
        },
        {
            "id": "b2acdf45",
            "features": {**CAPTEUR_NORMAL, "temp_cur": 18.3, "vib_total": 5.5},
        },
        {
            "id": "6e0c1740",
            "features": {**CAPTEUR_NORMAL, "temp_cur": 18.5, "vib_total": 6.1},
        },
        {
            "id": "07da47b8",
            "features": {**CAPTEUR_NORMAL, "temp_cur": 18.6, "vib_total": 4.8},
        },
    ]
}


def print_json(label: str, data: dict):
    print(f"\n{'═'*60}")
    print(f"  {label}")
    print(f"{'═'*60}")
    print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    # 1. Statut API
    r = requests.get(f"{BASE_URL}/health")
    print_json("GET /health", r.json())

    # 2. Endpoint principal — dashboard flotte
    r = requests.post(f"{BASE_URL}/fleet", json=FLEET_REQUEST, timeout=30)
    r.raise_for_status()
    data = r.json()

    print_json("POST /fleet → vue_globale (barre du haut)", data["vue_globale"])

    print(f"\n{'─'*60}")
    print("  POST /fleet → capteurs (liste gauche du dashboard)")
    print(f"{'─'*60}")
    for c in data["capteurs"]:
        anomalie_tag = "🔴 ANOMALIE" if c["anomalie"] else "🟢 normal"
        print(f"  [{c['id']}]  {anomalie_tag}  santé={c['score_sante']}  "
              f"risque={c['niveau_risque']}  {c['vote_resume']}")
        if c["rul"]:
            print(f"            RUL={c['rul']['heures']}h ({c['rul']['jours']}j)  "
                  f"alerte={c['rul']['alerte']}")
        m = c["mesures"]
        print(f"            T={m['temperature_c']}°C  VibZ={m['vib_z_rms_mg']}mg  "
              f"Vib3D={m['vib_totale_mg']}mg  Kurt={m['kurtosis_z']}  "
              f"Dég={m['degradation_pct']}%  {m['tendance_vib']}")

    # 3. Capteur unique
    r = requests.post(f"{BASE_URL}/predict", json=CAPTEUR_ANOMALIE, timeout=10)
    print_json("POST /predict (capteur unique anomalie)", r.json())
