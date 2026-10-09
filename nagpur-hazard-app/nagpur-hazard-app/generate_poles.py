"""Generate SYNTHETIC demo pole records for the Nagpur hazard app.

Reads data/areas.csv and writes data/poles.csv. The result is demo data only.
Replace data/poles.csv (or use the upload tab in the app) when real data is available.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
areas_path = BASE_DIR / "data" / "areas.csv"
if not areas_path.exists():
    sys.exit("data/areas.csv not found. Copy it from the Colab output into the 'data' folder first.")

rng = np.random.default_rng(7)
areas = pd.read_csv(areas_path)
today = pd.Timestamp.today().normalize()

POLES_PER_AREA = 25
MAINTENANCE_INTERVAL_DAYS = 365   # assumed inspection cycle (demo policy)

rows = []
for _, a in areas.iterrows():
    for i in range(1, POLES_PER_AREA + 1):
        age_days = int(np.clip(a.pole_age * 365 + rng.normal(0, 600), 180, 40 * 365))
        since_days = int(np.clip(a.days_since_maintenance + rng.normal(0, 90), 0, age_days - 30))
        age_years = age_days / 365.25

        line_type = str(rng.choice(["LT 415 V", "HT 11 kV"], p=[0.65, 0.35]))
        height = int(rng.choice([8, 9]) if line_type == "LT 415 V" else rng.choice([11, 13]))

        score = 0.5 * age_years / 40 + 0.5 * since_days / 730 + rng.normal(0, 0.1)
        inspection = "Poor" if score > 0.6 else "Fair" if score > 0.3 else "Good"
        earthing = "Needs check" if (inspection != "Good" and rng.random() < 0.5) else "OK"

        last_maint = today - pd.Timedelta(days=since_days)
        rows.append({
            "pole_id": f"DEMO-{int(a.area_id):02d}-{i:03d}",
            "area_id": int(a.area_id),
            "area_name": a.area_name,
            "area_type": a.area_type,
            "lat": round(a.lat + rng.normal(0, 0.003), 6),
            "lon": round(a.lon + rng.normal(0, 0.003), 6),
            "pole_type": str(rng.choice(["PSC (pre-stressed concrete)", "RCC", "Steel tubular", "Rail pole"],
                                        p=[0.45, 0.2, 0.25, 0.1])),
            "line_type": line_type,
            "height_m": height,
            "install_date": (today - pd.Timedelta(days=age_days)).date().isoformat(),
            "last_maintenance_date": last_maint.date().isoformat(),
            "next_due_date": (last_maint + pd.Timedelta(days=MAINTENANCE_INTERVAL_DAYS)).date().isoformat(),
            "inspection_result": inspection,
            "earthing_status": earthing,
            "transformer_attached": bool(rng.random() < 0.15),
            "defect_reports": int(rng.poisson(0.2 + 0.05 * age_years)),
        })

poles = pd.DataFrame(rows)
poles.to_csv(BASE_DIR / "data" / "poles.csv", index=False)
print("Poles generated:", len(poles))
