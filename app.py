import calendar

import folium
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from pathlib import Path
from streamlit_folium import st_folium

st.set_page_config(page_title="Nagpur Electrical Hazard Predictor", layout="wide")

# ---------- Load model and locality data ----------
BASE_DIR = Path(__file__).parent
artifact = joblib.load(BASE_DIR / "models" / "hazard_model.pkl")
model, features = artifact["model"], artifact["features"]
areas = pd.read_csv(BASE_DIR / "data" / "areas.csv")

NAGPUR_LAT, NAGPUR_LON = 21.1458, 79.0882

# ---------- Standard pole fields and column auto-detection ----------
FIELDS = {
    "pole_id": "Pole ID / number (required)",
    "area_name": "Locality / area",
    "lat": "Latitude",
    "lon": "Longitude",
    "pole_type": "Pole type / material",
    "line_type": "Line type / voltage",
    "height_m": "Height (m)",
    "install_date": "Installation date (or year)",
    "last_maintenance_date": "Last maintenance date",
    "next_due_date": "Next maintenance due",
    "inspection_result": "Inspection result / condition",
    "earthing_status": "Earthing status",
    "defect_reports": "Defect / complaint count",
}

KEYWORDS = {
    "pole_id": ["pole id", "pole no", "pole number", "pole code", "pole tag", "asset id"],
    "area_name": ["locality", "area name", "area", "ward", "zone", "location"],
    "lat": ["latitude", "lat"],
    "lon": ["longitude", "lon", "lng", "long"],
    "pole_type": ["pole type", "pole material", "material"],
    "line_type": ["line type", "voltage", "line"],
    "height_m": ["height m", "pole height", "height"],
    "install_date": ["install date", "installation date", "date of installation", "installed on",
                     "commissioned", "erection date", "year of installation", "install year"],
    "last_maintenance_date": ["last maintenance date", "last maintenance", "last service",
                              "last serviced", "maintenance date", "last inspection date"],
    "next_due_date": ["next due date", "next maintenance", "next due", "due date"],
    "inspection_result": ["inspection result", "inspection", "condition"],
    "earthing_status": ["earthing status", "earthing", "earth"],
    "defect_reports": ["defect reports", "defect", "complaints", "faults"],
}
EXACT_ONLY = {"lat", "lon"}   # short names would wrongly match words like "installation"

LABELS = {
    "pole_id": "Pole ID", "area_name": "Locality", "nearest_area": "Nearest known locality",
    "lat": "Latitude", "lon": "Longitude", "pole_type": "Pole type", "line_type": "Line type",
    "height_m": "Height (m)", "install_date": "Installed on", "age_years": "Age (years)",
    "last_maintenance_date": "Last maintenance", "days_since_maintenance": "Days since maintenance",
    "next_due_date": "Next due", "days_to_due": "Days to due",
    "maintenance_status": "Maintenance status", "inspection_result": "Inspection",
    "earthing_status": "Earthing", "defect_reports": "Defect reports",
    "area_risk": "Locality risk (%)", "area_risk_level": "Locality risk level",
}

TEMPLATE = pd.DataFrame({
    "pole_id": ["SAMPLE-001", "SAMPLE-002"],
    "locality": ["Dharampeth", "Sitabuldi"],
    "latitude": [21.1350, 21.1480],
    "longitude": [79.0650, 79.0830],
    "pole_type": ["PSC", "Steel tubular"],
    "line_type": ["LT 415 V", "HT 11 kV"],
    "height_m": [9, 11],
    "install_date": ["2015-06-01", "2009-03-15"],
    "last_maintenance_date": ["2025-11-20", "2024-02-10"],
    "inspection_result": ["Good", "Fair"],
    "earthing_status": ["OK", "Needs check"],
    "defect_reports": [0, 2],
})


def clamp(value, low, high):
    return float(max(low, min(high, float(value))))


@st.cache_data(ttl=1800)
def get_live_weather():
    """Fetch today's weather for Nagpur from Open-Meteo (free, no API key)."""
    r = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": NAGPUR_LAT, "longitude": NAGPUR_LON,
        "current": "temperature_2m,relative_humidity_2m",
        "daily": "precipitation_sum,wind_speed_10m_max,temperature_2m_max",
        "forecast_days": 1, "timezone": "Asia/Kolkata",
    }, timeout=15).json()
    cur, day = r["current"], r["daily"]
    return {
        "rain": day["precipitation_sum"][0],
        "wind": day["wind_speed_10m_max"][0],
        "temp_max": day["temperature_2m_max"][0],
        "temp": cur["temperature_2m"],
        "humidity": cur["relative_humidity_2m"],
    }


def guess_column(columns, keys, partial=True):
    """Pick the column whose name best matches one of the keywords."""
    norm = {c: str(c).strip().lower().replace("_", " ") for c in columns}
    for key in keys:
        for c, n in norm.items():
            if n == key:
                return c
    if partial:
        for key in keys:
            for c, n in norm.items():
                if key in n:
                    return c
    return None


def read_uploaded(file):
    """Read a CSV or Excel upload into a DataFrame."""
    if file.name.lower().endswith(".csv"):
        try:
            raw = pd.read_csv(file)
        except UnicodeDecodeError:
            file.seek(0)
            raw = pd.read_csv(file, encoding="latin-1")
    else:
        xl = pd.ExcelFile(file)
        sheet = xl.sheet_names[0]
        if len(xl.sheet_names) > 1:
            sheet = st.selectbox("Excel sheet", xl.sheet_names)
        raw = xl.parse(sheet)
    raw = raw.dropna(how="all").dropna(axis=1, how="all")
    raw.columns = [str(c).strip() for c in raw.columns]
    return raw


def to_date(series):
    """Parse dates; plain years like 2015 become 1 Jan 2015."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    num = pd.to_numeric(series, errors="coerce")
    is_year = num.between(1950, 2100) & (num == num.round())
    out = pd.to_datetime(series.where(~is_year), errors="coerce", dayfirst=True)
    if is_year.any():
        out[is_year] = pd.to_datetime(num[is_year].astype("Int64").astype(str),
                                      format="%Y", errors="coerce")
    return out


def build_poles(raw, mapping, interval_days, include_extras=True):
    """Turn a raw table into a standard pole table with derived fields."""
    notes = []
    std = pd.DataFrame(index=raw.index)
    for field, col in mapping.items():
        if col is not None:
            std[field] = raw[col]
    if "pole_id" not in std.columns:
        return None, ["Select the column that holds the pole ID."]

    for c in ["lat", "lon", "height_m", "defect_reports"]:
        if c in std.columns:
            std[c] = pd.to_numeric(std[c], errors="coerce")
    for c in ["install_date", "last_maintenance_date", "next_due_date"]:
        if c in std.columns:
            parsed = to_date(std[c])
            bad = int(parsed.isna().sum() - std[c].isna().sum())
            if bad > 0:
                notes.append(f"{bad} value(s) in '{FIELDS[c]}' could not be read as dates.")
            std[c] = parsed

    std["pole_id"] = std["pole_id"].astype(str).str.strip()
    before = len(std)
    std = std[~std["pole_id"].str.lower().isin(["", "nan", "none"])]
    if len(std) < before:
        notes.append(f"{before - len(std)} row(s) without a pole ID were skipped.")
    dups = int(std["pole_id"].duplicated().sum())
    if dups:
        std = std.drop_duplicates("pole_id")
        notes.append(f"{dups} duplicate pole ID row(s) were removed (first one kept).")

    if {"lat", "lon"} <= set(std.columns):
        outside = (~(std.lat.between(21.0, 21.3) & std.lon.between(78.9, 79.3))
                   & std.lat.notna() & std.lon.notna())
        if outside.any():
            notes.append(f"{int(outside.sum())} pole(s) have coordinates outside the Nagpur area.")

    today = pd.Timestamp.today().normalize()
    if "install_date" in std.columns:
        std["age_years"] = ((today - std["install_date"]).dt.days / 365.25).round(1)
    if "last_maintenance_date" in std.columns:
        std["days_since_maintenance"] = (today - std["last_maintenance_date"]).dt.days
        if "next_due_date" not in std.columns:
            std["next_due_date"] = std["last_maintenance_date"] + pd.Timedelta(days=interval_days)
            notes.append(f"No 'next due' column found, so it was estimated as last maintenance "
                         f"+ {interval_days} days.")
    if "next_due_date" in std.columns:
        std["days_to_due"] = (std["next_due_date"] - today).dt.days
        status = pd.Series(
            np.select([std["days_to_due"] < 0, std["days_to_due"] <= 30],
                      ["OVERDUE", "DUE SOON"], default="OK"),
            index=std.index,
        )
        status[std["days_to_due"].isna()] = "UNKNOWN"
        std["maintenance_status"] = status

    if include_extras:
        used = {c for c in mapping.values() if c is not None}
        for c in raw.columns:
            if c in used:
                continue
            name = c if c not in std.columns else f"{c} (file)"
            std[name] = raw.loc[std.index, c]
    return std, notes


def attach_risk(poles, risk_df):
    """Attach the current AI risk of the nearest known locality to each pole."""
    poles = poles.copy()
    poles["nearest_area"] = None
    poles["area_risk"] = np.nan
    poles["area_risk_level"] = None
    if {"lat", "lon"} <= set(poles.columns):
        valid = poles["lat"].notna() & poles["lon"].notna()
        if valid.any():
            la = poles.loc[valid, "lat"].to_numpy(float)[:, None]
            lo = poles.loc[valid, "lon"].to_numpy(float)[:, None]
            d = (la - risk_df.lat.to_numpy()[None, :]) ** 2 \
                + ((lo - risk_df.lon.to_numpy()[None, :]) * 0.934) ** 2
            idx = d.argmin(axis=1)
            poles.loc[valid, "nearest_area"] = risk_df.area_name.to_numpy()[idx]
            poles.loc[valid, "area_risk"] = risk_df.risk.to_numpy()[idx]
            poles.loc[valid, "area_risk_level"] = risk_df.level.to_numpy()[idx]
    if "area_name" in poles.columns:
        lookup = risk_df.set_index("area_name")
        missing = poles["area_risk"].isna() & poles["area_name"].isin(lookup.index)
        poles.loc[missing, "area_risk"] = poles.loc[missing, "area_name"].map(lookup["risk"])
        poles.loc[missing, "area_risk_level"] = poles.loc[missing, "area_name"].map(lookup["level"])
    return poles


def fmt(v):
    """Format one value for display; returns None for empty values."""
    if pd.isna(v):
        return None
    if isinstance(v, pd.Timestamp):
        return v.strftime("%d %b %Y")
    if isinstance(v, (float, np.floating)):
        return f"{v:.2f}".rstrip("0").rstrip(".")
    return str(v)


def display_table(frame):
    """Rename columns and format dates for the registry table."""
    out = frame.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%d-%b-%Y").fillna("")
    if "area_risk" in out.columns:
        out["area_risk"] = out["area_risk"].round(1)
    return out.rename(columns=LABELS)


# ---------- Header ----------
st.title("⚡ Nagpur Electrical Hazard Predictor")
st.caption("AI-based prediction of public electrical hazard risk across Nagpur localities, "
           "with a pole-level asset registry that works on any uploaded CSV/Excel file.")
st.warning("Academic project. Weather is real (Open-Meteo); the built-in demo pole records, "
           "infrastructure data and hazard labels are synthetic. This is not an official "
           "government system or a safety alert service.")

# ---------- Sidebar: weather inputs ----------
st.sidebar.header("Weather Conditions")
values = {"rain": 20.0, "humidity": 80.0, "wind": 15.0, "temp": 28.0, "temp_max": 33.0}

use_live = st.sidebar.checkbox("Use live Nagpur weather", value=False)
if use_live:
    try:
        values = get_live_weather()
        st.sidebar.success("Live weather loaded")
    except Exception:
        st.sidebar.warning("Could not load live weather. Using manual values.")

rain = st.sidebar.slider("Rainfall (mm/day)", 0.0, 150.0, clamp(values["rain"], 0, 150), step=0.5)
humidity = st.sidebar.slider("Humidity (%)", 10.0, 100.0, clamp(values["humidity"], 10, 100), step=1.0)
wind = st.sidebar.slider("Max wind speed (km/h)", 0.0, 80.0, clamp(values["wind"], 0, 80), step=1.0)
temp_mean = st.sidebar.slider("Average temperature (°C)", 5.0, 45.0, clamp(values["temp"], 5, 45), step=0.5)
temp_max = st.sidebar.slider("Maximum temperature (°C)", 10.0, 50.0, clamp(values["temp_max"], 10, 50), step=0.5)
month = st.sidebar.selectbox(
    "Month",
    list(range(1, 13)),
    index=pd.Timestamp.today().month - 1,
    format_func=lambda m: calendar.month_name[m],
)

# ---------- Predict risk for every locality ----------
df = areas.copy()
df["rainfall"], df["humidity"], df["wind_speed"] = rain, humidity, wind
df["temp_mean"], df["temp_max"], df["month"] = temp_mean, temp_max, month

X = pd.get_dummies(df, columns=["area_type"], dtype=int).reindex(columns=features, fill_value=0)
df["risk"] = model.predict_proba(X)[:, 1] * 100

high_cut, medium_cut = df.risk.quantile(0.9), df.risk.quantile(0.7)


def risk_level(r):
    if r >= high_cut:
        return "HIGH"
    if r >= medium_cut:
        return "MEDIUM"
    return "LOW"


df["level"] = df.risk.apply(risk_level)

# ---------- Tabs ----------
tab_map, tab_data, tab_detail, tab_about = st.tabs(
    ["Risk Map", "Pole Data (Upload)", "Pole Details", "Data & Sources"]
)

poles = None

# ===== Tab: Pole data (runs first so other tabs can use the result) =====
with tab_data:
    st.subheader("Pole Data")
    source = st.radio("Data source", ["Demo data", "Upload my file (CSV / Excel)"], horizontal=True)

    raw, mapping, include_extras = None, None, True
    interval_days = 365

    if source == "Demo data":
        demo_path = BASE_DIR / "data" / "poles.csv"
        if demo_path.exists():
            raw = pd.read_csv(demo_path)
            mapping = {f: guess_column(raw.columns, KEYWORDS[f], f not in EXACT_ONLY) for f in FIELDS}
            st.caption("Showing built-in synthetic demo poles.")
        else:
            st.info("Demo file data/poles.csv was not found. Choose 'Upload my file' instead.")
    else:
        st.info("Your file is processed in memory and is not saved by this app. Because this is a "
                "public demo hosted online, do not upload confidential or personal data here. "
                "For sensitive data, run the app on your own computer.")
        st.download_button("Download CSV template", TEMPLATE.to_csv(index=False).encode("utf-8"),
                           "pole_data_template.csv", "text/csv")
        interval_days = int(st.number_input(
            "Assumed maintenance interval (days), used only if the file has no 'next due' column",
            min_value=30, max_value=1825, value=365, step=30))
        up = st.file_uploader("Upload pole data", type=["csv", "xlsx"])
        if up is not None:
            try:
                raw = read_uploaded(up)
            except Exception as e:
                st.error(f"Could not read this file: {e}")
            if raw is not None:
                st.caption(f"{len(raw)} rows, {len(raw.columns)} columns detected.")
                with st.expander("Match your columns (auto-detected, change if needed)"):
                    options = ["(not in my file)"] + list(raw.columns)
                    mapping = {}
                    ui = st.columns(3)
                    for i, (field, label) in enumerate(FIELDS.items()):
                        guess = guess_column(raw.columns, KEYWORDS[field], field not in EXACT_ONLY)
                        idx = options.index(guess) if guess in options else 0
                        with ui[i % 3]:
                            choice = st.selectbox(label, options, index=idx, key=f"map_{field}_{up.name}")
                        mapping[field] = None if choice == options[0] else choice
                    st.dataframe(raw.head(10), hide_index=True)

    if raw is not None and mapping is not None:
        built, notes = build_poles(raw, mapping, interval_days, include_extras)
        for n in notes:
            st.warning(n)
        if built is not None and len(built) > 0:
            poles = attach_risk(built, df)
            if "area_risk" in poles.columns and poles["area_risk"].notna().any():
                st.caption("Locality risk is the current AI risk of the nearest known Nagpur "
                           "locality (approximate).")

    if poles is not None:
        c1, c2, c3 = st.columns(3)
        loc_col = None
        for cand in ["area_name", "nearest_area"]:
            if cand in poles.columns and poles[cand].notna().any():
                loc_col = cand
                break
        view = poles
        if loc_col:
            chosen = c1.multiselect("Locality", sorted(poles[loc_col].dropna().unique()))
            if chosen:
                view = view[view[loc_col].isin(chosen)]
        if "maintenance_status" in poles.columns:
            status_sel = c2.multiselect("Maintenance status", ["OVERDUE", "DUE SOON", "OK", "UNKNOWN"])
            if status_sel:
                view = view[view["maintenance_status"].isin(status_sel)]
        query = c3.text_input("Search pole ID")
        if query:
            view = view[view["pole_id"].str.contains(query, case=False, regex=False)]

        mc = st.columns(4)
        mc[0].metric("Poles shown", len(view))
        if "maintenance_status" in view.columns:
            mc[1].metric("Maintenance overdue", int((view["maintenance_status"] == "OVERDUE").sum()))
            mc[2].metric("Due within 30 days", int((view["maintenance_status"] == "DUE SOON").sum()))
        if "age_years" in view.columns:
            mc[3].metric("Average age (years)",
                         round(float(view["age_years"].mean()), 1) if view["age_years"].notna().any() else "n/a")

        shown = display_table(view)
        st.dataframe(shown, hide_index=True)
        st.download_button("Download filtered list (CSV)", shown.to_csv(index=False).encode("utf-8"),
                           "pole_registry.csv", "text/csv")
    elif source != "Demo data" and raw is None:
        st.info("Upload a CSV or Excel file to see pole details instantly.")

# ===== Tab: Risk map =====
with tab_map:
    show_poles = False
    if poles is not None and {"lat", "lon"} <= set(poles.columns):
        show_poles = st.checkbox("Also show poles on the map (first 500)", value=False)

    colors = {"HIGH": "red", "MEDIUM": "orange", "LOW": "green"}
    m = folium.Map(location=[NAGPUR_LAT, NAGPUR_LON], zoom_start=12)
    for _, r in df.iterrows():
        folium.CircleMarker(
            [r.lat, r.lon], radius=10, color=colors[r.level], fill=True, fill_opacity=0.8,
            tooltip=f"{r.area_name} ({r.area_type}): {r.risk:.1f}% risk",
        ).add_to(m)

    if show_poles:
        status_colors = {"OVERDUE": "red", "DUE SOON": "orange", "OK": "green", "UNKNOWN": "gray"}
        pts = poles.dropna(subset=["lat", "lon"]).head(500)
        for _, p in pts.iterrows():
            status = p["maintenance_status"] if "maintenance_status" in pts.columns else "UNKNOWN"
            folium.CircleMarker(
                [p["lat"], p["lon"]], radius=4, color=status_colors.get(status, "gray"),
                fill=True, fill_opacity=0.9, tooltip=f"{p['pole_id']} ({status})",
            ).add_to(m)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Nagpur Risk Map")
        st_folium(m, width=800, height=560, key="risk_map", returned_objects=[])
    with col2:
        st.subheader("Top 10 Risky Localities")
        top = df.sort_values("risk", ascending=False)[["area_name", "area_type", "risk", "level"]].head(10)
        st.dataframe(top, hide_index=True)
        st.metric("High-risk localities", int((df.level == "HIGH").sum()))
        st.caption("Risk levels are relative: top 10% of localities are marked HIGH. "
                   "Big circles = localities, small dots = poles (colored by maintenance status).")

# ===== Tab: Single pole details =====
with tab_detail:
    st.subheader("Pole Details")
    if poles is None or poles.empty:
        st.info("Choose demo data or upload a file in the 'Pole Data (Upload)' tab first.")
    else:
        pole_id = st.selectbox("Select a pole (type to search)", sorted(poles["pole_id"]))
        p = poles[poles["pole_id"] == pole_id].iloc[0]

        k = st.columns(3)
        k[0].metric("Maintenance status", p["maintenance_status"] if "maintenance_status" in p.index else "n/a")
        k[1].metric("Age (years)", fmt(p["age_years"]) if "age_years" in p.index and fmt(p["age_years"]) else "n/a")
        risk_text = f"{p['area_risk']:.1f}% ({p['area_risk_level']})" if pd.notna(p["area_risk"]) else "n/a"
        k[2].metric("Locality risk now", risk_text)

        left, right = st.columns(2)
        with left:
            rows = []
            for col in poles.columns:
                if col in ("area_risk", "area_risk_level"):
                    continue
                value = fmt(p[col])
                if value is not None:
                    rows.append((LABELS.get(col, col), value))
            st.table(pd.DataFrame(rows, columns=["Field", "Value"]).set_index("Field"))
        with right:
            if {"lat", "lon"} <= set(p.index) and pd.notna(p["lat"]) and pd.notna(p["lon"]):
                pm = folium.Map(location=[p["lat"], p["lon"]], zoom_start=17)
                folium.Marker([p["lat"], p["lon"]], tooltip=p["pole_id"]).add_to(pm)
                st_folium(pm, width=500, height=420, key="pole_map", returned_objects=[])
            else:
                st.info("This pole has no coordinates, so no map is shown.")

# ===== Tab: Data and sources =====
with tab_about:
    st.subheader("Data & Sources")
    st.markdown("""
| Data | Status | Source |
|---|---|---|
| Daily weather (rain, humidity, wind, temperature) | **Real** | Open-Meteo / ERA5 |
| Locality names and coordinates | **Real** | OpenStreetMap |
| Built-in demo pole records | **Synthetic** | Generated by `generate_poles.py` |
| Infrastructure attributes and hazard labels used to train the model | **Synthetic** | Rule-based simulation |
| Uploaded pole files | **As provided by the user** | Not verified by this app |

**How uploads work:** choose *Upload my file*, drop a CSV or Excel file, check the auto-detected
column matching, and the registry, maintenance status and pole details appear instantly.
Only the pole ID column is required; every other column is optional.

**Disclaimer:** this is an academic prototype. It is not affiliated with, endorsed by, or
connected to MSEDCL, Nagpur Municipal Corporation, or any government body.
""")