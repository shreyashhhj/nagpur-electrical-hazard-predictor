import streamlit as st
import pandas as pd
import joblib
import folium
import requests
from pathlib import Path
from streamlit_folium import st_folium

st.set_page_config(page_title="Nagpur Electrical Hazard Predictor", layout="wide")

# ---------- Load model and data ----------
BASE_DIR = Path(__file__).parent
artifact = joblib.load(BASE_DIR / "models" / "hazard_model.pkl")
model, features = artifact["model"], artifact["features"]
areas = pd.read_csv(BASE_DIR / "data" / "areas.csv")

NAGPUR_LAT, NAGPUR_LON = 21.1458, 79.0882


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


st.title("⚡ Nagpur Electrical Hazard Predictor")
st.caption("AI-based prediction of public electrical hazard risk across Nagpur localities, "
           "using weather and infrastructure conditions.")

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
month = st.sidebar.selectbox("Month", list(range(1, 13)), index=pd.Timestamp.today().month - 1)

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

# ---------- Map ----------
colors = {"HIGH": "red", "MEDIUM": "orange", "LOW": "green"}
m = folium.Map(location=[NAGPUR_LAT, NAGPUR_LON], zoom_start=12)
for _, r in df.iterrows():
    folium.CircleMarker(
        [r.lat, r.lon], radius=10, color=colors[r.level], fill=True, fill_opacity=0.8,
        tooltip=f"{r.area_name} ({r.area_type}): {r.risk:.1f}% risk",
    ).add_to(m)

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Nagpur Risk Map")
    st_folium(m, width=800, height=560)
with col2:
    st.subheader("Top 10 Risky Localities")
    top = df.sort_values("risk", ascending=False)[["area_name", "area_type", "risk", "level"]].head(10)
    st.dataframe(top, hide_index=True)
    st.metric("High-risk localities", int((df.level == "HIGH").sum()))
    st.caption("Risk levels are relative: top 10% of localities are marked HIGH.")