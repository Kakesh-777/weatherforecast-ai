import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WeatherCast AI",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0b0f1a;
    color: #e8eaf0;
}

.main { background-color: #0b0f1a; }
.block-container { padding: 2rem 3rem; max-width: 1100px; }

h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
    color: #f0f4ff;
}

.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3.4rem;
    line-height: 1.1;
    color: #f0f4ff;
    margin-bottom: 0.3rem;
}

.hero-sub {
    font-size: 1.05rem;
    color: #7b83a0;
    font-weight: 300;
    margin-bottom: 2.5rem;
}

.card {
    background: #131929;
    border: 1px solid #1e2740;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.2rem;
}

.metric-card {
    background: linear-gradient(135deg, #131929 0%, #0f1e35 100%);
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}

.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    color: #5b9cf6;
    line-height: 1;
}

.metric-label {
    font-size: 0.78rem;
    color: #5d6580;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.4rem;
}

.forecast-day {
    background: #131929;
    border: 1px solid #1e2740;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    transition: border-color 0.2s;
}

.forecast-day:hover { border-color: #5b9cf6; }

.forecast-temp {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: #5b9cf6;
}

.forecast-date {
    font-size: 0.75rem;
    color: #5d6580;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.status-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.05em;
}

.badge-live { background: #0d2e1a; color: #4ade80; border: 1px solid #166534; }
.badge-warn { background: #2e1a0d; color: #fb923c; border: 1px solid #7c2d12; }

.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 2rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    font-weight: 500;
    cursor: pointer;
    width: 100%;
    transition: opacity 0.2s;
}

.stButton > button:hover { opacity: 0.85; }

.stTextInput > div > div > input {
    background: #131929 !important;
    border: 1px solid #1e2740 !important;
    border-radius: 10px !important;
    color: #e8eaf0 !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 0.6rem 1rem !important;
}

.stSelectbox > div > div {
    background: #131929 !important;
    border: 1px solid #1e2740 !important;
    border-radius: 10px !important;
    color: #e8eaf0 !important;
}

.stSlider > div > div > div { background: #2563eb !important; }

div[data-testid="stMetric"] {
    background: #131929;
    border: 1px solid #1e2740;
    border-radius: 12px;
    padding: 1rem 1.2rem;
}

div[data-testid="stMetricValue"] { color: #5b9cf6 !important; font-size: 1.8rem !important; }
div[data-testid="stMetricLabel"] { color: #7b83a0 !important; }

.section-title {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #5d6580;
    margin-bottom: 1rem;
    border-bottom: 1px solid #1e2740;
    padding-bottom: 0.5rem;
}

hr { border-color: #1e2740 !important; }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ──────────────────────────────────────────────────────────

def get_coordinates(city_name):
    """Get lat/lon from Open-Meteo geocoding API."""
    try:
        params = {"name": city_name, "count": 1, "language": "en", "format": "json"}
        r = requests.get("https://geocoding-api.open-meteo.com/v1/search", params=params, timeout=8)
        data = r.json()
        if "results" not in data or len(data["results"]) == 0:
            return None, None
        city = data["results"][0]
        return city["latitude"], city["longitude"]
    except Exception:
        return None, None


def get_current_weather(lat, lon):
    """Fetch latest conditions from Open-Meteo forecast API."""
    try:
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
            "timezone": "auto"
        }
        r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=8)
        return r.json().get("current", {})
    except Exception:
        return {}


def get_historical_data(lat, lon, days=30):
    """Fetch recent historical daily data from Open-Meteo archive."""
    try:
        end = datetime.today() - timedelta(days=2)
        start = end - timedelta(days=days)
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,relative_humidity_2m_mean",
            "timezone": "auto",
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d")
        }
        r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=10)
        data = r.json()
        if "daily" not in data:
            return None
        df = pd.DataFrame({
            "date": pd.to_datetime(data["daily"]["time"]),
            "TMAX": data["daily"]["temperature_2m_max"],
            "TMIN": data["daily"]["temperature_2m_min"],
            "precipitation": data["daily"]["precipitation_sum"],
            "relative_humidity": data["daily"]["relative_humidity_2m_mean"],
        })
        df["temperature"] = df["TMAX"]
        df["PRCP"] = df["precipitation"]
        df["SNOW"] = 0.0
        return df.dropna()
    except Exception:
        return None


def weather_code_to_description(code):
    descriptions = {
        0: ("Clear Sky", "☀️"), 1: ("Mainly Clear", "🌤️"), 2: ("Partly Cloudy", "⛅"),
        3: ("Overcast", "☁️"), 45: ("Foggy", "🌫️"), 48: ("Icy Fog", "🌫️"),
        51: ("Light Drizzle", "🌦️"), 53: ("Drizzle", "🌦️"), 55: ("Heavy Drizzle", "🌧️"),
        61: ("Light Rain", "🌧️"), 63: ("Rain", "🌧️"), 65: ("Heavy Rain", "🌧️"),
        71: ("Light Snow", "🌨️"), 73: ("Snow", "❄️"), 75: ("Heavy Snow", "❄️"),
        80: ("Rain Showers", "🌦️"), 81: ("Heavy Showers", "⛈️"), 82: ("Violent Showers", "⛈️"),
        95: ("Thunderstorm", "⛈️"), 96: ("Thunderstorm w/ Hail", "⛈️"),
    }
    return descriptions.get(code, ("Unknown", "🌡️"))


def run_lstm_forecast(df, model_path, scaler_path, lookback=7, horizon=7):
    """Run LSTM model if available, else return None."""
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)

        features = ["temperature", "relative_humidity", "precipitation", "PRCP", "TMAX", "TMIN"]
        data = df[features].values
        scaled = scaler.transform(data)

        if len(scaled) < lookback:
            return None

        seq = scaled[-lookback:]
        preds_scaled = []
        current_seq = seq.copy()
        tmax_idx = features.index("TMAX")

        for _ in range(horizon):
            inp = current_seq.reshape(1, lookback, len(features))
            pred = model.predict(inp, verbose=0)[0][0]
            preds_scaled.append(pred)
            new_row = current_seq[-1].copy()
            new_row[tmax_idx] = pred
            current_seq = np.vstack([current_seq[1:], new_row])

        # Inverse scale only TMAX column
        dummy = np.zeros((len(preds_scaled), len(features)))
        dummy[:, tmax_idx] = preds_scaled
        inv = scaler.inverse_transform(dummy)
        return inv[:, tmax_idx]

    except Exception:
        return None


def simple_forecast(df, horizon=7):
    """Fallback: simple moving average forecast."""
    recent_tmax = df["TMAX"].tail(14).values
    avg = np.mean(recent_tmax)
    trend = (recent_tmax[-1] - recent_tmax[0]) / len(recent_tmax)
    return [avg + trend * i for i in range(1, horizon + 1)]


# ── App layout ────────────────────────────────────────────────────────────────

# Hero header
st.markdown('<div class="hero-title">WeatherCast AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">LSTM-powered temperature forecasting · Real station data · Any city worldwide</div>', unsafe_allow_html=True)

# Search bar
col_input, col_btn, col_days = st.columns([3, 1, 1])
with col_input:
    city = st.text_input("", placeholder="Enter a city name  e.g. Starkville, New York, London...", label_visibility="collapsed")
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    search = st.button("Forecast →")
with col_days:
    horizon = st.selectbox("", [3, 5, 7, 10, 14], index=2, label_visibility="collapsed")

st.markdown("---")

# ── Main forecast logic ───────────────────────────────────────────────────────
if search and city:
    with st.spinner("Fetching weather data..."):
        lat, lon = get_coordinates(city)

    if lat is None:
        st.error(f"Could not find coordinates for '{city}'. Please check the city name.")
        st.stop()

    with st.spinner("Loading historical data..."):
        hist_df = get_historical_data(lat, lon, days=60)
        current = get_current_weather(lat, lon)

    if hist_df is None or len(hist_df) < 7:
        st.error("Not enough historical data available for this location.")
        st.stop()

    # ── Current conditions row ────────────────────────────────────────────────
    st.markdown(f'<div class="section-title">Current Conditions — {city.title()}</div>', unsafe_allow_html=True)

    weather_desc, weather_icon = weather_code_to_description(current.get("weather_code", 0))
    temp_now = current.get("temperature_2m", hist_df["TMAX"].iloc[-1])
    humidity_now = current.get("relative_humidity_2m", hist_df["relative_humidity"].iloc[-1])
    precip_now = current.get("precipitation", 0)
    wind_now = current.get("wind_speed_10m", 0)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric(f"{weather_icon} Condition", weather_desc)
    with c2:
        st.metric("🌡️ Temperature", f"{temp_now:.1f}°C")
    with c3:
        st.metric("💧 Humidity", f"{humidity_now:.0f}%")
    with c4:
        st.metric("🌧️ Precipitation", f"{precip_now:.1f} mm")
    with c5:
        st.metric("💨 Wind Speed", f"{wind_now:.1f} km/h")

    st.markdown("---")

    # ── Run forecast ─────────────────────────────────────────────────────────
    model_path = "lstm_weather_model.h5"
    scaler_path = "scaler.save"
    model_available = os.path.exists(model_path) and os.path.exists(scaler_path)

    if model_available:
        forecast_temps = run_lstm_forecast(hist_df, model_path, scaler_path, lookback=7, horizon=horizon)
        forecast_source = "LSTM Model"
        if forecast_temps is None:
            forecast_temps = simple_forecast(hist_df, horizon=horizon)
            forecast_source = "Statistical Baseline"
    else:
        forecast_temps = simple_forecast(hist_df, horizon=horizon)
        forecast_source = "Statistical Baseline (No model found)"

    forecast_dates = [datetime.today() + timedelta(days=i+1) for i in range(horizon)]

    # ── Forecast cards ────────────────────────────────────────────────────────
    badge_class = "badge-live" if "LSTM" in forecast_source else "badge-warn"
    st.markdown(
        f'<div class="section-title">{horizon}-Day Forecast &nbsp;&nbsp;'
        f'<span class="status-badge {badge_class}">{forecast_source}</span></div>',
        unsafe_allow_html=True
    )

    cols = st.columns(horizon)
    for i, (col, temp, date) in enumerate(zip(cols, forecast_temps, forecast_dates)):
        with col:
            day_name = date.strftime("%a")
            date_str = date.strftime("%b %d")
            delta = temp - hist_df["TMAX"].iloc[-1]
            arrow = "↑" if delta > 0 else "↓"
            color = "#ef4444" if delta > 0 else "#60a5fa"
            st.markdown(f"""
                <div class="forecast-day">
                    <div class="forecast-date">{day_name}</div>
                    <div class="forecast-date">{date_str}</div>
                    <div class="forecast-temp">{temp:.1f}°</div>
                    <div style="font-size:0.78rem;color:{color};margin-top:0.3rem">{arrow} {abs(delta):.1f}°</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Historical + forecast chart ───────────────────────────────────────────
    st.markdown('<div class="section-title">Temperature History & Forecast</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#0b0f1a")
    ax.set_facecolor("#0b0f1a")

    hist_recent = hist_df.tail(30)
    ax.fill_between(hist_recent["date"], hist_recent["TMIN"], hist_recent["TMAX"],
                    alpha=0.15, color="#5b9cf6", label="Daily Range")
    ax.plot(hist_recent["date"], hist_recent["TMAX"],
            color="#5b9cf6", linewidth=2, label="Historical TMAX")

    ax.axvline(x=datetime.today(), color="#1e2740", linestyle="--", linewidth=1)

    fc_dates_dt = [pd.Timestamp(d) for d in forecast_dates]
    ax.plot(fc_dates_dt, forecast_temps,
            color="#f59e0b", linewidth=2.5, linestyle="--",
            marker="o", markersize=5, label="Forecast")

    ax.set_xlabel("Date", color="#5d6580", fontsize=9)
    ax.set_ylabel("Temperature (°C)", color="#5d6580", fontsize=9)
    ax.tick_params(colors="#5d6580", labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.xticks(rotation=30)

    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2740")

    ax.legend(facecolor="#131929", edgecolor="#1e2740",
              labelcolor="#e8eaf0", fontsize=8)
    ax.grid(axis="y", color="#1e2740", linewidth=0.5, alpha=0.5)

    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ── Historical stats ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Recent 30-Day Statistics</div>', unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Avg Max Temp", f"{hist_df['TMAX'].tail(30).mean():.1f}°C")
    with s2:
        st.metric("Avg Min Temp", f"{hist_df['TMIN'].tail(30).mean():.1f}°C")
    with s3:
        st.metric("Total Precipitation", f"{hist_df['precipitation'].tail(30).sum():.1f} mm")
    with s4:
        st.metric("Avg Humidity", f"{hist_df['relative_humidity'].tail(30).mean():.0f}%")

    st.markdown("---")

    # ── Raw data expander ─────────────────────────────────────────────────────
    with st.expander("📊 View Raw Forecast Data"):
        forecast_df = pd.DataFrame({
            "Date": [d.strftime("%Y-%m-%d") for d in forecast_dates],
            "Day": [d.strftime("%A") for d in forecast_dates],
            "Predicted TMAX (°C)": [f"{t:.2f}" for t in forecast_temps]
        })
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)

else:
    # Landing state
    st.markdown("""
    <div class="card" style="text-align:center; padding: 3rem 2rem;">
        <div style="font-size:3rem;margin-bottom:1rem">🌤️</div>
        <div style="font-family:'DM Serif Display',serif;font-size:1.5rem;color:#f0f4ff;margin-bottom:0.5rem">
            Enter a city to get started
        </div>
        <div style="color:#5d6580;font-size:0.95rem">
            Powered by NOAA station data · Open-Meteo Archive · LSTM Neural Network
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">How it works</div>', unsafe_allow_html=True)
    h1, h2, h3 = st.columns(3)
    with h1:
        st.markdown("""
        <div class="card">
            <div style="font-size:1.8rem">📡</div>
            <div style="font-weight:500;margin:0.5rem 0;color:#f0f4ff">Data Collection</div>
            <div style="color:#5d6580;font-size:0.88rem">Pulls real historical weather data from Open-Meteo archive and NOAA climate stations for any city worldwide.</div>
        </div>
        """, unsafe_allow_html=True)
    with h2:
        st.markdown("""
        <div class="card">
            <div style="font-size:1.8rem">🧠</div>
            <div style="font-weight:500;margin:0.5rem 0;color:#f0f4ff">LSTM Forecasting</div>
            <div style="color:#5d6580;font-size:0.88rem">A trained Long Short-Term Memory neural network learns seasonal and daily temperature patterns to predict future conditions.</div>
        </div>
        """, unsafe_allow_html=True)
    with h3:
        st.markdown("""
        <div class="card">
            <div style="font-size:1.8rem">📈</div>
            <div style="font-weight:500;margin:0.5rem 0;color:#f0f4ff">Visual Insights</div>
            <div style="color:#5d6580;font-size:0.88rem">Interactive charts show 30-day history alongside the forecast so you can see trends and patterns at a glance.</div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align:center;color:#2a3050;font-size:0.78rem;margin-top:3rem;padding-top:1rem;border-top:1px solid #1e2740">
    WeatherCast AI · Built with TensorFlow, Open-Meteo & NOAA CDO · Data updates daily
</div>
""", unsafe_allow_html=True)