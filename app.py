"""
app.py — Stock Price Trend Predictor (Streamlit App)
Run with: streamlit run app.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ─── Import our feature engineering from data_prep.py ─────────────────────────
from data_prep import add_features

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Trend Predictor",
    page_icon="📈",
    layout="wide"
)

# ─── Popular Stocks Dropdown ───────────────────────────────────────────────────
POPULAR_STOCKS = {
    "🇮🇳 Indian Stocks (NSE)": {
        "TCS":        "TCS.NS",
        "Infosys":    "INFY.NS",
        "Reliance":   "RELIANCE.NS",
        "HDFC Bank":  "HDFCBANK.NS",
        "Wipro":      "WIPRO.NS",
        "ITC":        "ITC.NS",
        "Bajaj Finance": "BAJFINANCE.NS",
    },
    "🌍 Global Stocks": {
        "Apple":      "AAPL",
        "Microsoft":  "MSFT",
        "Google":     "GOOGL",
        "Tesla":      "TSLA",
        "Amazon":     "AMZN",
        "Meta":       "META",
        "NVIDIA":     "NVDA",
    }
}


# ─── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "models/best_model.pkl"
    scaler_path = "models/scaler.pkl"
    cols_path   = "models/feature_cols.pkl"

    if not all(os.path.exists(p) for p in [model_path, scaler_path, cols_path]):
        return None, None, None

    model        = joblib.load(model_path)
    scaler       = joblib.load(scaler_path)
    feature_cols = joblib.load(cols_path)
    return model, scaler, feature_cols


@st.cache_data(ttl=300)  # Cache data for 5 minutes
def fetch_data(ticker: str, days: int = 365):
    end   = datetime.today()
    start = end - timedelta(days=days)
    df    = yf.download(ticker, start=start, end=end, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    # Quick pick
    st.subheader("Quick Pick")
    category = st.selectbox("Category", list(POPULAR_STOCKS.keys()))
    stock_map = POPULAR_STOCKS[category]
    chosen_name = st.selectbox("Stock", list(stock_map.keys()))
    quick_ticker = stock_map[chosen_name]

    st.divider()

    # Manual input
    st.subheader("Or type a ticker")
    manual_ticker = st.text_input("Ticker symbol", placeholder="e.g. AAPL, TCS.NS")

    # Final ticker
    ticker = manual_ticker.strip().upper() if manual_ticker.strip() else quick_ticker

    st.divider()
    days = st.slider("Historical data (days)", 90, 730, 365)
    st.caption(f"Showing data for: **{ticker}**")


# ─── Main App ─────────────────────────────────────────────────────────────────
st.title("📈 Stock Price Trend Predictor")
st.caption("Predicts whether a stock will go UP or DOWN the next trading day using Machine Learning.")

model, scaler, feature_cols = load_model()

if model is None:
    st.warning(
        "⚠️ No trained model found. Please run `train_model.py` first to train and save the model.",
        icon="⚠️"
    )
    st.code("python data_prep.py\npython train_model.py", language="bash")
    st.stop()

# ─── Fetch Data ───────────────────────────────────────────────────────────────
with st.spinner(f"Fetching data for {ticker}..."):
    try:
        raw_df = fetch_data(ticker, days)
        if raw_df.empty:
            st.error(f"No data found for '{ticker}'. Please check the ticker symbol.")
            st.stop()
        df = add_features(raw_df)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

# ─── Prediction ───────────────────────────────────────────────────────────────
latest_features = df[feature_cols].iloc[[-1]]
latest_scaled   = scaler.transform(latest_features)
prediction      = model.predict(latest_scaled)[0]
confidence      = model.predict_proba(latest_scaled)[0][prediction] * 100

latest_price    = float(df['Close'].iloc[-1])
prev_price      = float(df['Close'].iloc[-2])
price_change    = latest_price - prev_price
price_change_pct = (price_change / prev_price) * 100

# ─── Metrics Row ──────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Current Price", f"${latest_price:.2f}", f"{price_change_pct:+.2f}%")

with col2:
    rsi_val = float(df['RSI'].iloc[-1])
    rsi_status = "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral")
    st.metric("RSI", f"{rsi_val:.1f}", rsi_status)

with col3:
    ma20 = float(df['MA20'].iloc[-1])
    above_below = "Above MA20 ✅" if latest_price > ma20 else "Below MA20 ⚠️"
    st.metric("MA20", f"${ma20:.2f}", above_below)

with col4:
    vol_change = float(df['Volume_Change'].iloc[-1])
    st.metric("Volume Change", f"{vol_change:+.1f}%")

st.divider()

# ─── Prediction Card ──────────────────────────────────────────────────────────
col_pred, col_chart = st.columns([1, 2])

with col_pred:
    if prediction == 1:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #1a7a4a, #2ecc71);
                        border-radius: 16px; padding: 30px; text-align: center; color: white;">
                <div style="font-size: 60px;">📈</div>
                <div style="font-size: 28px; font-weight: bold; margin: 10px 0;">TREND: UP</div>
                <div style="font-size: 16px; opacity: 0.9;">Confidence: {confidence:.1f}%</div>
                <div style="font-size: 13px; margin-top: 12px; opacity: 0.7;">
                    Model predicts price will rise<br>on the next trading day
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #7a1a1a, #e74c3c);
                        border-radius: 16px; padding: 30px; text-align: center; color: white;">
                <div style="font-size: 60px;">📉</div>
                <div style="font-size: 28px; font-weight: bold; margin: 10px 0;">TREND: DOWN</div>
                <div style="font-size: 16px; opacity: 0.9;">Confidence: {confidence:.1f}%</div>
                <div style="font-size: 13px; margin-top: 12px; opacity: 0.7;">
                    Model predicts price will fall<br>on the next trading day
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.caption("⚠️ This is an ML prediction, not financial advice.")

# ─── Price Chart with Moving Averages ─────────────────────────────────────────
with col_chart:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index, y=df['Close'],
        name='Close Price', line=dict(color='#4A90D9', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MA5'],
        name='MA5', line=dict(color='orange', width=1.2, dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MA20'],
        name='MA20', line=dict(color='red', width=1.2, dash='dash')
    ))

    fig.update_layout(
        title=f"{ticker} — Price & Moving Averages",
        xaxis_title="Date", yaxis_title="Price (USD)",
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=350, margin=dict(l=0, r=0, t=40, b=0),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

# ─── RSI Chart ────────────────────────────────────────────────────────────────
st.subheader("RSI — Relative Strength Index")
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')))
fig_rsi.add_hline(y=70, line_dash='dash', line_color='red',   annotation_text='Overbought (70)')
fig_rsi.add_hline(y=30, line_dash='dash', line_color='green', annotation_text='Oversold (30)')
fig_rsi.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=0), yaxis=dict(range=[0, 100]))
st.plotly_chart(fig_rsi, use_container_width=True)

# ─── Raw Data ─────────────────────────────────────────────────────────────────
with st.expander("📋 View Raw Feature Data (last 10 rows)"):
    display_cols = ['Close', 'Return', 'MA5', 'MA20', 'RSI', 'Volume_Change', 'Volatility', 'Target']
    st.dataframe(df[display_cols].tail(10).style.format("{:.2f}"), use_container_width=True)
