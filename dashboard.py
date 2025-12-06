import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import plotly.graph_objects as go
import os



###############################
# CONFIG
###############################

st.set_page_config(
    page_title="Stock Predictor Dashboard",
    layout="wide",
)

tickers = ["TSLA", "AMZN", "NVDA", "MSFT", "GOOG"]

FEATURES = [
    'MA20','MA50','RSI','MACD','MACD_signal','MACD_hist',
    'BB_width','Return','Momentum3','Momentum10',
    'Trend','Volatility10','Lag1','Lag5'
]

###############################
# MODEL LOADER
###############################
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models")

def load_model(ticker):
    try:
        model = joblib.load(os.path.join(MODEL_PATH, f"{ticker}_model.pkl"))
        scaler = joblib.load(os.path.join(MODEL_PATH, f"{ticker}_scaler.pkl"))
        return model, scaler
    except Exception as e:
        st.warning(f"⚠️ Model not found for {ticker}")
        return None, None

###############################
# DATA FETCHER
###############################
def fetch_data(ticker):
    df = yf.download(ticker, period="1y", interval="1d")
    return df[['Open','High','Low','Close','Volume']]

###############################
# FEATURE ENGINEERING
###############################
def engineer_features(df):
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()

    #df['RSI'] = df['Close'].pct_change().rolling(14).mean()
    change = df['Close'].diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    


    df['MACD'] = df['Close'].ewm(12).mean() - df['Close'].ewm(26).mean()
    df['MACD_signal'] = df['MACD'].ewm(9).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    df['BB_width'] = (
        df['Close'].rolling(20).mean() + 2*df['Close'].rolling(20).std()
    ) - (
        df['Close'].rolling(20).mean() - 2*df['Close'].rolling(20).std()
    )

    df['Return'] = df['Close'].pct_change()
    df['Momentum3'] = df['Close']/df['Close'].shift(3)-1
    df['Momentum10'] = df['Close']/df['Close'].shift(10)-1
    df['Trend'] = df['Close'] - df['Close'].shift(10)
    df['Volatility10'] = df['Return'].rolling(10).std()

    df['Lag1'] = df['Close'].shift(1)
    df['Lag5'] = df['Close'].shift(5)

    df.dropna(inplace=True)
    return df
    


###############################
# PREDICTION ENGINE
###############################
def predict_next_day(ticker):
    # --- Load primary model ---
    model, scaler = load_model(ticker)
    if model is None:
        return "Model missing", 0

    # --- Get data ---
    df = fetch_data(ticker)
    df = engineer_features(df)

    if df.empty:
        return "No Data", 0
    
    # --- Clean data ---
    clean_df = df[FEATURES].replace([np.inf, -np.inf], np.nan).dropna()
    clean_df = clean_df[(clean_df != 0).any(axis=1)]

    if clean_df.empty:
        return "No valid data", 0
    
    # --- Prepare input ---
    X = clean_df.tail(1).values
    X_scaled = scaler.transform(X)

    # ------------------------------------------------------
    # 1. PRIMARY MODEL PREDICTION
    # ------------------------------------------------------
    try:
        pred = model.predict(X_scaled)[0]

        try:
            prob = model.predict_proba(X_scaled)[0][1]
        except:
            prob = 0.5

        return ("UP" if pred == 1 else "DOWN"), prob

    except Exception as e:
        # ------------------------------------------------------
        # 2. FALLBACK: XGBOOST MODEL ONLY FOR MSFT
        # ------------------------------------------------------
        if ticker == "MSFT":
            try:
                fallback_model = joblib.load(os.path.join(MODEL_PATH, "MSFT_xgb.pkl"))
                
                pred = fallback_model.predict(X_scaled)[0]
                prob = fallback_model.predict_proba(X_scaled)[0][1]

                return ("UP" if pred == 1 else "DOWN"), prob

            except Exception as e2:
                # fallback failed
                return "DOWN", 0.5

        # ------------------------------------------------------
        # 3. FOR OTHER STOCKS → fail safe
        # ------------------------------------------------------
        return "DOWN", 0.5

###############################
# PLOT FUNCTIONS
###############################

def price_chart(ticker):
    df = yf.download(ticker, period="1y")
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"].rolling(20).mean(), name="MA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"].rolling(50).mean(), name="MA50"))

    fig.update_layout(height=300, title=f"{ticker} Price Trend")
    st.plotly_chart(fig, use_container_width=True)


def indicator_chart(ticker):
    df = yf.download(ticker, period="1y")

    df['RSI'] = df['Close'].pct_change().rolling(14).mean()
    df['MACD'] = df['Close'].ewm(12).mean() - df['Close'].ewm(26).mean()

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"))

    fig.update_layout(height=300, title="Momentum Indicators")
    st.plotly_chart(fig, use_container_width=True)


from sklearn.metrics import roc_curve, auc
import plotly.graph_objects as go

def plot_roc_curve(model, scaler, ticker):
    # Fetch & prepare data
    df = fetch_data(ticker)
    df = engineer_features(df)

    if df.empty:
        st.warning("Not enough data for ROC analysis")
        return

    # Create X and y from historical data
    X = df[FEATURES].values[:-1]
    y = (df['Close'].shift(-1) > df['Close']).astype(int).values[:-1]  # next-day direction
    
    # Scale
    X_scaled = scaler.transform(X)

    # Predict probabilities
    try:
        probs = model.predict_proba(X_scaled)[:, 1]
    except:
        st.warning("🚫 ROC not available for this model (no predict_proba)")
        return

    # Compute ROC
    fpr, tpr, _ = roc_curve(y, probs)
    auc_score = auc(fpr, tpr)

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"ROC (AUC={auc_score:.2f})"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random", line=dict(dash="dash")))

    fig.update_layout(
        title=f"{ticker} ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=350
    )

    st.plotly_chart(fig, use_container_width=True)


def show_feature_importance(ticker):
    model, _ = load_model(ticker)

    if not hasattr(model, "feature_importances_"):
        st.info("Feature importance not supported for this model.")
        return

    imp = model.feature_importances_
    fig = go.Figure(go.Bar(x=imp, y=FEATURES, orientation="h"))
    fig.update_layout(height=350, title="Feature Importance")
    st.plotly_chart(fig, use_container_width=True)

###############################
# UI / DASHBOARD
###############################
st.title("📊 Stock Direction Forecasting Using Machine Learning")

ticker = st.selectbox("Select stock", tickers)

###############################
# PREDICTION CARD
###############################
if st.button("Predict Next Day Movement"):
    direction, prob = predict_next_day(ticker)
    st.subheader(f"Prediction for {ticker}:")
    
    if direction == "UP":
        st.success(f"📈 UP ({prob:.2f} confidence)")
    else:
        st.error(f"📉 DOWN ({prob:.2f} confidence)")

    st.progress(int(prob*100))

    # BUY/SELL Logic:
    if prob > 0.60 and direction == "UP":
        st.success("💡 Suggested Signal: BUY")
    elif prob > 0.60 and direction == "DOWN":
        st.error("💡 Suggested Signal: SELL")
    else:
        st.warning("💡 Suggested Signal: HOLD")
    
        # Show ROC Plot
    model, scaler = load_model(ticker)
    if model is not None:
        plot_roc_curve(model, scaler, ticker)




st.write("---")

###############################
# PRICE + INDICATORS
###############################
col1, col2 = st.columns(2)

with col1:
    price_chart(ticker)

with col2:
    indicator_chart(ticker)

###############################
# FEATURE IMPORTANCE
###############################
st.write("---")
st.subheader("Model Insight")
show_feature_importance(ticker)

###############################
# MULTISTOCK SCAN PANEL
###############################
st.write("---")
st.subheader(" Multi-Stock Market Scan")

scan = []
for t in tickers:
    try:
        d, p = predict_next_day(t)
        scan.append([t, d, round(p,2)])
    except Exception as e:
        scan.append([t, "Error", 0])

df_scan = pd.DataFrame(scan, columns=["Ticker","Direction","Confidence"])
st.table(df_scan)



