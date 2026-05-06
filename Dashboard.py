import numpy as np
import streamlit as st
import yfinance as yf
import pandas as pd
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error 
import plotly.graph_objects as go
from groq import Groq
import streamlit.components.v1 as components

# ---------------- SESSION STATE ----------------
if "balance" not in st.session_state:
    st.session_state.balance = 10000.0

if "shares" not in st.session_state:
    st.session_state.shares = 0

if "avg_price" not in st.session_state:
    st.session_state.avg_price = 0.0

if "total_invested" not in st.session_state: 
    st.session_state.total_invested = 0.0    

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Stock Market Dashboard", layout="wide")
st.markdown("""
<style>
    .stApp {
        background-color: #0b0e14;
        color: #e1e7ef;
        font-family: 'Inter', sans-serif;
    }
    h1 {
        font-size: 3rem !important;
        font-weight: 800 !important;
        background: linear-gradient(to right, #fff, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    [data-testid="stMetric"] {
        background: #161b22 !important;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 15px;
    }
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        color: #38bdf8 !important;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background-color: #38bdf8 !important;
        color: #0b0e14 !important;
        font-weight: bold;
        border: none;
    }
</style>
""", unsafe_allow_html=True)
# ------------------- SIDEBAR -------------------
st.title(" Stock Market DashBoard")
with st.sidebar:
    st.header("Settings")
    stock = st.text_input("Enter Stock", "AAPL").upper()
    if stock == "RELIANCE":
        stock = "RELIANCE.NS"   
    timeframe = st.selectbox("Select Range", ["3 Months", "6 Months", "1 Year", "5 Years","10 Years"])
    period_map = {
        "3 Months": "3mo", 
        "6 Months": "6mo", 
        "1 Year": "1y", 
        "5 Years": "5y",
        "10 Years": "10y"
    }
    period = period_map[timeframe]
    
    st.divider()
    st.write("### Portfolio Status")
    st.metric("Wallet", f"${st.session_state.balance:.2f}")
    st.metric("Holdings", f"{st.session_state.shares} Shares")

# ------------------- DATA FETCHING -------------------
time.sleep(2)
data = yf.download(stock, period=period, interval="1d", progress=False)

if data.empty:
    st.error(f"Invalid stock symbol: {stock}")
    st.stop()
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)
st.subheader("📉 Graph ")
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data.index,
    y=data['Close'],
    name="Close Price"
))
fig.update_layout(
    template="plotly_dark",
    plot_bgcolor="#020617",
    paper_bgcolor="#020617",
    font=dict(color="#e2e8f0"),
    margin=dict(l=10, r=10, t=30, b=10)
)

st.plotly_chart(fig, use_container_width=True)    
# ---------------- CANDLESTICK ----------------
st.subheader("📉 Candlestick + Volume")
fig = go.Figure()

# Candlestick
fig.add_trace(go.Candlestick(
    x=data.index,
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close'],
    name="Price"
))

# Volume bars
fig.add_trace(go.Bar(
    x=data.index,
    y=data['Volume'],
    name="Volume",
    yaxis="y2",
    opacity=0.3
))

fig.update_layout(
    template="plotly_dark",
    plot_bgcolor="#020617",
    paper_bgcolor="#020617",
    font=dict(color="#e2e8f0"),
    margin=dict(l=10, r=10, t=30, b=10),

    yaxis=dict(title="Price"),
    yaxis2=dict(
        title="Volume",
        overlaying="y",
        side="right",
        showgrid=False
    ),

    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig, use_container_width=True)
# ---------------- MOVING AVERAGES ----------------
data['MA20'] = data['Close'].rolling(20).mean()
data['MA50'] = data['Close'].rolling(50).mean()

st.subheader("📊 Moving Averages")

st.line_chart(data[['Close', 'MA20', 'MA50']])

# ---------------- RSI ----------------
def compute_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

data['RSI'] = compute_rsi(data)

st.subheader("📉 RSI Indicator")
rsi_val = data['RSI'].iloc[-1]
st.write(f"**Current RSI:** {rsi_val:.2f}")
if rsi_val > 70: st.warning("Status: Overbought")
elif rsi_val < 30: st.info("Status: Oversold")
# ---------------- SIGNAL LOGIC ----------------
data['Signal'] = 0

for i in range(1, len(data)):
    # BUY
    if (data['RSI'].iloc[i] < 30) and (data['Close'].iloc[i] > data['MA20'].iloc[i]):
        data.loc[data.index[i], 'Signal'] = 1

    # SELL
    elif (data['RSI'].iloc[i] > 70) and (data['Close'].iloc[i] < data['MA20'].iloc[i]):
        data.loc[data.index[i], 'Signal'] = -1
        buy_signals = data[data['Signal'] == 1]
sell_signals = data[data['Signal'] == -1]

# ---------------- FEATURE ENGINEERING ----------------
data['Prev1'] = data['Close'].shift(1)
data['Prev2'] = data['Close'].shift(2)
data['Prev3'] = data['Close'].shift(3)

data = data.dropna()
X = data[['Prev1','Prev2','Prev3']]
y = data['Close']

if X.shape[0] == 0:
    st.error("No data available for training. Increase timeframe.")
    st.stop()
if len(data) < 10:
    st.warning("⚠️ Not enough data for prediction. Try a larger timeframe.")
    st.stop()
# ---------------- MODEL ----------------
model = RandomForestRegressor()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)
model.fit(X_train, y_train)

# Show accuracy
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
st.metric("Model RMSE", f"{rmse:.2f}")

# ---------------- PREDICTION ----------------
last = X.iloc[-1].values.reshape(1, -1)
predicted_price = model.predict(last)[0]
current_price = data['Close'].values[-1]

investment = st.number_input(
    "💰 Enter Investment Amount",
    min_value=100.0,
    value=1000.0,
    step=100.0
)

shares = investment / current_price

future_value = shares * predicted_price

profit_loss = future_value - investment
profit_percent = (profit_loss / investment) * 100

st.markdown("## 💰 Investment Analysis")

col1, col2, col3 = st.columns(3)

col1.metric("💼 Investment", f"${investment:.2f}")
col2.metric("📈 Future Value", f"${future_value:.2f}")
# Profit / Loss with percentage
if profit_loss > 0:
    col3.metric(
        "🟢 Profit",
        f"${profit_loss:.2f}",
        delta=f"+{profit_percent:.2f}%"
    )
else:
    col3.metric(
        "🔴 Loss",
        f"${profit_loss:.2f}",
        delta=f"{profit_percent:.2f}%"
    )
# ---------------- METRICS ----------------
st.subheader("📊 Prediction Summary")

col1, col2 = st.columns(2)
col1.metric("💰 Current Price", f"{current_price:.2f}")
col2.metric("📈 Predicted Price", f"{predicted_price:.2f}")
# ---------------- SIGNAL ----------------
st.subheader("📢 Trading Signal")

if predicted_price > current_price:
    st.success("📈 BUY Signal")
else:
    st.error("📉 SELL Signal")
price_change = predicted_price - current_price

if price_change > 0:
    st.markdown(f"### 🟢 Uptrend Expected (+${price_change:.2f})")
else:
    st.markdown(f"### 🔴 Downtrend Expected (${price_change:.2f})")
# ---------------- RSI SIGNAL ----------------
if data['RSI'].iloc[-1] > 70:
    st.warning("⚠️ Overbought (RSI > 70)")
elif data['RSI'].iloc[-1] < 30:
    st.info("💡 Oversold (RSI < 30)")
 # ---------------- ADVANCED SIGNAL ----------------
st.subheader("📢 Smart Trading Signal")

rsi_value = data['RSI'].iloc[-1]
ma20 = data['MA20'].iloc[-1]

if (predicted_price > current_price) and (rsi_value < 35) and (current_price > ma20):
    st.success("🟢 STRONG BUY SIGNAL")

elif (predicted_price < current_price) and (rsi_value > 65) and (current_price < ma20):
    st.error("🔴 STRONG SELL SIGNAL")

else:
    st.warning("🟡 HOLD / WAIT")

    st.subheader("📊 Indicator Values")

col1, col2, col3 = st.columns(3)

col1.metric("RSI", f"{rsi_value:.2f}")
col2.metric("MA20", f"{ma20:.2f}")
col3.metric("Current Price", f"{current_price:.2f}")

# ---------------- QUANTITY ----------------
qty = st.number_input("📦 Quantity", min_value=1, value=1, step=1)

col_buy, col_sell = st.columns(2)

# ---------------- BUY ----------------
if st.button("🟢 Buy"):
    cost = qty * current_price
    if st.session_state.balance >= cost:
        st.session_state.balance -= cost
        st.session_state.total_invested += cost

        # update avg price
        total_shares_cost = st.session_state.avg_price * st.session_state.shares
        total_shares_cost += cost

        st.session_state.shares += qty
        st.session_state.avg_price = total_shares_cost / st.session_state.shares

        st.success("Bought successfully")
# ---------------- SELL ----------------
if st.button("🔴 Sell"):
    if st.session_state.shares >= qty:
        sell_value = qty * current_price

        st.session_state.balance += sell_value
        st.session_state.shares -= qty

        # reduce invested amount proportionally
        st.session_state.total_invested -= qty * st.session_state.avg_price

        st.success("Sold successfully")

portfolio_value = st.session_state.balance + (st.session_state.shares * current_price)

profit = portfolio_value - st.session_state.total_invested

if "chat" not in st.session_state:
    st.session_state.chat = []

if "chat_open" not in st.session_state:
    st.session_state.chat_open = False
# -------- GROQ CLIENT --------
client = Groq(api_key="Your API Key") 
# -------- SAFE PREDICTION --------
next_price = None  

try:
    if len(X) < 3:
        next_price = data['Close'].rolling(3).mean().iloc[-1]
    else:
        last = X.iloc[-1].values.reshape(1, -1)
        next_price = model.predict(last)[0]
except:
    next_price = data['Close'].iloc[-1]
    predicted_value = next_price if next_price is not None else "N/A"

if "price" not in st.session_state:
    st.session_state.price = 0.0
st.subheader("💼 Predicted Price")

col1, col2 = st.columns(2)

col1.metric("💰 Price", f"Predicted Price: {predicted_price:.2f}")

user_input = st.chat_input("Ask about the stock...")

if user_input:
    # save user message
    st.session_state.chat.append({"role": "user", "content": user_input})

    # API call
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=st.session_state.chat
    )

    reply = response.choices[0].message.content

    # save AI reply
    st.session_state.chat.append({"role": "assistant", "content": reply})

for msg in st.session_state.chat:     
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])
context = f"""
You are a real-time stock assistant.

Use ONLY the data below to answer.

Stock: {stock}
Current Price: {current_price}
Predicted Price: {next_price}
RSI: {data['RSI'].iloc[-1] if 'RSI' in data.columns else 'N/A'}
Portfolio Profit: {profit}
"""

# -------- CHAT POPUP --------
if st.session_state.chat_open:
    col1, col2 = st.columns([4,1])
    # clear chat
    with col1:
        if st.button("🧹 Clear Chat"):
            st.session_state.chat = []
            st.rerun()
    # close chat
    with col2:
        if st.button("❌ Close"):
            st.session_state.chat_open = False
            st.rerun()       

             
