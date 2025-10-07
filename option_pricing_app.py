# option_pricing_dashboard_enhanced.py
import math
import numpy as np
import pandas as pd
from datetime import datetime
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq, minimize_scalar
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="European Option Pricing Assistant", layout="wide")

# -----------------------------
# 1. Helper Functions
# -----------------------------
def bs_d1_d2(S, K, r, sigma, T):
    if T <= 0 or sigma <= 0:
        return float('inf'), float('inf')
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2

def bs_price(S, K, r, sigma, T, option_type='call'):
    if T <= 0:
        return max(0.0, (S-K) if option_type=='call' else (K-S))
    d1, d2 = bs_d1_d2(S, K, r, sigma, T)
    if option_type=='call':
        return S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
    else:
        return K*math.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def binomial_price_european(S, K, r, sigma, T, N=100, option_type='call', discrete_rf=False):
    if T <= 0:
        return max(0.0, (S-K) if option_type=='call' else (K-S))
    dt = T/N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1/u
    p = (math.exp(r*dt)-d)/(u-d) if not discrete_rf else (1+r*dt - d)/(u-d)
    discount = math.exp(-r*dt) if not discrete_rf else 1/(1+r*dt)
    j = np.arange(0, N+1)
    ST = S * (u**j) * (d**(N-j))
    payoff = np.maximum(ST-K, 0) if option_type=='call' else np.maximum(K-ST, 0)
    for i in range(N, 0, -1):
        payoff = discount*(p*payoff[1:] + (1-p)*payoff[:-1])
    return float(payoff[0])

def bs_greeks(S, K, r, sigma, T, option_type='call'):
    if T <= 0:
        return {'delta': None, 'gamma':0.0, 'vega':0.0, 'theta':0.0, 'rho':0.0}
    d1,d2 = bs_d1_d2(S,K,r,sigma,T)
    pdf_d1 = norm.pdf(d1)
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    delta = N_d1 if option_type=='call' else N_d1-1
    gamma = pdf_d1/(S*sigma*math.sqrt(T))
    vega = S*pdf_d1*math.sqrt(T)
    theta = (-S*pdf_d1*sigma/(2*math.sqrt(T)) - r*K*math.exp(-r*T)*N_d2) if option_type=='call' else (-S*pdf_d1*sigma/(2*math.sqrt(T)) + r*K*math.exp(-r*T)*norm.cdf(-d2))
    rho = K*T*math.exp(-r*T)*N_d2 if option_type=='call' else -K*T*math.exp(-r*T)*norm.cdf(-d2)
    return {'delta':float(delta), 'gamma':float(gamma), 'vega':float(vega), 'theta':float(theta), 'rho':float(rho)}

def implied_volatility_from_price(market_price, S, K, r, T, option_type='call', sigma_bounds=(1e-6,5.0), tol=1e-8):
    intrinsic = max(0.0, S-K*math.exp(-r*T)) if option_type=='call' else max(0.0, K*math.exp(-r*T)-S)
    if market_price <= intrinsic+1e-12: return 0.0
    def obj(vol): return bs_price(S,K,r,vol,T,option_type)-market_price
    low, high = sigma_bounds
    try:
        if obj(low)*obj(high)<=0:
            return float(brentq(obj,low,high,xtol=tol,maxiter=200))
    except: pass
    res = minimize_scalar(lambda vol:(bs_price(S,K,r,vol,T,option_type)-market_price)**2,bounds=sigma_bounds,method='bounded')
    return float(res.x if res.success else low)

@st.cache_data
def get_spot_yf(ticker):
    hist = yf.Ticker(ticker).history(period='5d')
    return float(hist['Close'].iloc[-1])

@st.cache_data
def get_option_chain_yf(ticker, expiry):
    oc = yf.Ticker(ticker).option_chain(expiry)
    return oc.calls.copy(), oc.puts.copy()

def compute_mid_price(row):
    bid, ask, last = row.get('bid', np.nan), row.get('ask', np.nan), row.get('lastPrice', np.nan)
    if not np.isnan(bid) and not np.isnan(ask) and (bid>0 or ask>0): return float((bid+ask)/2)
    if not np.isnan(last) and last>=0: return float(last)
    return float('nan')

@st.cache_data
def historical_volatility(ticker, period='1y', interval='1d', annualization=252):
    hist = yf.Ticker(ticker).history(period=period, interval=interval)
    log_ret = np.log(hist['Close']/hist['Close'].shift(1)).dropna()
    return float(log_ret.std()*math.sqrt(annualization))

def interpret_metrics(bs_price_val, binom_price, delta, gamma, vega, theta, rho, market_price, S, K):
    insights = {}
    diff = binom_price-bs_price_val
    insights['price'] = f"BS: {bs_price_val:.2f}, Binomial: {binom_price:.2f}, Diff: {diff:.4f}. "
    if market_price<bs_price_val: insights['price'] += "✅ Undervalued → potential buy."
    elif market_price>bs_price_val: insights['price'] += "⚠️ Overvalued → consider selling."
    else: insights['price'] += "ℹ️ Fairly priced."
    insights['delta'] = f"{delta:.4f} → {'Strong' if delta>0.5 else 'Moderate'} directional exposure."
    insights['gamma'] = f"{gamma:.4f} → High gamma → delta changes quickly near ATM."
    insights['vega'] = f"{vega:.4f} → Sensitive to volatility changes."
    insights['theta'] = f"{theta:.4f} → Negative → loses value over time."
    insights['rho'] = f"{rho:.4f} → Sensitivity to interest rate."
    insights['moneyness'] = f"S/K = {S/K:.3f} → {'ITM' if (S>K and delta>0) or (S<K and delta<0) else 'OTM/ATM'}"
    intrinsic = max(S-K,0) if delta>0 else max(K-S,0)
    extrinsic = market_price - intrinsic
    insights['extrinsic'] = f"{extrinsic:.2f} → portion attributable to time & volatility."
    return insights

# -----------------------------
# 2. Sidebar Inputs with Explanations
# -----------------------------
st.sidebar.header("Option Parameters & Significance")
ticker_symbol = st.sidebar.text_input("Ticker Symbol", value="AAPL", help="Enter the stock symbol to fetch spot and options data.")
option_type = st.sidebar.selectbox("Option Type", ["call","put"], help="Call: right to buy, Put: right to sell.")
r = st.sidebar.number_input("Risk-Free Rate (annual, decimal)", value=0.06, step=0.001, help="Annual risk-free rate used in pricing models.")
N_tree = st.sidebar.slider("Binomial Steps", min_value=10, max_value=1000, value=200, step=10, help="Higher steps → more accurate Binomial price, increases computation.")

try:
    S = get_spot_yf(ticker_symbol)
    ticker = yf.Ticker(ticker_symbol)
    expiries = ticker.options
    if not expiries: st.warning("No options found."); st.stop()
    expiry = st.sidebar.selectbox("Expiry", expiries, help="Select option expiry date.")
    calls_df, puts_df = get_option_chain_yf(ticker_symbol, expiry)
    calls_df['mid'] = calls_df.apply(compute_mid_price, axis=1)
    puts_df['mid'] = puts_df.apply(compute_mid_price, axis=1)
except Exception as e:
    st.error(f"Market data error: {e}"); st.stop()

strike_input = st.sidebar.text_input("Strike Price (leave blank for ATM)", help="If blank, the closest strike to spot price is selected.")
if strike_input: strike = float(strike_input)
else:
    strikes = calls_df['strike'].values
    strike = float(strikes[np.argmin(np.abs(strikes - S))])

row = calls_df.loc[calls_df['strike']==strike].iloc[0] if option_type=='call' else puts_df.loc[puts_df['strike']==strike].iloc[0]
market_mid = compute_mid_price(row)

T = max((datetime.strptime(expiry,'%Y-%m-%d')-datetime.today()).days/365, 1e-6)
iv = implied_volatility_from_price(market_mid, S, strike, r, T, option_type)
bs_p = bs_price(S, strike, r, iv, T, option_type)
bs_g = bs_greeks(S, strike, r, iv, T, option_type)
binom_price = binomial_price_european(S, strike, r, iv, T, N=N_tree, option_type=option_type)
hv = historical_volatility(ticker_symbol)
insights = interpret_metrics(bs_p, binom_price, bs_g['delta'], bs_g['gamma'], bs_g['vega'], bs_g['theta'], bs_g['rho'], market_mid, S, strike)

# -----------------------------
# 3. Dashboard Layout
# -----------------------------
st.title("📊 European Option Pricing Assistant")
st.markdown("Interactive assistant providing **pricing, Greeks, insights, and actionable recommendations**.")

st.subheader("MFA335N Applied Derivates and Risk Management: CIA III")
st.markdown("by SALMA ILYAS 24224013.")
st.markdown("Submitted to Dr. Nitin Kulshrestha")

# -----------------------------
# Stock Overview
# -----------------------------
st.subheader("🏢 Stock Overview")

try:
    ticker_info = ticker.info
    company_name = ticker_info.get('longName', 'N/A')
    market_cap = ticker_info.get('marketCap', 'N/A')
    sector = ticker_info.get('sector', 'N/A')
    industry = ticker_info.get('industry', 'N/A')
    country = ticker_info.get('country', 'N/A')
    currency = ticker_info.get('currency', 'N/A')
    
    st.markdown(f"""
    **Company Name:** {company_name}  
    **Market Cap:** {market_cap if market_cap=='N/A' else f'{market_cap/1e9:.2f}B'} {currency}  
    **Sector:** {sector}  
    **Industry:** {industry}  
    **Country:** {country}  
    **Currency:** {currency}  
    """)
except Exception as e:
    st.warning(f"Unable to fetch stock overview: {e}")


# Option Summary & Metrics
col1, col2 = st.columns([1.5,1.5])

# Option Summary & Metrics with Educational Explanations
col1, col2 = st.columns([1.5, 1.5])

# Option Summary & Metrics with Explanations on Next Line
col1, col2 = st.columns([1.5, 1.5])

with col1:
    st.markdown("### 💹 Option Summary")
    st.markdown(f"- **Spot (S):** {S:.2f}  <br>Current price of the underlying asset.", unsafe_allow_html=True)
    st.markdown(f"- **Strike (K):** {strike:.2f}  <br>Price at which the option can be exercised.", unsafe_allow_html=True)
    st.markdown(f"- **Expiry:** {expiry}  <br>Date when the option contract expires.", unsafe_allow_html=True)
    st.markdown(f"- **Market Mid:** {market_mid:.2f}  <br>Average of bid and ask prices; gives market consensus.", unsafe_allow_html=True)
    st.markdown(f"- **Risk-Free Rate:** {r:.2%}  <br>Used in pricing to discount future cash flows.", unsafe_allow_html=True)
    st.markdown(f"- **Implied Volatility (σ):** {iv:.2%}  <br>Market-expected volatility derived from option price.", unsafe_allow_html=True)
    st.markdown(f"- **Historical Volatility:** {hv:.2%}  <br>Past realized volatility of the underlying asset.", unsafe_allow_html=True)
    st.markdown(f"- **Moneyness:** {insights['moneyness']}  <br>Indicates ITM/OTM/ATM status.", unsafe_allow_html=True)
    st.markdown(f"- **Extrinsic Value:** {insights['extrinsic']}  <br>Portion of option price due to time & volatility.", unsafe_allow_html=True)
    st.markdown(f"- **Pricing Models:** BS={bs_p:.2f}, Binomial={binom_price:.2f}  <br>Comparison of theoretical prices.", unsafe_allow_html=True)

with col2:
    st.markdown("### ⚡ Greeks & Insights")
    st.markdown(f"- **Delta:** {insights['delta']}  <br>Measures sensitivity of option price to changes in underlying asset. High delta → strong directional exposure.", unsafe_allow_html=True)
    st.markdown(f"- **Gamma:** {insights['gamma']}  <br>Rate of change of delta; high gamma means delta changes rapidly near ATM.", unsafe_allow_html=True)
    st.markdown(f"- **Vega:** {insights['vega']}  <br>Sensitivity of option price to volatility changes. Higher vega → more sensitive.", unsafe_allow_html=True)
    st.markdown(f"- **Theta:** {insights['theta']}  <br>Time decay of option value; negative theta → loses value as expiry approaches.", unsafe_allow_html=True)
    st.markdown(f"- **Rho:** {insights['rho']}  <br>Sensitivity of option price to interest rate changes.", unsafe_allow_html=True)


# Trader Recommendations
st.subheader("💼 Trader Recommendations")
if market_mid < bs_p:
    st.success("✅ Undervalued → potential buy opportunity.")
elif market_mid > bs_p:
    st.warning("⚠️ Overvalued → consider selling or waiting.")
else:
    st.info("ℹ️ Fairly priced according to model.")

if bs_g['delta'] is not None:
    st.info(f"Directional Exposure: {'High (Delta >0.5)' if bs_g['delta']>0.5 else 'Moderate (Delta <0.5)'}")

# -----------------------------
# 4. Sensitivity Charts
# -----------------------------
st.subheader("📈 Sensitivity Charts")

# Define ranges
S_vals = np.linspace(0.8*S, 1.2*S, 50)
T_vals = np.linspace(0.01, T, 50)
vols = np.linspace(0.1, 0.6, 50)

# Use two columns to display charts side by side
col1, col2 = st.columns(2)

# --- Option Price vs Volatility ---
with col1:
    st.markdown("**Option Price vs Volatility:** Shows how the option price reacts to changes in market volatility. Higher volatility generally increases option premiums.")
    prices_vol = [bs_price(S, strike, r, v, T, option_type) for v in vols]
    fig, ax = plt.subplots(figsize=(5,3))
    ax.plot(vols, prices_vol, color='navy', lw=2)
    ax.set_xlabel("Volatility (σ)")
    ax.set_ylabel("Option Price")
    ax.grid(True)
    st.pyplot(fig)

# --- Option Price vs Spot Price ---
with col2:
    st.markdown("**Option Price vs Spot Price:** Illustrates sensitivity to underlying asset movements. Call prices increase with the spot price; put prices decrease.")
    prices_S = [bs_price(Sv, strike, r, iv, T, option_type) for Sv in S_vals]
    fig, ax = plt.subplots(figsize=(5,3))
    ax.plot(S_vals, prices_S, color='purple', lw=2)
    ax.set_xlabel("Spot Price")
    ax.set_ylabel("Option Price")
    ax.grid(True)
    st.pyplot(fig)

# --- Delta vs Spot Price ---
with col1:
    st.markdown("**Delta vs Spot Price:** Delta measures directional exposure — how much the option price changes with a 1-unit move in the underlying asset. High delta → strong directional sensitivity.")
    deltas = [bs_greeks(Sv, strike, r, iv, T, option_type)['delta'] for Sv in S_vals]
    fig, ax = plt.subplots(figsize=(5,3))
    ax.plot(S_vals, deltas, color='orange', lw=2)
    ax.set_xlabel("Spot Price")
    ax.set_ylabel("Delta")
    ax.grid(True)
    st.pyplot(fig)

# --- Option Price vs Time to Expiry ---
with col2:
    st.markdown("**Option Price vs Time to Expiry:** Shows the effect of time decay (Theta). As expiry approaches, extrinsic value decreases.")
    prices_T = [bs_price(S, strike, r, iv, Tv, option_type) for Tv in T_vals]
    fig, ax = plt.subplots(figsize=(5,3))
    ax.plot(T_vals, prices_T, color='green', lw=2)
    ax.set_xlabel("Time to Expiry (Years)")
    ax.set_ylabel("Option Price")
    ax.grid(True)
    st.pyplot(fig)
