# 📊 European Option Pricing Assistant

An interactive **Streamlit dashboard** for pricing European options, calculating Greeks, and providing actionable trading insights using **real-time market data**. This project demonstrates the application of financial models such as **Black-Scholes** and **Binomial Tree** in option pricing and risk management.

---

## **Overview**

This dashboard allows users to:  

- Fetch **real-time stock and option data** from Yahoo Finance.  
- Calculate **Black-Scholes and Binomial option prices**.  
- Compute **Greeks**: Delta, Gamma, Vega, Theta, and Rho.  
- Compare **implied vs historical volatility**.  
- Generate **sensitivity charts** showing how price and Greeks change with spot price, volatility, and time to expiry.  
- Provide **trader recommendations** based on pricing and exposure.

It serves as both an **educational** and **analytical** tool for understanding option pricing dynamics.

---
## 🚀 Launch the Dashboard

You can interact with the European Option Pricing Assistant live on Streamlit:

[🔗 Open Dashboard on Streamlit](https://european-option-pricing-o2t2lsxcrerqe4ejs5hvdu.streamlit.app/)
---

## **Objective**

- Demonstrate practical application of option pricing models in Python.  
- Interpret the impact of key factors on option valuation.  
- Educate users about **Greeks and market sensitivities**.  
- Support decision-making for trading European options.  

---

## **Features**

- **Stock Overview:** Company name, market capitalization, sector, industry, country, currency.  
- **Option Summary:** Spot price, strike price, expiry, market mid-price, risk-free rate, implied & historical volatility, moneyness, extrinsic value.  
- **Greeks & Insights:** Delta, Gamma, Vega, Theta, Rho with educational explanations.  
- **Trader Recommendations:** Buy, sell, or hold signals based on model and market price.  
- **Sensitivity Charts:** Price vs volatility, price vs spot, delta vs spot, price vs time to expiry.  

---

## **Methodology**

### **1. Black-Scholes Model**
- Provides theoretical price of European call and put options.  
- Assumes **log-normal distribution** of stock prices and **constant volatility**.  
- Produces **closed-form solutions** for pricing and Greeks.  

### **2. Binomial Tree Model**
- A **discrete-time approach** for option pricing.  
- Constructs a recombining tree of underlying asset prices.  
- Allows for multiple steps, increasing accuracy.  
- Useful to **validate Black-Scholes prices**.

### **3. Implied vs Historical Volatility**
- **Historical Volatility (HV):** Based on past price movements.  
- **Implied Volatility (IV):** Derived from market prices; reflects expectations.  

### **4. Option Greeks**
- **Delta:** Sensitivity of option price to underlying price changes.  
- **Gamma:** Rate of change of delta; indicates risk near ATM.  
- **Vega:** Sensitivity to volatility changes.  
- **Theta:** Time decay; negative theta → loss over time.  
- **Rho:** Sensitivity to interest rate changes.

---

## **Use Cases**

- **Educational:** Understanding option pricing, Greeks, and market sensitivities.  
- **Trading Analysis:** Identify under- or overvalued options for potential strategies.  
- **Risk Management:** Assess directional exposure and sensitivity to volatility and interest rates.  
- **Portfolio Planning:** Evaluate impact of time decay and volatility changes on positions.  

---

## **Implications & Insights**

- **Pricing Accuracy:** Black-Scholes and Binomial models provide close results; differences highlight discretization effects.  
- **Market Mispricing:** Observed when options trade above or below theoretical prices → guides potential trades.  
- **Sensitivity Analysis:**  
  - Option prices increase with spot for calls, decrease for puts.  
  - Higher volatility increases premiums.  
  - Delta indicates directional exposure; Gamma indicates risk of rapid delta change.  
  - Theta highlights time decay of value.  
- **Trader Recommendations:** Model outputs and Greeks combine to suggest buy, hold, or sell strategies.

---

## **Future Scope**

- Add **American options pricing** using advanced Binomial or Monte Carlo methods.  
- Include **real-time option chain updates** for multiple tickers.  
- Expand **graphical analysis** for Greeks under different market scenarios.  
- Integrate **portfolio-level risk metrics** for multiple options.  
- Add **alerts and notifications** for mispricing or high-risk exposure.
