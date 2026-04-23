#PERFECT+SECTOR+CURRENCY
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from datetime import date, timedelta
from scipy.optimize import minimize
from groq import Groq
import streamlit.components.v1 as components

# ==================================
# Configuration & Styling
# ==================================
st.set_page_config(
    page_title="Stock Summarizer Pro",
    layout="wide",
    page_icon="📊"
)

# Custom CSS for a "TradingView-like" dark aesthetic + Floating Chat
st.markdown("""
    <style>
    .stApp {
        background-color: #131722;
        color: #d1d4dc;
    }
    .stTextInput > div > div > input {
        color: #d1d4dc;
        background-color: #2a2e39;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    [data-testid="stMetricLabel"] {
        color: #b2b5be !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1e222d;
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #787b86;
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2a2e39 !important;
        color: #26a69a !important;
        border-bottom: 2px solid #26a69a !important;
    }

    /* Portfolio card styling */
    .portfolio-metric-card {
        background: linear-gradient(135deg, #1e222d 0%, #262b38 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255,255,255,0.06);
        text-align: center;
    }
    .portfolio-metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #ffffff;
    }
    .portfolio-metric-label {
        font-size: 0.8rem;
        color: #787b86;
        margin-top: 4px;
    }
    /* Floating chatbot styling is handled entirely via JS in components.html */
    </style>
    """, unsafe_allow_html=True)

# ----------------------------------
# Sector → Company → (Ticker, Currency) Mapping
# ----------------------------------
SECTOR_COMPANIES = {
    "Technology": {
        "Apple": ("AAPL", "$"),
        "Microsoft": ("MSFT", "$"),
        "Google": ("GOOGL", "$"),
        "Amazon": ("AMZN", "$"),
        "Nvidia": ("NVDA", "$")
    },
    "Finance": {
        "JPMorgan": ("JPM", "$"),
        "Goldman Sachs": ("GS", "$"),
        "HDFC Bank": ("HDFCBANK.NS", "₹"),
        "ICICI Bank": ("ICICIBANK.NS", "₹"),
        "Bajaj Finance": ("BAJFINANCE.NS", "₹")
    },
    "Energy": {
        "Reliance": ("RELIANCE.NS", "₹"),
        "ONGC": ("ONGC.NS", "₹"),
        "Tata Power": ("TATAPOWER.NS", "₹"),
        "Adani Green Energy": ("ADANIGREEN.NS", "₹"),
        "JSW Energy": ("JSWENERGY.NS", "₹"),
        "ExxonMobil": ("XOM", "$"),
        "Chevron": ("CVX", "$")
    },
    "Automobile": {
        "Tesla": ("TSLA", "$"),
        "Tata Motors Ltd": ("TMPV", "₹"),
        "Maruti": ("MARUTI.NS", "₹"),
        "Bajaj Auto Ltd": ("BAJAJ-AUTO.NS", "₹"),
        "Ford": ("F", "$"),
        "Nifty Auto Index": ("^CNXAUTO", "₹")
    },
    "Pharma": {
        "Sun Pharma": ("SUNPHARMA.NS", "₹"),
        "Cipla": ("CIPLA.NS", "₹"),
        "Mankind Pharma": ("MANKIND.NS", "₹"),
        "Ajanta Pharma": ("AJANTPHARM.NS", "₹"),
        "Dr Reddy": ("DRREDDY.NS", "₹"),
        "Pfizer": ("PFE", "$")
    }
}

# ==================================
# Helper Functions
# ==================================

def fetch_data(ticker, start, end):
    """
    Fetches historical data from Yahoo Finance.
    Handles errors and empty returns.
    """
    try:
        # yfinance download
        df = yf.download(ticker, start=start, end=end, progress=False)
        
        # Handling MultiIndex columns if they exist (common in new yf versions)
        if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
             # Dropping the 'Ticker' level if present, usually the column is (Price, Ticker)
            df.columns = df.columns.droplevel(1)
            
        if df.empty:
            return None
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

def calculate_support_resistance(df, window=14):
    """
    Identifies dynamic Support and Resistance levels based on rolling min/max.
    """
    df = df.copy()
    df['Support'] = df['Low'].rolling(window=window, center=False).min()
    df['Resistance'] = df['High'].rolling(window=window, center=False).max()
    return df

def compute_statistics(df):
    """
    Computes key financial metrics for a given stock dataframe.
    Uses dropna() on Close prices to avoid NaN issues.
    """
    close_clean = df["Close"].dropna()
    returns = close_clean.pct_change().dropna()
    
    if len(returns) == 0:
        return {}
    
    cagr_val = 0
    if len(close_clean) > 1:
        first_price = close_clean.iloc[0]
        last_price = close_clean.iloc[-1]
        if first_price > 0:
            cagr_val = (last_price / first_price) ** (252 / len(close_clean)) - 1

    stats = {
        "Annual Volatility": returns.std() * np.sqrt(252),
        "Sharpe Ratio": (np.sqrt(252) * returns.mean() / returns.std()) if returns.std() != 0 else 0,
        "Coefficient of Variation": (returns.std() / returns.mean()) if returns.mean() != 0 else 0,
        "Max Drawdown": ((1 + returns).cumprod() / (1 + returns).cumprod().cummax() - 1).min(),
        "CAGR": cagr_val,
        "Mean Daily Return": returns.mean(),
        "Average Volume": df["Volume"].mean()
    }
    return stats

def plot_tradingview_chart(df, ticker, indicators):
    """
    Creates a professional TradingView-style candlestick chart with Volume.
    """
    # Create Subplots: Row 1 = Price, Row 2 = Volume
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        subplot_titles=(f"{ticker} Price Action", "Volume"),
        row_heights=[0.7, 0.3]
    )

    # 1. Candlestick Trace
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Candles',
            increasing_line_color='#26a69a', # TradingView Green
            decreasing_line_color='#ef5350', # TradingView Red
            increasing_fillcolor='#26a69a',
            decreasing_fillcolor='#ef5350'
        ),
        row=1, col=1
    )

    # 2. Indicators
    close = df['Close']
    
    if "SMA" in indicators:
        sma = ta.trend.sma_indicator(close, window=20)
        fig.add_trace(go.Scatter(x=df.index, y=sma, name="SMA 20", line=dict(color='#ff9800', width=1.5)), row=1, col=1)

    if "EMA" in indicators:
        ema = ta.trend.ema_indicator(close, window=20)
        fig.add_trace(go.Scatter(x=df.index, y=ema, name="EMA 20", line=dict(color='#2962ff', width=1.5)), row=1, col=1)

    if "Bollinger Bands" in indicators:
        try:
            bb_upper = ta.volatility.bollinger_hband(close, window=20)
            bb_lower = ta.volatility.bollinger_lband(close, window=20)
            
            # Changed to a visible Cyan color with solid opacity for lines
            fig.add_trace(go.Scatter(x=df.index, y=bb_upper, name="BB Upper", 
                                     line=dict(color='#00e676', width=1.2),  # Bright Green/Cyan
                                     opacity=0.7), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=df.index, y=bb_lower, name="BB Lower", 
                                     line=dict(color='#00e676', width=1.2), 
                                     opacity=0.7,
                                     fill='tonexty', fillcolor='rgba(0, 230, 118, 0.1)'), row=1, col=1)
        except Exception:
            pass # handle data too short

    if "Support & Resistance" in indicators:
        df_sr = calculate_support_resistance(df)
        # Plotting the most recent levels as horizontal lines or continuous step lines
        # Continuous lines for dynamic view:
        fig.add_trace(go.Scatter(x=df.index, y=df_sr['Resistance'], name="Resistance", line=dict(color='rgba(239, 83, 80, 0.6)', dash='dot', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df_sr['Support'], name="Support", line=dict(color='rgba(38, 166, 154, 0.6)', dash='dot', width=1)), row=1, col=1)

    # 3. Volume Trace
    colors = [
        '#26a69a' if row['Close'] >= row['Open'] else '#ef5350'
        for index, row in df.iterrows()
    ]
    
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.5
        ),
        row=2, col=1
    )

    # Layout Updates for "TradingView" Feel
    fig.update_layout(
        template="plotly_dark",
        height=700,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.02, x=0, bgcolor='rgba(0,0,0,0)'),
        xaxis_rangeslider_visible=False,
        dragmode='pan',
        hovermode='x unified'
    )
    
    # Grid styling
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#2a2e39')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#2a2e39')

    return fig


# ==================================
# Portfolio Helper Functions
# ==================================

def portfolio_performance(weights, mean_returns, cov_matrix):
    """Calculate annualized portfolio return, volatility, and Sharpe ratio."""
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe = returns / std if std != 0 else 0
    return returns, std, sharpe

def neg_sharpe(weights, mean_returns, cov_matrix):
    """Negative Sharpe ratio for minimization."""
    return -portfolio_performance(weights, mean_returns, cov_matrix)[2]

def min_variance(weights, mean_returns, cov_matrix):
    """Portfolio variance for minimization."""
    return portfolio_performance(weights, mean_returns, cov_matrix)[1]

def optimize_portfolio(mean_returns, cov_matrix, num_assets, objective='sharpe'):
    """
    Optimize portfolio using scipy.optimize.minimize.
    objective: 'sharpe' for max Sharpe, 'min_vol' for minimum volatility
    """
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    init_guess = num_assets * [1. / num_assets]
    
    if objective == 'sharpe':
        result = minimize(neg_sharpe, init_guess, args=(mean_returns, cov_matrix),
                         method='SLSQP', bounds=bounds, constraints=constraints)
    else:
        result = minimize(min_variance, init_guess, args=(mean_returns, cov_matrix),
                         method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def get_close_series(df):
    """
    Safely extract Close price as a 1-D pandas Series from a DataFrame.
    Handles cases where df['Close'] might be 2-D due to yfinance MultiIndex quirks.
    """
    close = df['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.squeeze()


# ==================================
# Chatbot Helper
# ==================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

SYSTEM_PROMPT = """You are FinBot, an expert AI assistant specializing in quantitative finance, stock markets, and portfolio management. You are embedded inside a Stock Summarizer Pro application.

Your capabilities:
- Explain financial concepts (Sharpe ratio, CAGR, volatility, drawdowns, etc.)
- Discuss market trends, sectors, and individual stocks
- Provide insights on portfolio optimization (Markowitz, efficient frontier, risk-return tradeoffs)
- Explain technical indicators (SMA, EMA, Bollinger Bands, RSI, MACD, support/resistance)
- Help interpret charts and statistical data
- Discuss fundamental and technical analysis approaches

Rules:
- Be concise yet informative. Use bullet points for clarity.
- Always mention that you are not providing financial advice — for educational/informational purposes only.
- Use quantitative reasoning when possible.
- Format responses with markdown for readability.
- If asked about real-time prices, remind the user to check the Stats tab or a live data source.
"""

def get_groq_response(messages):
    """Get a response from the Groq API."""
    try:
        client = Groq(api_key=GROQ_API_KEY)
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=1024,
            top_p=0.9,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error connecting to AI service: {str(e)}"


# ==================================
# Main App UI
# ==================================

st.title("📊 Stock Summarizer Pro")

# --- Sidebar Controls ---
st.sidebar.header("Configuration")

# 1. Selection Mode
input_mode = st.sidebar.radio("Input Mode", ["Select by Sector", "Manual Input"], index=0)

tickers = []
# Dictionary to store explicit currency for each ticker
ticker_currency_map = {}
selected_sectors = []  # Initialize to avoid NameError

if input_mode == "Select by Sector":
    # Sector Selection
    sorted_sectors = sorted(SECTOR_COMPANIES.keys())
    # CHANGED: Multiselect for sectors
    selected_sectors = st.sidebar.multiselect("Select Sector(s)", sorted_sectors, placeholder="Choose sectors...")
    
    if selected_sectors:
        # Aggregate companies from ALL selected sectors
        available_companies = {}
        for sector in selected_sectors:
            available_companies.update(SECTOR_COMPANIES[sector])
            
        # Company Selection based on Sectors
        selected_companies = st.sidebar.multiselect(
            f"Select Companies",
            options=available_companies.keys(),
            default=[] 
        )
        
        # Map names to tickers AND populate currency map
        for name in selected_companies:
            ticker_data = available_companies[name]
            # Handle tuple (Ticker, Currency) or just string for backward compat if needed (but we updated all)
            if isinstance(ticker_data, tuple):
                t_symbol = ticker_data[0]
                t_curr = ticker_data[1]
                tickers.append(t_symbol)
                ticker_currency_map[t_symbol] = t_curr
            else:
                tickers.append(ticker_data)
        
elif input_mode == "Manual Input":
    ticker_input = st.sidebar.text_input(
        "Enter Stock Tickers (comma separated)", 
        value="",
        placeholder="e.g. AAPL, TSLA, BTC-USD"
    )
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

# 2. Date Range
start_date = st.sidebar.date_input("Start Date", date.today() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", date.today())

# 3. Indicators
indicators_list = ["SMA", "EMA", "Bollinger Bands", "Support & Resistance"]
selected_indicators = st.sidebar.multiselect("Technical Indicators", indicators_list, default=[])


# --- Main Content with Tabs ---

tab1, tab2 = st.tabs(["📊 Stats", "💼 Portfolio"])

# ==================================
# TAB 1: Stats (Original Functionality)
# ==================================
with tab1:
    if not tickers:
        if input_mode == "Select by Sector":
            if not selected_sectors:
                st.info("👈 Please start by selecting one or more **Sectors** from the sidebar.")
            else:
                st.info(f"👈 Now select one or more related companies from the sidebar.")
        else:
            st.info("👈 Please enter at least one stock ticker in the sidebar.")
    else:
        stats_data = []

        for ticker_symbol in tickers:
            st.subheader(f"Analyzing: {ticker_symbol}")
            
            with st.spinner(f"Loading data for {ticker_symbol}..."):
                df = fetch_data(ticker_symbol, start_date, end_date)
                
            if df is None:
                st.warning(f"Could not load data for {ticker_symbol}. Check the ticker symbol.")
                continue
                
            # Chart
            fig = plot_tradingview_chart(df, ticker_symbol, selected_indicators)
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
            
            # Calculate Stats
            stats = compute_statistics(df)
            stats["Ticker"] = ticker_symbol
            stats_data.append(stats)

            # Basic Stats Expander — with NaN-safe close price extraction
            with st.expander(f"Key Statistics for {ticker_symbol}"):
                if len(df) > 0:
                    close_valid = df['Close'].dropna()
                    
                    if len(close_valid) >= 2:
                        last_close = float(close_valid.iloc[-1])
                        prev_close = float(close_valid.iloc[-2])
                        change = last_close - prev_close
                        pct_change = (change / prev_close) * 100 if prev_close != 0 else 0
                    elif len(close_valid) == 1:
                        last_close = float(close_valid.iloc[0])
                        change = 0.0
                        pct_change = 0.0
                    else:
                        last_close = 0.0
                        change = 0.0
                        pct_change = 0.0
                    
                    # Determine currency
                    # Priority: Explicit Map -> Heuristic Foldblack
                    if ticker_symbol in ticker_currency_map:
                        currency = ticker_currency_map[ticker_symbol]
                    else:
                        # Fallback for manual inputs
                        if ticker_symbol.endswith((".NS", ".BO")) or ticker_symbol.startswith(("^CNX", "^NSE", "^BSE")):
                            currency = "₹"
                        else:
                            currency = "$"

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Last Price", f"{currency} {last_close:.2f}", f"{change:.2f} ({pct_change:.2f}%)")
                    col2.metric("High (Period)", f"{currency} {df['High'].max():.2f}")
                    col3.metric("Low (Period)", f"{currency} {df['Low'].min():.2f}")
                    col4.metric("Volume (Avg)", f"{df['Volume'].mean():.0f}")
            
            st.markdown("---")

        # --- Statistical Summary Table ---
        if stats_data:
            st.subheader("📈 Statistical Summary Comparison")
            st.markdown("Compare key metrics across all selected assets.")
            
            stats_df = pd.DataFrame(stats_data).set_index("Ticker")
            
            # Format styling for better readability
            # Using style.format with strings directly avoids potential type issues
            st.dataframe(
                stats_df.style.format({
                    "Annual Volatility": "{:.2%}",
                    "Sharpe Ratio": "{:.2f}",
                    "Coefficient of Variation": "{:.2f}",
                    "Max Drawdown": "{:.2%}",
                    "CAGR": "{:.2%}",
                    "Mean Daily Return": "{:.4%}",
                    "Average Volume": "{:,.0f}"
                }),
                use_container_width=True
            )


# ==================================
# TAB 2: Portfolio Analyzer & Optimizer
# ==================================
with tab2:
    if not tickers or len(tickers) < 2:
        st.info("💼 Please select **at least 2 stocks** from the sidebar to use the Portfolio Analyzer & Optimizer.")
    else:
        # Fetch all data — use get_close_series() to ensure 1-D Series
        all_close_data = {}
        with st.spinner("Fetching portfolio data..."):
            for ticker_symbol in tickers:
                df_port = fetch_data(ticker_symbol, start_date, end_date)
                if df_port is not None and len(df_port) > 0:
                    all_close_data[ticker_symbol] = get_close_series(df_port)

        if len(all_close_data) < 2:
            st.warning("Need at least 2 stocks with valid data to build a portfolio.")
        else:
            # Build combined close price DataFrame
            close_df = pd.DataFrame(all_close_data).dropna()
            returns_df = close_df.pct_change().dropna()
            
            valid_tickers = list(close_df.columns)
            num_assets = len(valid_tickers)
            mean_returns = returns_df.mean()
            cov_matrix = returns_df.cov()

            # ─────────────────────────────────
            # Section 1: Portfolio Analyzer
            # ─────────────────────────────────
            st.subheader("📊 Portfolio Analyzer")
            st.markdown("Customize weights and analyze your portfolio's risk-return profile.")
            
            # ── Auto-adjusting weight sliders ──
            # Detect if stock selection changed → re-initialize equal weights
            current_tickers_hash = ",".join(sorted(valid_tickers))
            if st.session_state.get("_ptf_hash") != current_tickers_hash:
                st.session_state._ptf_hash = current_tickers_hash
                equal_w = round(1.0 / num_assets, 2)
                for tkr in valid_tickers:
                    st.session_state[f"weight_{tkr}"] = equal_w
                # Fix rounding so sum = 1.0
                actual = equal_w * num_assets
                if abs(actual - 1.0) > 0.001:
                    st.session_state[f"weight_{valid_tickers[0]}"] = round(
                        st.session_state[f"weight_{valid_tickers[0]}"] + (1.0 - actual), 2
                    )

            def redistribute_weights(changed_tkr):
                """When one weight changes, proportionally adjust others so sum = 1."""
                new_val = st.session_state[f"weight_{changed_tkr}"]
                remaining = max(0.0, 1.0 - new_val)
                other_tickers = [t for t in valid_tickers if t != changed_tkr]
                old_other_sum = sum(st.session_state.get(f"weight_{t}", 0) for t in other_tickers)

                if remaining <= 0:
                    for t in other_tickers:
                        st.session_state[f"weight_{t}"] = 0.0
                elif old_other_sum > 0:
                    scale = remaining / old_other_sum
                    for t in other_tickers:
                        st.session_state[f"weight_{t}"] = round(
                            max(0.0, st.session_state.get(f"weight_{t}", 0) * scale), 2
                        )
                    # Fix rounding error on last ticker
                    total_others = sum(st.session_state[f"weight_{t}"] for t in other_tickers)
                    diff = remaining - total_others
                    if abs(diff) > 0.001:
                        st.session_state[f"weight_{other_tickers[-1]}"] = round(
                            st.session_state[f"weight_{other_tickers[-1]}"] + diff, 2
                        )
                else:
                    equal_share = round(remaining / len(other_tickers), 2)
                    for t in other_tickers:
                        st.session_state[f"weight_{t}"] = equal_share
                    total_others = equal_share * len(other_tickers)
                    diff = remaining - total_others
                    if abs(diff) > 0.001:
                        st.session_state[f"weight_{other_tickers[-1]}"] = round(
                            st.session_state[f"weight_{other_tickers[-1]}"] + diff, 2
                        )

            st.markdown("##### ⚖️ Portfolio Weights")
            st.caption("Weights auto-adjust proportionally to always sum to 1.00")
            col_weights = st.columns(min(num_assets, 4))
            for i, tkr in enumerate(valid_tickers):
                with col_weights[i % len(col_weights)]:
                    st.slider(
                        f"{tkr}", 
                        min_value=0.0, max_value=1.0,
                        step=0.01, key=f"weight_{tkr}",
                        on_change=redistribute_weights,
                        args=(tkr,)
                    )
            
            # Read finalized weights from session state
            w_array = np.array([st.session_state.get(f"weight_{t}", 1.0/num_assets) for t in valid_tickers])
            weight_sum = w_array.sum()
            
            # Safety normalize (shouldn't be needed but just in case)
            if weight_sum > 0:
                w_array = w_array / weight_sum
            
            # Show sum confirmation
            st.success(f"✅ Weights sum to **{weight_sum:.2f}**")
            
            # Portfolio metrics
            p_return, p_vol, p_sharpe = portfolio_performance(w_array, mean_returns, cov_matrix)
            
            # Portfolio daily returns
            portfolio_daily_returns = (returns_df * w_array).sum(axis=1)
            cumulative_returns = (1 + portfolio_daily_returns).cumprod()
            p_max_dd = (cumulative_returns / cumulative_returns.cummax() - 1).min()

            # Metric cards  
            st.markdown("##### 📈 Portfolio Metrics")
            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1:
                st.markdown(f"""
                <div class="portfolio-metric-card">
                    <div class="portfolio-metric-value" style="color: {'#26a69a' if p_return > 0 else '#ef5350'};">{p_return:.2%}</div>
                    <div class="portfolio-metric-label">Expected Annual Return</div>
                </div>
                """, unsafe_allow_html=True)
            with mc2:
                st.markdown(f"""
                <div class="portfolio-metric-card">
                    <div class="portfolio-metric-value">{p_vol:.2%}</div>
                    <div class="portfolio-metric-label">Annual Volatility</div>
                </div>
                """, unsafe_allow_html=True)
            with mc3:
                color_sharpe = '#26a69a' if p_sharpe > 1 else ('#ff9800' if p_sharpe > 0 else '#ef5350')
                st.markdown(f"""
                <div class="portfolio-metric-card">
                    <div class="portfolio-metric-value" style="color: {color_sharpe};">{p_sharpe:.2f}</div>
                    <div class="portfolio-metric-label">Sharpe Ratio</div>
                </div>
                """, unsafe_allow_html=True)
            with mc4:
                st.markdown(f"""
                <div class="portfolio-metric-card">
                    <div class="portfolio-metric-value" style="color: #ef5350;">{p_max_dd:.2%}</div>
                    <div class="portfolio-metric-label">Max Drawdown</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("")  # spacing

            # ── Charts Row 1: Cumulative Returns + Correlation ──
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                st.markdown("##### 📈 Cumulative Returns")
                # Normalized price chart
                normalized = close_df / close_df.iloc[0] * 100
                fig_cum = go.Figure()
                
                colors_list = ['#26a69a', '#ef5350', '#ff9800', '#2962ff', '#ab47bc', 
                               '#00e676', '#ff6d00', '#42a5f5', '#ec407a', '#66bb6a']
                
                for i, col in enumerate(normalized.columns):
                    fig_cum.add_trace(go.Scatter(
                        x=normalized.index, y=normalized[col],
                        name=col, mode='lines',
                        line=dict(color=colors_list[i % len(colors_list)], width=2)
                    ))
                
                # Add portfolio line
                portfolio_norm = cumulative_returns * 100
                fig_cum.add_trace(go.Scatter(
                    x=portfolio_norm.index, y=portfolio_norm,
                    name='Portfolio', mode='lines',
                    line=dict(color='#ffffff', width=3, dash='dash')
                ))
                
                fig_cum.update_layout(
                    template="plotly_dark",
                    height=400,
                    margin=dict(l=10, r=10, t=10, b=10),
                    legend=dict(orientation="h", y=-0.15, bgcolor='rgba(0,0,0,0)'),
                    yaxis_title="Normalized Value (Base=100)",
                    hovermode='x unified'
                )
                fig_cum.update_xaxes(showgrid=True, gridcolor='#2a2e39')
                fig_cum.update_yaxes(showgrid=True, gridcolor='#2a2e39')
                st.plotly_chart(fig_cum, use_container_width=True)

            with chart_col2:
                st.markdown("##### 🔗 Correlation Matrix")
                corr_matrix = returns_df.corr()
                
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns.tolist(),
                    y=corr_matrix.index.tolist(),
                    colorscale=[
                        [0, '#ef5350'],      # Strong negative = red
                        [0.5, '#1e222d'],    # Zero = dark
                        [1, '#26a69a']       # Strong positive = green
                    ],
                    zmin=-1, zmax=1,
                    text=np.round(corr_matrix.values, 2),
                    texttemplate='%{text}',
                    textfont={"size": 12, "color": "#ffffff"},
                    hovertemplate='%{x} vs %{y}: %{z:.3f}<extra></extra>'
                ))
                
                fig_corr.update_layout(
                    template="plotly_dark",
                    height=400,
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig_corr, use_container_width=True)

            # ── Charts Row 2: Returns Distribution + Weight Allocation ──
            dist_col1, dist_col2 = st.columns(2)
            
            with dist_col1:
                st.markdown("##### 📊 Portfolio Returns Distribution")
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=portfolio_daily_returns,
                    nbinsx=50,
                    marker_color='#26a69a',
                    opacity=0.7,
                    name='Daily Returns'
                ))
                # Add VaR line (95%)
                var_95 = np.percentile(portfolio_daily_returns, 5)
                fig_dist.add_vline(
                    x=var_95, line_dash="dash", line_color="#ef5350",
                    annotation_text=f"VaR 95%: {var_95:.4f}", 
                    annotation_position="top left",
                    annotation_font_color="#ef5350"
                )
                fig_dist.update_layout(
                    template="plotly_dark",
                    height=350,
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis_title="Daily Return",
                    yaxis_title="Frequency",
                    showlegend=False
                )
                st.plotly_chart(fig_dist, use_container_width=True)

            with dist_col2:
                st.markdown("##### 🥧 Current Weight Allocation")
                fig_pie = go.Figure(data=[go.Pie(
                    labels=valid_tickers,
                    values=w_array,
                    hole=0.45,
                    marker=dict(colors=colors_list[:num_assets]),
                    textinfo='label+percent',
                    textfont=dict(color='#ffffff', size=12)
                )])
                fig_pie.update_layout(
                    template="plotly_dark",
                    height=350,
                    margin=dict(l=10, r=10, t=10, b=10),
                    showlegend=False
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown("---")

            # ─────────────────────────────────
            # Section 2: Portfolio Optimizer
            # ─────────────────────────────────
            st.subheader("🎯 Portfolio Optimizer — Efficient Frontier")
            st.markdown("Monte Carlo simulation with **10,000 random portfolios** to find optimal allocations using Modern Portfolio Theory (Markowitz).")

            with st.spinner("Running Monte Carlo simulation & optimization..."):
                num_portfolios = 10000
                results = np.zeros((3, num_portfolios))
                weights_record = []
                
                for i in range(num_portfolios):
                    w = np.random.dirichlet(np.ones(num_assets))
                    weights_record.append(w)
                    p_ret, p_std, p_sr = portfolio_performance(w, mean_returns, cov_matrix)
                    results[0, i] = p_std   # volatility
                    results[1, i] = p_ret   # return
                    results[2, i] = p_sr    # sharpe
                
                # Optimize: Max Sharpe
                opt_sharpe = optimize_portfolio(mean_returns, cov_matrix, num_assets, 'sharpe')
                opt_sharpe_weights = opt_sharpe.x
                opt_sharpe_ret, opt_sharpe_vol, opt_sharpe_sr = portfolio_performance(
                    opt_sharpe_weights, mean_returns, cov_matrix
                )
                
                # Optimize: Min Volatility
                opt_minvol = optimize_portfolio(mean_returns, cov_matrix, num_assets, 'min_vol')
                opt_minvol_weights = opt_minvol.x
                opt_minvol_ret, opt_minvol_vol, opt_minvol_sr = portfolio_performance(
                    opt_minvol_weights, mean_returns, cov_matrix
                )

            # ── Efficient Frontier Chart ──
            fig_ef = go.Figure()
            
            # All random portfolios as scatter
            fig_ef.add_trace(go.Scatter(
                x=results[0, :], y=results[1, :],
                mode='markers',
                marker=dict(
                    size=3,
                    color=results[2, :],
                    colorscale=[
                        [0, '#ef5350'],
                        [0.5, '#ff9800'],
                        [1, '#26a69a']
                    ],
                    colorbar=dict(title="Sharpe", tickformat=".2f"),
                    showscale=True,
                    opacity=0.6
                ),
                name='Random Portfolios',
                hovertemplate='Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
            ))
            
            # Max Sharpe point
            fig_ef.add_trace(go.Scatter(
                x=[opt_sharpe_vol], y=[opt_sharpe_ret],
                mode='markers+text',
                marker=dict(size=18, color='#26a69a', symbol='star', line=dict(width=2, color='#ffffff')),
                text=['Max Sharpe'], textposition='top center',
                textfont=dict(color='#26a69a', size=13, family='Arial Black'),
                name=f'Max Sharpe ({opt_sharpe_sr:.2f})',
                hovertemplate=f'Return: {opt_sharpe_ret:.2%}<br>Volatility: {opt_sharpe_vol:.2%}<br>Sharpe: {opt_sharpe_sr:.2f}<extra></extra>'
            ))
            
            # Min Volatility point
            fig_ef.add_trace(go.Scatter(
                x=[opt_minvol_vol], y=[opt_minvol_ret],
                mode='markers+text',
                marker=dict(size=18, color='#ff9800', symbol='diamond', line=dict(width=2, color='#ffffff')),
                text=['Min Vol'], textposition='top center',
                textfont=dict(color='#ff9800', size=13, family='Arial Black'),
                name=f'Min Volatility ({opt_minvol_vol:.2%})',
                hovertemplate=f'Return: {opt_minvol_ret:.2%}<br>Volatility: {opt_minvol_vol:.2%}<br>Sharpe: {opt_minvol_sr:.2f}<extra></extra>'
            ))
            
            # Current portfolio point
            fig_ef.add_trace(go.Scatter(
                x=[p_vol], y=[p_return],
                mode='markers+text',
                marker=dict(size=14, color='#ffffff', symbol='x', line=dict(width=2, color='#ffffff')),
                text=['Your Portfolio'], textposition='bottom center',
                textfont=dict(color='#ffffff', size=11),
                name=f'Your Portfolio',
                hovertemplate=f'Return: {p_return:.2%}<br>Volatility: {p_vol:.2%}<br>Sharpe: {p_sharpe:.2f}<extra></extra>'
            ))
            
            fig_ef.update_layout(
                template="plotly_dark",
                height=550,
                margin=dict(l=10, r=10, t=30, b=10),
                xaxis_title="Annual Volatility (Risk)",
                yaxis_title="Annual Expected Return",
                legend=dict(orientation="h", y=-0.15, bgcolor='rgba(0,0,0,0)'),
                hovermode='closest',
                xaxis=dict(tickformat='.0%', showgrid=True, gridcolor='#2a2e39'),
                yaxis=dict(tickformat='.0%', showgrid=True, gridcolor='#2a2e39'),
            )
            st.plotly_chart(fig_ef, use_container_width=True)

            # ── Optimal Weights Display ──
            st.markdown("##### 🏆 Optimal Portfolio Allocations")
            
            opt_col1, opt_col2 = st.columns(2)
            
            with opt_col1:
                st.markdown(f"""
                <div class="portfolio-metric-card" style="border: 1px solid rgba(38, 166, 154, 0.4);">
                    <div class="portfolio-metric-value" style="color: #26a69a;">⭐ Max Sharpe Portfolio</div>
                    <div class="portfolio-metric-label">Return: {opt_sharpe_ret:.2%} | Vol: {opt_sharpe_vol:.2%} | Sharpe: {opt_sharpe_sr:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Weights bar chart
                fig_w1 = go.Figure(go.Bar(
                    x=valid_tickers,
                    y=opt_sharpe_weights,
                    marker_color='#26a69a',
                    text=[f'{w:.1%}' for w in opt_sharpe_weights],
                    textposition='outside',
                    textfont=dict(color='#d1d4dc')
                ))
                fig_w1.update_layout(
                    template="plotly_dark",
                    height=280,
                    margin=dict(l=10, r=10, t=10, b=10),
                    yaxis=dict(tickformat='.0%', showgrid=True, gridcolor='#2a2e39'),
                    xaxis=dict(showgrid=False)
                )
                st.plotly_chart(fig_w1, use_container_width=True)

            with opt_col2:
                st.markdown(f"""
                <div class="portfolio-metric-card" style="border: 1px solid rgba(255, 152, 0, 0.4);">
                    <div class="portfolio-metric-value" style="color: #ff9800;">🛡️ Min Volatility Portfolio</div>
                    <div class="portfolio-metric-label">Return: {opt_minvol_ret:.2%} | Vol: {opt_minvol_vol:.2%} | Sharpe: {opt_minvol_sr:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
                
                fig_w2 = go.Figure(go.Bar(
                    x=valid_tickers,
                    y=opt_minvol_weights,
                    marker_color='#ff9800',
                    text=[f'{w:.1%}' for w in opt_minvol_weights],
                    textposition='outside',
                    textfont=dict(color='#d1d4dc')
                ))
                fig_w2.update_layout(
                    template="plotly_dark",
                    height=280,
                    margin=dict(l=10, r=10, t=10, b=10),
                    yaxis=dict(tickformat='.0%', showgrid=True, gridcolor='#2a2e39'),
                    xaxis=dict(showgrid=False)
                )
                st.plotly_chart(fig_w2, use_container_width=True)
            
            # ── Weights Comparison Table ──
            st.markdown("##### 📋 Weights Comparison Table")
            weights_comparison = pd.DataFrame({
                'Stock': valid_tickers,
                'Your Weights': [f'{w:.2%}' for w in w_array],
                'Max Sharpe': [f'{w:.2%}' for w in opt_sharpe_weights],
                'Min Volatility': [f'{w:.2%}' for w in opt_minvol_weights],
            }).set_index('Stock')
            
            st.dataframe(weights_comparison, use_container_width=True)
            
            # Performance comparison table
            st.markdown("##### 📊 Performance Comparison")
            perf_comparison = pd.DataFrame({
                'Portfolio': ['Your Portfolio', 'Max Sharpe', 'Min Volatility'],
                'Annual Return': [f'{p_return:.2%}', f'{opt_sharpe_ret:.2%}', f'{opt_minvol_ret:.2%}'],
                'Annual Volatility': [f'{p_vol:.2%}', f'{opt_sharpe_vol:.2%}', f'{opt_minvol_vol:.2%}'],
                'Sharpe Ratio': [f'{p_sharpe:.2f}', f'{opt_sharpe_sr:.2f}', f'{opt_minvol_sr:.2f}'],
            }).set_index('Portfolio')
            
            st.dataframe(perf_comparison, use_container_width=True)


# ==================================
# Floating Chatbot — Bottom-Right
# Chat panel: st.container with CSS marker (.finbot-marker-qf2026)
# Floating button: injected via components.html() JS
# Toggle: body.finbot-is-open class (persisted via localStorage)
# ==================================

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

with st.container(border=True):
    # Unique marker — CSS :has() selector targets ONLY this container
    st.markdown('<div class="finbot-marker-qf2026" style="display:none;"></div>', unsafe_allow_html=True)
    
    # Header
    st.markdown(
        "<div style='display:flex;align-items:center;gap:10px;padding-bottom:8px;border-bottom:1px solid rgba(255,255,255,0.06);margin-bottom:8px;'>"
        "<span style='font-size:1.2rem;'>🤖</span>"
        "<span style='font-weight:700;color:#ffffff;font-size:0.95rem;flex:1;'>FinBot AI</span>"
        "<span style='font-size:0.68rem;color:#787b86;'>LLaMA 3.3 · 70B</span>"
        "</div>",
        unsafe_allow_html=True
    )
    
    # Messages container — always rendered (visibility controlled by CSS/JS)
    msg_box = st.container(height=300)
    with msg_box:
        if not st.session_state.chat_messages:
            st.markdown(
                "<div style='text-align:center;padding:40px 16px;color:#787b86;'>"
                "<div style='font-size:2rem;margin-bottom:8px;'>💬</div>"
                "<div style='font-weight:600;color:#d1d4dc;margin-bottom:4px;'>Welcome to FinBot</div>"
                "<div style='font-size:0.82rem;'>Ask me anything about stocks, markets,<br>indicators, or portfolio strategies.</div>"
                "</div>",
                unsafe_allow_html=True
            )
        for msg in st.session_state.chat_messages:
            avatar = "🤖" if msg["role"] == "assistant" else "👤"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])
    
    # Input form
    with st.form("finbot_form", clear_on_submit=True, border=False):
        user_msg = st.text_input(
            "Message FinBot",
            placeholder="Ask about stocks, markets, portfolios...",
            label_visibility="collapsed"
        )
        send_col1, send_col2 = st.columns([4, 1])
        with send_col1:
            send = st.form_submit_button("Send →", use_container_width=True)
        with send_col2:
            if st.session_state.chat_messages:
                clear = st.form_submit_button("🗑️", use_container_width=True)
            else:
                clear = False
        
        if send and user_msg.strip():
            st.session_state.chat_messages.append({"role": "user", "content": user_msg.strip()})
            
            api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            if tickers:
                api_messages.append({
                    "role": "system",
                    "content": f"The user is currently analyzing these stocks: {', '.join(tickers)} from {start_date} to {end_date}."
                })
            for m in st.session_state.chat_messages[-10:]:
                api_messages.append({"role": m["role"], "content": m["content"]})
            
            response = get_groq_response(api_messages)
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
            st.rerun()
        
        if clear:
            st.session_state.chat_messages = []
            st.rerun()


# ── Floating Chat Widget (JS handles ALL positioning + toggle) ──
components.html("""
<script>
(function() {
    const doc = window.parent.document;

    // ── Find the chat container via our unique marker ──
    function findChatPanel() {
        const marker = doc.querySelector('.finbot-marker-qf2026');
        if (!marker) return null;
        // Walk up the DOM to find the outermost Streamlit wrapper
        let el = marker;
        while (el && el.parentElement) {
            el = el.parentElement;
            // Look for the bordered container wrapper OR a top-level block
            if (el.getAttribute('data-testid') === 'stVerticalBlockBorderWrapper' ||
                el.getAttribute('data-testid') === 'stVerticalBlock') {
                // Keep going up to get the outermost wrapper
                if (el.parentElement &&
                    el.parentElement.getAttribute('data-testid') === 'stVerticalBlockBorderWrapper') {
                    el = el.parentElement;
                }
                return el;
            }
        }
        // Fallback: go up a fixed number of levels from marker
        el = marker;
        for (let i = 0; i < 12; i++) {
            if (el.parentElement) el = el.parentElement;
            else break;
        }
        return el;
    }

    function init() {
        const panel = findChatPanel();
        if (!panel) {
            setTimeout(init, 300);
            return;
        }

        // ── Read saved state ──
        const isOpen = localStorage.getItem('finbot_open') === 'true';

        // ── Style the chat panel as a fixed floating widget ──
        panel.style.cssText = `
            position: fixed !important;
            bottom: 96px !important;
            right: 24px !important;
            width: 390px !important;
            z-index: 9999 !important;
            background: linear-gradient(180deg, #1a1e2e, #131722) !important;
            border-radius: 16px !important;
            border: 1px solid rgba(38,166,154,0.25) !important;
            box-shadow: 0 12px 48px rgba(0,0,0,0.5) !important;
            max-height: 70vh !important;
            overflow: auto !important;
            transition: opacity 0.3s ease, transform 0.3s ease !important;
            opacity: ${ isOpen ? '1' : '0' } !important;
            pointer-events: ${ isOpen ? 'auto' : 'none' } !important;
            transform: ${ isOpen ? 'translateY(0) scale(1)' : 'translateY(20px) scale(0.95)' } !important;
        `;

        // ── Style the Send button inside the form ──
        panel.querySelectorAll('[data-testid="stFormSubmitButton"] button').forEach(function(b) {
            b.style.cssText = `
                background: linear-gradient(135deg, #26a69a, #1e8c82) !important;
                color: #fff !important;
                border: none !important;
                border-radius: 10px !important;
                font-weight: 600 !important;
            `;
        });

        // ── Force white text on all chat message content ──
        panel.querySelectorAll('[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li, [data-testid="stChatMessage"] h1, [data-testid="stChatMessage"] h2, [data-testid="stChatMessage"] h3, [data-testid="stChatMessage"] h4, [data-testid="stChatMessage"] span, [data-testid="stChatMessage"] div, [data-testid="stMarkdownContainer"] p').forEach(function(el) {
            el.style.color = '#ffffff';
        });

        // ── Create (or reuse) floating bubble button ──
        let btn = doc.getElementById('finbot-float-btn');
        if (!btn) {
            btn = doc.createElement('div');
            btn.id = 'finbot-float-btn';
            doc.body.appendChild(btn);
        }

        btn.innerHTML = isOpen ? '&#x2715;' : '&#x1F4AC;';
        btn.style.cssText = `
            position: fixed;
            bottom: 24px;
            right: 24px;
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: ${ isOpen
                ? 'linear-gradient(135deg, #ef5350, #c62828)'
                : 'linear-gradient(135deg, #26a69a, #1e8c82)' };
            color: #fff;
            font-size: 1.5rem;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            z-index: 10000;
            box-shadow: ${ isOpen
                ? '0 6px 24px rgba(239,83,80,0.4)'
                : '0 6px 24px rgba(38,166,154,0.4)' };
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            user-select: none;
            -webkit-user-select: none;
        `;

        btn.onmouseenter = function() { this.style.transform = 'scale(1.1)'; };
        btn.onmouseleave = function() { this.style.transform = 'scale(1)'; };

        btn.onclick = function() {
            const wasOpen = localStorage.getItem('finbot_open') === 'true';
            const nowOpen = !wasOpen;
            localStorage.setItem('finbot_open', nowOpen);

            // Animate panel
            panel.style.opacity = nowOpen ? '1' : '0';
            panel.style.pointerEvents = nowOpen ? 'auto' : 'none';
            panel.style.transform = nowOpen
                ? 'translateY(0) scale(1)'
                : 'translateY(20px) scale(0.95)';

            // Update button
            this.innerHTML = nowOpen ? '&#x2715;' : '&#x1F4AC;';
            this.style.background = nowOpen
                ? 'linear-gradient(135deg, #ef5350, #c62828)'
                : 'linear-gradient(135deg, #26a69a, #1e8c82)';
            this.style.boxShadow = nowOpen
                ? '0 6px 24px rgba(239,83,80,0.4)'
                : '0 6px 24px rgba(38,166,154,0.4)';
        };
    }

    // Wait for Streamlit to finish rendering, then init
    setTimeout(init, 400);
})();
</script>
""", height=0)
