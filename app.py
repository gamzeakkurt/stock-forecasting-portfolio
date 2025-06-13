import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import plotly.graph_objs as go
from datetime import timedelta, date, datetime
import plotly.graph_objs as go
from prophet import Prophet
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import json
import os
import talib
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import requests
from bs4 import BeautifulSoup
import calendar
from dateutil.relativedelta import relativedelta
import time
from requests.exceptions import ReadTimeout
from scipy.stats import norm
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pmdarima import auto_arima
from requests.exceptions import HTTPError

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Enhanced sentiment analysis
def analyze_enhanced_sentiment(news_list):
    results = []
    for item in news_list:
        try:
            # Extract content from Yahoo Finance news structure
            content = item.get("content", {})
            title = content.get("title", "")
            summary = content.get("summary", "")
            publisher = content.get("provider", {}).get("displayName", "Unknown")
            link = content.get("canonicalUrl", {}).get("url", "")
            
            # Get publication date
            pub_date_str = content.get("pubDate")
            if pub_date_str:
                pub_date = pd.to_datetime(pub_date_str)
            else:
                display_time = content.get("displayTime")
                if display_time:
                    pub_date = pd.to_datetime(display_time)
                else:
                    continue

            # Combine title and summary for analysis
            text = f"{title}. {summary}"
            
            # Get sentiment scores
            sentiment = analyzer.polarity_scores(text)
            
            # Determine sentiment label
            if sentiment['compound'] >= 0.05:
                sentiment_label = "Positive"
            elif sentiment['compound'] <= -0.05:
                sentiment_label = "Negative"
            else:
                sentiment_label = "Neutral"

            results.append({
                "title": title,
                "summary": summary,
                "publisher": publisher,
                "link": link,
                "date": pub_date,
                "sentiment": sentiment_label,
                "compound_score": sentiment['compound'],
                "positive_score": sentiment['pos'],
                "negative_score": sentiment['neg'],
                "neutral_score": sentiment['neu']
            })
        except Exception as e:
            st.warning(f"Error analyzing sentiment for news item: {str(e)}")
            continue

    return results

# Function to calculate LSTM forecast
def lstm_forecast(data, days=30):
    try:
        # Ensure data is a pandas Series
        if isinstance(data, pd.Series):
            data_series = data.copy()
        else:
            data_series = pd.Series(data)
        
        # Scale the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data_series.values.reshape(-1, 1))
        
        # Prepare data for LSTM
        def create_dataset(dataset, look_back=1):
            X, Y = [], []
            for i in range(len(dataset)-look_back-1):
                a = dataset[i:(i+look_back), 0]
                X.append(a)
                Y.append(dataset[i + look_back, 0])
            return np.array(X), np.array(Y)
        
        look_back = 5
        X, Y = create_dataset(scaled_data, look_back)
        X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
        
        # Build LSTM model
        model = Sequential()
        model.add(LSTM(50, input_shape=(1, look_back)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        
        # Train model
        model.fit(X, Y, epochs=100, batch_size=1, verbose=0)
        
        # Make predictions
        last_sequence = scaled_data[-look_back:]
        predictions = []
        for _ in range(days):
            x_input = last_sequence.reshape(1, 1, look_back)
            yhat = model.predict(x_input, verbose=0)
            predictions.append(yhat[0,0])
            last_sequence = np.append(last_sequence[1:], yhat)
        
        # Inverse transform predictions
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        # Create forecast series
        forecast_index = pd.date_range(start=data_series.index[-1] + pd.Timedelta(days=1), periods=days)
        return pd.Series(predictions.flatten(), index=forecast_index)
    except Exception as e:
        st.error(f"LSTM forecast error: {str(e)}")
        return None

# --------------- App Title and Sidebar ---------------
st.set_page_config(page_title="Portfolio Tracker", layout="wide")
st.sidebar.title("Portfolio Tracker")

# Button to refresh (for future dynamic fetching)
if st.sidebar.button("Refresh Data"):
    st.rerun()

# --------------- Portfolio Setup ---------------
st.sidebar.subheader("Portfolio Composition")
portfolio = {
    # Tech Stocks
    'AAPL': st.sidebar.number_input("AAPL (Apple)", value=10),
    'MSFT': st.sidebar.number_input("MSFT (Microsoft)", value=0),
    'GOOGL': st.sidebar.number_input("GOOGL (Alphabet)", value=0),
    'NVDA': st.sidebar.number_input("NVDA (NVIDIA)", value=10),
    'META': st.sidebar.number_input("META (Meta)", value=0),
    'AMD': st.sidebar.number_input("AMD (Advanced Micro Devices)", value=0),
    'INTC': st.sidebar.number_input("INTC (Intel)", value=0),
    'QCOM': st.sidebar.number_input("QCOM (Qualcomm)", value=0),
    
    # Consumer Discretionary
    'AMZN': st.sidebar.number_input("AMZN (Amazon)", value=0),
    'TSLA': st.sidebar.number_input("TSLA (Tesla)", value=0),
    'NKE': st.sidebar.number_input("NKE (Nike)", value=0),
    
    # Financials
    'JPM': st.sidebar.number_input("JPM (JPMorgan Chase)", value=0),
    'BAC': st.sidebar.number_input("BAC (Bank of America)", value=0),
    'V': st.sidebar.number_input("V (Visa)", value=0),
    
    # Healthcare
    'JNJ': st.sidebar.number_input("JNJ (Johnson & Johnson)", value=0),
    'PFE': st.sidebar.number_input("PFE (Pfizer)", value=0),
    'UNH': st.sidebar.number_input("UNH (UnitedHealth)", value=0),
    
    # Industrials
    'BA': st.sidebar.number_input("BA (Boeing)", value=0),
    'CAT': st.sidebar.number_input("CAT (Caterpillar)", value=0),
    'HON': st.sidebar.number_input("HON (Honeywell)", value=0),
}

portfolio = {ticker: qty for ticker, qty in portfolio.items() if qty > 0}
tickers = list(portfolio.keys())

# --------------- Fetch Data ---------------

#st.write(tickers)
@st.cache_data

def fetch_data():
    while True:
        try:
            data = yf.download(tickers, start="2024-01-01", end="2024-12-31", auto_adjust=False)['Adj Close']
            return data
        except HTTPError as e:
            if 'Too Many Requests' in str(e):
                st.warning("Rate limit exceeded. Retrying in 60 seconds...")
                time.sleep(60)
            else:
                st.error(f"An HTTP error occurred: {str(e)}")
                return None
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            return None
        
data = fetch_data()
#st.write(data)
# Calculate portfolio value
portfolio_df = data * pd.Series(portfolio)
portfolio_value = portfolio_df.sum(axis=1).dropna()

if portfolio_value.empty:
    st.error("Portfolio value data is empty. Please check ticker symbols and quantities.")
    st.stop()

# --------------- User Features ---------------
st.sidebar.subheader("User Features")

# Custom Watchlists
st.sidebar.subheader("Custom Watchlists")
watchlist_name = st.sidebar.text_input("Create New Watchlist")
if watchlist_name:
    watchlist_tickers = st.sidebar.multiselect(
        "Select Tickers for Watchlist",
        options=tickers,
        key=f"watchlist_{watchlist_name}"
    )
    if st.sidebar.button("Save Watchlist"):
        watchlists = st.session_state.get('watchlists', {})
        watchlists[watchlist_name] = watchlist_tickers
        st.session_state['watchlists'] = watchlists
        st.sidebar.success(f"Watchlist '{watchlist_name}' saved!")

# Display existing watchlists
if 'watchlists' in st.session_state:
    st.sidebar.subheader("Your Watchlists")
    for name, tickers in st.session_state['watchlists'].items():
        with st.sidebar.expander(name):
            watchlist_data = data[tickers].tail(1)
            st.dataframe(watchlist_data)

# Price Alerts
st.sidebar.subheader("Price Alerts")
alert_ticker = st.sidebar.selectbox("Select Ticker", options=tickers)
alert_type = st.sidebar.selectbox("Alert Type", ["Above", "Below"])
alert_price = st.sidebar.number_input("Price Threshold", value=0.0)
if st.sidebar.button("Set Alert"):
    alerts = st.session_state.get('alerts', [])
    alerts.append({
        'ticker': alert_ticker,
        'type': alert_type,
        'price': alert_price,
        'created': datetime.now()
    })
    st.session_state['alerts'] = alerts
    st.sidebar.success("Alert set!")

# Display active alerts
if 'alerts' in st.session_state:
    st.sidebar.subheader("Active Alerts")
    for alert in st.session_state['alerts']:
        current_price = data[alert['ticker']].iloc[-1]
        triggered = (alert['type'] == "Above" and current_price > alert['price']) or \
                   (alert['type'] == "Below" and current_price < alert['price'])
        st.sidebar.write(f"{alert['ticker']}: {alert['type']} {alert['price']} - {'Triggered!' if triggered else 'Active'}")

# Portfolio Performance Tracking
st.subheader("Portfolio Performance Tracking")

# Performance metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Return", f"{((portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1) * 100):.2f}%")
with col2:
    st.metric("Daily Return", f"{portfolio_value.pct_change().iloc[-1] * 100:.2f}%")
with col3:
    st.metric("Volatility", f"{portfolio_value.pct_change().std() * np.sqrt(252) * 100:.2f}%")

# Performance chart
fig_performance = go.Figure()
fig_performance.add_trace(go.Scatter(
    x=portfolio_value.index,
    y=portfolio_value,
    mode='lines',
    name='Portfolio Value'
))
fig_performance.update_layout(
    title="Portfolio Performance",
    xaxis_title="Date",
    yaxis_title="Value (USD)",
    height=400
)
st.plotly_chart(fig_performance, use_container_width=True)

# Export Functionality
st.subheader("Export Reports")

# Create export options
export_format = st.selectbox("Select Export Format", ["CSV", "Excel", "PDF"])
export_content = st.multiselect(
    "Select Content to Export",
    options=["Portfolio Performance", "Forecast Results", "Risk Metrics", "Sector Analysis"]
)

if st.button("Generate Report"):
    # Create export directory if it doesn't exist
    if not os.path.exists("exports"):
        os.makedirs("exports")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if export_format == "CSV":
        # Export portfolio performance
        if "Portfolio Performance" in export_content:
            portfolio_value.to_csv(f"exports/portfolio_performance_{timestamp}.csv")
        
        # Export forecast results
        if "Forecast Results" in export_content and forecasts:
            forecast_df = pd.DataFrame(forecasts)
            forecast_df.to_csv(f"exports/forecast_results_{timestamp}.csv")
        
        # Export risk metrics
        if "Risk Metrics" in export_content:
            risk_metrics_df = pd.DataFrame([risk_metrics])
            risk_metrics_df.to_csv(f"exports/risk_metrics_{timestamp}.csv")
        
        # Export sector analysis
        if "Sector Analysis" in export_content:
            sector_df = pd.DataFrame(sector_weights.items(), columns=['Sector', 'Weight'])
            sector_df.to_csv(f"exports/sector_analysis_{timestamp}.csv")
    
    elif export_format == "Excel":
        # Create Excel writer
        with pd.ExcelWriter(f"exports/report_{timestamp}.xlsx") as writer:
            # Export portfolio performance
            if "Portfolio Performance" in export_content:
                portfolio_value.to_excel(writer, sheet_name="Portfolio Performance")
            
            # Export forecast results
            if "Forecast Results" in export_content and forecasts:
                forecast_df = pd.DataFrame(forecasts)
                forecast_df.to_excel(writer, sheet_name="Forecast Results")
            
            # Export risk metrics
            if "Risk Metrics" in export_content:
                risk_metrics_df = pd.DataFrame([risk_metrics])
                risk_metrics_df.to_excel(writer, sheet_name="Risk Metrics")
            
            # Export sector analysis
            if "Sector Analysis" in export_content:
                sector_df = pd.DataFrame(sector_weights.items(), columns=['Sector', 'Weight'])
                sector_df.to_excel(writer, sheet_name="Sector Analysis")
    
    st.success(f"Report exported successfully to the 'exports' directory!")

# Add alert checking functionality
def check_alerts():
    if 'alerts' in st.session_state:
        for alert in st.session_state['alerts']:
            current_price = data[alert['ticker']].iloc[-1]
            if (alert['type'] == "Above" and current_price > alert['price']) or \
               (alert['type'] == "Below" and current_price < alert['price']):
                st.warning(f"ALERT: {alert['ticker']} is {alert['type']} {alert['price']}!")

# Check alerts periodically
check_alerts()

# --------------- Forecast Controls ---------------
st.title("📈 Portfolio Value Forecasting")

# Model selection
forecasting_models = st.multiselect(
    "Select Forecasting Models",
    options=["ARIMA", "GARCH", "XGBoost", "LightGBM", "Ensemble", "Polynomial Regression", "LSTM"],
    default=["ARIMA", "GARCH", "XGBoost"]
)

# Forecast parameters
col1, col2, col3 = st.columns(3)
forecast_days = col1.slider("Forecast Period (Days)", 30, 365, 30, key="forecast_days_slider")
poly_degree = col2.slider("Polynomial Degree", 1, 5, 2, key="poly_degree_slider")
confidence_level = col3.slider("Confidence Level", 0.80, 0.99, 0.95, key="confidence_level_slider")

# Calculate polynomial regression forecast
X = np.arange(len(portfolio_value)).reshape(-1, 1)
y = portfolio_value.values

poly = PolynomialFeatures(degree=poly_degree)
X_poly = poly.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)
y_pred = model.predict(X_poly)

# Future forecast
X_future = np.arange(len(portfolio_value), len(portfolio_value) + forecast_days).reshape(-1, 1)
X_future_poly = poly.transform(X_future)
y_future = model.predict(X_future_poly)

# Create forecast series
future_dates = [portfolio_value.index[-1] + timedelta(days=i) for i in range(1, forecast_days + 1)]
y_future = pd.Series(y_future, index=future_dates)

def arima_forecast(data, days=30):
    try:
        # Ensure data is a pandas Series
        if isinstance(data, pd.Series):
            data_series = data.copy()
        else:
            data_series = pd.Series(data)
        
        # Remove NaN values
        data_series = data_series.dropna()
        if len(data_series) < 30:
            st.warning("Not enough data points for ARIMA forecasting")
            return None
        
        # Ensure datetime index
        if not isinstance(data_series.index, pd.DatetimeIndex):
            data_series.index = pd.date_range(start='2000-01-01', periods=len(data_series))
        
        # Take log of data to ensure positive values
        log_data = np.log(data_series)
        
        # Find optimal ARIMA model
        model = auto_arima(
            log_data
        )
        
        # Forecast
        forecast = model.predict(n_periods=days)
        
        # Convert back from log scale
        forecast = np.exp(forecast)
        
        # Create forecast index
        start_date = data_series.index[-1] + pd.Timedelta(days=1)
        forecast_index = pd.date_range(start=start_date, periods=days)
        
        # Build forecast series
        forecast_series = pd.Series(forecast, index=forecast_index)
        
        # Ensure values are reasonable
        last_value = data_series.iloc[-1]
        forecast_series = forecast_series.clip(lower=last_value * 0.5, upper=last_value * 1.5)
        
        # Debug information
        #st.write(f"ARIMA Model Order: {model.order}")
        #st.write(f"Model AIC: {model.aic():.2f}")
        #st.write(f"Model BIC: {model.bic():.2f}")
        #st.write(f"Forecast Shape: {forecast.shape}")
        
        # Format and display forecast results
        forecast_df = pd.DataFrame({
            'Date': forecast_series.index,
            'Forecast Value': forecast_series.values.round(2)
        })
        forecast_df['Forecast Value'] = forecast_df['Forecast Value'].apply(lambda x: f"${x:,.2f}")
        st.write("Forecast Results:")
        st.dataframe(forecast_df)
        
        return forecast_series
        
    except Exception as e:
        st.error(f"ARIMA forecast error: {str(e)}")
        return None

# Function to calculate GARCH forecast
def garch_forecast(data, days=30, confidence_level=0.95):
    try:
        # Ensure data is a pandas Series
        if isinstance(data, pd.Series):
            data_series = data.copy()
        else:
            data_series = pd.Series(data)
        
        # Check if we have enough data points
        if len(data_series) < 30:
            st.warning("Not enough data points for GARCH forecast. Need at least 30 points.")
            return None
            
        # Remove any NaN values
        data_series = data_series.dropna()
        
        # Calculate returns
        returns = data_series.pct_change().dropna()
        
        # Check if returns have enough variance
        if returns.var() < 1e-10:
            st.warning("Returns have too little variance for GARCH model.")
            return None
        
        # Fit GARCH model with more robust parameters
        model = arch_model(returns, vol='Garch', p=1, q=1, dist='normal')
        model_fit = model.fit(disp='off', show_warning=False)
        
        # Forecast volatility
        forecast = model_fit.forecast(horizon=days)
        
        # Calculate VaR using the forecasted volatility
        last_price = data_series.iloc[-1]
        forecast_volatility = np.sqrt(forecast.variance.iloc[-1].values)
        
        # Calculate forecasted prices using the last price and forecasted volatility
        forecast_prices = []
        current_price = last_price
        
        for vol in forecast_volatility:
            # Generate random return based on forecasted volatility
            random_return = np.random.normal(0, vol)
            current_price = current_price * (1 + random_return)
            forecast_prices.append(current_price)
        
        # Create forecast series
        forecast_index = pd.date_range(start=data_series.index[-1] + pd.Timedelta(days=1), periods=days)
        return pd.Series(forecast_prices, index=forecast_index)
    except Exception as e:
        st.error(f"GARCH forecast error: {str(e)}")
        return None

# Function to calculate XGBoost forecast
def xgboost_forecast(data, days=30):
    try:
        # Ensure data is a pandas Series
        if isinstance(data, pd.Series):
            data_series = data.copy()
        else:
            data_series = pd.Series(data)
        
        # Prepare data
        X = np.arange(len(data_series)).reshape(-1, 1)
        y = data_series.values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train model
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
        model.fit(X_train, y_train)
        
        # Make forecast
        future_X = np.arange(len(data_series), len(data_series) + days).reshape(-1, 1)
        forecast_values = model.predict(future_X)
        
        # Create forecast series with proper index
        forecast_index = pd.date_range(start=data_series.index[-1] + pd.Timedelta(days=1), periods=days)
        return pd.Series(forecast_values, index=forecast_index)
    except Exception as e:
        st.error(f"XGBoost forecast error: {str(e)}")
        return None

# Function to calculate LightGBM forecast
def lightgbm_forecast(data, days=30):
    try:
        # Ensure data is a pandas Series
        if isinstance(data, pd.Series):
            data_series = data.copy()
        else:
            data_series = pd.Series(data)
        
        # Prepare data
        X = np.arange(len(data_series)).reshape(-1, 1)
        y = data_series.values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Train model
        model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1)
        model.fit(X_train, y_train)
        
        # Make forecast
        future_X = np.arange(len(data_series), len(data_series) + days).reshape(-1, 1)
        forecast_values = model.predict(future_X)
        
        # Create forecast series with proper index
        forecast_index = pd.date_range(start=data_series.index[-1] + pd.Timedelta(days=1), periods=days)
        return pd.Series(forecast_values, index=forecast_index)
    except Exception as e:
        st.error(f"LightGBM forecast error: {str(e)}")
        return None

# Function to calculate ensemble forecast
def ensemble_forecast(data, days=30):
    try:
        # Ensure data is a pandas Series
        if isinstance(data, pd.Series):
            data_series = data.copy()
        else:
            data_series = pd.Series(data)
        
        # Get individual model forecasts
        arima_pred = arima_forecast(data_series, days)
        xgb_pred = xgboost_forecast(data_series, days)
        lgb_pred = lightgbm_forecast(data_series, days)
        
        # Check if we have at least one valid forecast
        valid_forecasts = [pred for pred in [arima_pred, xgb_pred, lgb_pred] if pred is not None]
        
        if not valid_forecasts:
            st.warning("No valid forecasts available for ensemble")
            return None
        
        # Combine forecasts with equal weights
        ensemble_pred = sum(valid_forecasts) / len(valid_forecasts)
        return ensemble_pred
        
    except Exception as e:
        st.error(f"Ensemble forecast error: {str(e)}")
        return None

# Create figure for forecasts
fig = go.Figure()

# Add historical data
fig.add_trace(go.Scatter(
    x=portfolio_value.index,
    y=portfolio_value.values,
    mode='lines',
    name='Historical Data',
    line=dict(color='blue')
))

# Calculate and plot selected forecasts
forecasts = {}
for model in forecasting_models:
    if model == "ARIMA":
        forecast = arima_forecast(portfolio_value, forecast_days)
    elif model == "GARCH":
        forecast = garch_forecast(portfolio_value, forecast_days, confidence_level)
    elif model == "XGBoost":
        forecast = xgboost_forecast(portfolio_value, forecast_days)
    elif model == "LightGBM":
        forecast = lightgbm_forecast(portfolio_value, forecast_days)
    elif model == "Ensemble":
        forecast = ensemble_forecast(portfolio_value, forecast_days)
    elif model == "Polynomial Regression":
        forecast = y_future  # Using existing polynomial forecast
    elif model == "LSTM":
        forecast = lstm_forecast(portfolio_value, forecast_days)
    
    if forecast is not None:
        forecasts[model] = forecast
fig.add_trace(go.Scatter(
            x=forecast.index,
            y=forecast.values,
    mode='lines',
            name=f'{model} Forecast',
            line=dict(dash='dash')
))

# Update layout
fig.update_layout(
    title="Advanced Forecasting Models",
    xaxis_title="Date",
    yaxis_title="Portfolio Value",
    height=600
)

# Display forecast plot
st.plotly_chart(fig, use_container_width=True)

# Display forecast metrics
if forecasts:
    st.write("### Forecast Metrics")
    
    # Calculate metrics for each forecast
    metrics = []
    for model, forecast in forecasts.items():
        last_actual = portfolio_value.iloc[-1]
        forecast_value = forecast.iloc[-1]
        change_pct = ((forecast_value - last_actual) / last_actual) * 100
        
        metrics.append({
            'Model': model,
            'Forecast Value': forecast_value,
            'Change (%)': change_pct
        })
    
    metrics_df = pd.DataFrame(metrics)
    st.dataframe(metrics_df.style.format({
        'Forecast Value': '${:,.2f}',
        'Change (%)': '{:.2f}%'
    }), use_container_width=True)

# Add model descriptions
with st.expander("Model Descriptions"):
    st.markdown("""
    **ARIMA (AutoRegressive Integrated Moving Average)**
    - Captures temporal dependencies in time series data
    - Good for short-term forecasting
    
    **GARCH (Generalized Autoregressive Conditional Heteroskedasticity)**
    - Models volatility clustering
    - Useful for risk management and VaR calculations
    
    **XGBoost/LightGBM**
    - Gradient boosting algorithms
    - Captures complex non-linear relationships
    - Good for medium to long-term forecasting
    
    **Ensemble Method**
    - Combines predictions from multiple models
    - Reduces individual model biases
    - Often provides more robust forecasts
    
    **Polynomial Regression**
    - Fits polynomial functions to historical data
    - Good for capturing trends
    
    **LSTM (Long Short-Term Memory)**
    - Deep learning model for sequence prediction
    - Excellent at capturing long-term dependencies
    """)
# --------------- Prepare Data for Prophet ---------------

# 📊 Step 2: Forecast function with Prophet
def forecast_with_prophet(df, days=30):
    df_prophet = pd.DataFrame({
        'ds': df.index,
        'y': df.values
    })
    model = Prophet()
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    return forecast[['ds', 'yhat']]

# 📈 Step 3: Generate forecast
forecast = forecast_with_prophet(portfolio_value, forecast_days)

# 📅 Step 4: Create Plotly figure
fig = go.Figure()

# Add historical data
fig.add_trace(go.Scatter(
    x=portfolio_value.index,
    y=portfolio_value.values,
    mode='lines',
    name='Historical Data',
    line=dict(color='blue')
))

# Add forecast line from Prophet
fig.add_trace(go.Scatter(
    x=forecast['ds'],
    y=forecast['yhat'],
    mode='lines',
    name='Forecast (Prophet)',
    line=dict(dash='dash', color='green')
))

# 📐 Step 5: Update layout
fig.update_layout(
    title="Portfolio Value Forecast",
    xaxis_title="Date",
    yaxis_title="Portfolio Value (USD)",
    height=500
)

# 📺 Step 6: Show it in Streamlit
st.plotly_chart(fig, use_container_width=True)

# --------------- Plot ---------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=portfolio_value.index, y=portfolio_value, mode='lines', name='Historical Data', line=dict(color='blue')))
future_dates = [portfolio_value.index[-1] + timedelta(days=i) for i in range(1, forecast_days + 1)]
fig.add_trace(go.Scatter(x=future_dates, y=y_future, mode='lines', name='Forecast', line=dict(dash='dash', color='red')))

fig.update_layout(
    title="Portfolio Value Forecast",
    xaxis_title="Date",
    yaxis_title="Portfolio Value (USD)",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# --------------- Summary Section ---------------
st.subheader("Forecast Summary")

current_value = portfolio_value.iloc[-1]
forecasted_value = y_future[-1]
growth_pct = ((forecasted_value - current_value) / current_value) * 100
daily_growth = np.mean(np.diff(y_pred[-forecast_days:]) / y_pred[-forecast_days:-1]) * 100
r2 = r2_score(y, y_pred)

col1, col2, col3 = st.columns(3)
col1.metric("Current Value", f"${current_value:,.2f}")
col2.metric("Forecasted Value", f"${forecasted_value:,.2f}", f"{growth_pct:.2f}%")
col3.metric("Daily Growth Rate", f"{daily_growth:.2f}%")

# --------------- Assumptions ---------------
with st.expander("Forecast Assumptions"):
    st.markdown("""
    - Using Polynomial Regression (Degree {}) for forecasting  
    - Forecast starts from the latest portfolio value  
    - Past performance does not guarantee future results  
    """.format(poly_degree))

# --------------- Model Info ---------------
st.subheader("Model Fit Information")
st.write(f"**R² Score:** {r2:.4f}")

#*-----------Risk Analysis------------------
# Clean ticker list to match what's actually in the data
available_tickers = [ticker for ticker in tickers if ticker in data.columns]

if not available_tickers:
    st.error("No valid tickers found in the data. Please check your input.")
else:
    # Calculate returns only for valid tickers
    returns = data[available_tickers].pct_change().dropna()

    # Proceed with risk analysis
    volatility = returns.std().sort_values(ascending=False)
    risk_threshold = st.slider("Risk Threshold (Volatility %)", 0.01, 0.10, 0.03)
    risky_tickers = volatility[volatility > risk_threshold]

    col1, col2 = st.columns(2)
    with col1:
        st.write("📈 **Ticker Volatility (%):**")
        st.dataframe((volatility * 100).round(2).to_frame(name="Volatility %"))

    with col2:
        st.write("⚠️ **Risky Tickers (Above Threshold):**")
        if not risky_tickers.empty:
            st.write(", ".join(risky_tickers.index))
        else:
            st.success("No risky tickers at this threshold 🚀")

# ---------------- Comparative Analysis ----------------

colors = {
    'AAPL': "#f2f23f", 'MSFT': "#92f005", 'GOOGL': "#f55c0a", 'NVDA': "#ff7f0e",
    'TSLA': "#1f77b4", 'AMZN': "#2ca02c", 'META': "#d62728", 'AMD': "#9467bd",
}

# Define the last 60 days of data
plot_data = data.last("60D")

# Create a Plotly figure
fig = go.Figure()

# Add lines for each ticker
for ticker in plot_data.columns:
    fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data[ticker],
                             mode='lines', name=ticker,
                             line=dict(color=colors.get(ticker, "#1f77b4"))))

# Highlight the last 15 days
highlight_start = plot_data.index.max() - pd.Timedelta(days=15)
fig.add_vrect(x0=highlight_start, x1=plot_data.index.max(), fillcolor="rgba(227, 170, 166, 0.5)", line_width=0)

# Annotate the highest value
max_value = plot_data.max().max()
max_ticker = plot_data.max().idxmax()
max_date = plot_data[max_ticker].idxmax()

fig.add_annotation(
    x=max_date, y=max_value,
    text=f"Highest: {max_value:.2f}\n({max_ticker})",
    showarrow=True, arrowhead=2,
    font=dict(size=12)
)

# Update layout to add titles and axis labels
fig.update_layout(
    title="Stock Prices (Adjusted Close)",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    legend_title="Ticker",
    height=600,
)

# Display the plot in Streamlit
st.subheader("Comparative Stock Performance (Last 60 Days)")
st.plotly_chart(fig)
#--------------Stock Movement------------

st.subheader("📊 Stock Momentum Summary")

# Dropdown to select lookback period
lookback_days = st.selectbox("Select Lookback Period", options=[5, 15, 30, 90], index=2)

# Ensure 'Date' column is in datetime format
data = data.reset_index()
data["Date"] = pd.to_datetime(data["Date"])

# Filter last N days
highlight_start = data["Date"].max() - pd.Timedelta(days=lookback_days)
highlight_data = data[data["Date"] >= highlight_start]

# Summary list
summary = []

# Loop through tickers
for ticker in tickers:
    df = highlight_data[["Date", ticker]].dropna()
    
    if not df.empty and len(df) > 1:
        first_price = df.iloc[0][ticker]
        last_price = df.iloc[-1][ticker]
        pct_change = ((last_price - first_price) / first_price) * 100
        summary.append({
            "Ticker": ticker,
            "Start Price": round(first_price, 2),
            "End Price": round(last_price, 2),
            "Change (%)": round(pct_change, 2)
        })

# Show DataFrame
summary_df = pd.DataFrame(summary)

if not summary_df.empty:
    summary_df = summary_df.sort_values(by="Change (%)", ascending=False)
    st.dataframe(summary_df, use_container_width=True)
else:
    st.info(f"⚠ No data available for the last {lookback_days} days.")

#-----------Enhanced News Analysis--------------
st.subheader("📰 Advanced News Analysis Dashboard")

# Main news analysis flow
st.subheader("📰 Recent News Analysis")

# Add news source filter
news_sources = st.multiselect(
    "Filter by News Source",
    options=["All", "Bloomberg", "Reuters", "CNBC", "MarketWatch", "Yahoo Finance"],
    default=["All"],
    key="news_sources_filter"
)

selected = st.selectbox("Select a ticker for news analysis", options=["All"] + tickers, key="news_ticker_select")

# Add a refresh button
if st.button("🔄 Refresh News Data", key="news_refresh_button"):
    st.cache_data.clear()

# Enhanced news fetching function
def fetch_enhanced_news(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.get_news()
        
        if not news:
            st.warning(f"No news available for {ticker}")
            return []
        
        # Process all news items
        processed_news = []
        for item in news:
            try:
                # Access the nested content
                content = item.get("content", {})
                
                # Get the publication date from the nested structure
                pub_date_str = content.get("pubDate")
                if pub_date_str:
                    pub_date = pd.to_datetime(pub_date_str)
                else:
                    # Try display time if pubDate is not available
                    display_time = content.get("displayTime")
                    if display_time:
                        pub_date = pd.to_datetime(display_time)
                    else:
                        continue
                
                # Extract the required fields from the nested structure
                news_item = {
                    "title": content.get("title", ""),
                    "summary": content.get("summary", ""),
                    "publisher": content.get("provider", {}).get("displayName", "Unknown"),
                    "link": content.get("canonicalUrl", {}).get("url", ""),
                    "date": pub_date
                }
                processed_news.append(news_item)
            except Exception as e:
                        st.warning(f"Error processing news item: {str(e)}")
                        continue
        
        # Sort news by date (most recent first) and take the 10 most recent
        processed_news.sort(key=lambda x: x["date"], reverse=True)
        recent_news = processed_news[:10]
        
        return recent_news
    except Exception as e:
        st.error(f"Error fetching news for {ticker}: {str(e)}")
        return []

# Enhanced sentiment analysis
def analyze_enhanced_sentiment(news_list):
    results = []

    for item in news_list:
        try:
            title = item.get("title", "")
            summary = item.get("summary", "")
            publisher = item.get("publisher", "Unknown")
            link = item.get("link", "")
            pub_date = item.get("date")

            # Analyze both title and summary
            title_sentiment = TextBlob(title).sentiment.polarity
            summary_sentiment = TextBlob(summary).sentiment.polarity if summary else 0

            # Calculate weighted sentiment
            weighted_sentiment = (title_sentiment * 0.7) + (summary_sentiment * 0.3)

            # Now correctly inside the try block
            results.append({
                "Title": title,
                "Summary": summary,
                "Publisher": publisher,
                "Published": pub_date.strftime('%Y-%m-%d %H:%M') if pub_date else "Unknown",
                "Title Sentiment": title_sentiment,
                "Summary Sentiment": summary_sentiment,
                "Weighted Sentiment": weighted_sentiment,
                "Link": link
            })

        except Exception as e:
            st.warning(f"Error analyzing sentiment for news item: {str(e)}")
            continue

    return results


# Main news analysis flow
st.subheader("📰 Recent News Analysis")

# Add news source filter
news_sources = st.multiselect(
    "Filter by News Source",
    options=["All", "Bloomberg", "Reuters", "CNBC", "MarketWatch", "Yahoo Finance"],
    default=["All"],
    key="news_sources_filter_v1"
)

selected = st.selectbox("Select a ticker for news analysis", options=["All"] + tickers, key="news_ticker_select_v")

# Add a refresh button
if st.button("🔄 Refresh News Data", key="news_refresh_button_2"):
    st.cache_data.clear()

all_news = []
if selected == "All":
    for ticker in tickers:
        news = fetch_enhanced_news(ticker)
        analyzed = analyze_enhanced_sentiment(news)
        for row in analyzed:
            row["Ticker"] = ticker
        all_news.extend(analyzed)
else:
    news = fetch_enhanced_news(selected)
    all_news = analyze_enhanced_sentiment(news)
    for row in all_news:
        row["Ticker"] = selected

# Display results
if all_news:
    df = pd.DataFrame(all_news)
    
    # Filter by news source if selected
    if "All" not in news_sources:
        df = df[df["Publisher"].isin(news_sources)]
    
    # Display news table
    st.subheader("📋 Recent News Articles")
    st.dataframe(df[["Ticker", "Published", "Title", "Publisher", "Weighted Sentiment"]], 
                 use_container_width=True)
    
    # Sentiment Analysis Charts
    st.subheader("📊 Sentiment Analysis")
    
    # Daily sentiment trend
    daily_sentiment = df.groupby("Published")["Weighted Sentiment"].mean().reset_index()
    fig_trend = px.line(daily_sentiment, x="Published", y="Weighted Sentiment",
                       title="Sentiment Trend")
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Sentiment distribution
    fig_dist = px.histogram(df, x="Weighted Sentiment", 
                          title="Sentiment Distribution",
                          nbins=20)
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Publisher analysis
    publisher_sentiment = df.groupby("Publisher")["Weighted Sentiment"].mean().reset_index()
    fig_pub = px.bar(publisher_sentiment, x="Publisher", y="Weighted Sentiment",
                    title="Average Sentiment by Publisher")
    st.plotly_chart(fig_pub, use_container_width=True)
    
    # Detailed article view
    st.subheader("📰 Article Details")
    selected_article = st.selectbox("Select an article to view details", 
                                  options=df["Title"].tolist(),
                                  key="article_details_select")
    article_details = df[df["Title"] == selected_article].iloc[0]
    
    st.markdown(f"""
    **Title:** {article_details['Title']}
    
    **Publisher:** {article_details['Publisher']}
    
    **Published:** {article_details['Published']}
    
    **Summary:** {article_details['Summary']}
    
    **Sentiment Scores:**
    - Title Sentiment: {article_details['Title Sentiment']:.2f}
    - Summary Sentiment: {article_details['Summary Sentiment']:.2f}
    - Weighted Sentiment: {article_details['Weighted Sentiment']:.2f}
    
    [Read Full Article]({article_details['Link']})
    """)
else:
    st.info("No news data available.")



# Technical Analysis Section
st.subheader("📊 Technical Analysis")

# Add technical indicators selector
tech_indicators = st.multiselect(
    "Select Technical Indicators",
    options=["RSI", "SMA", "EMA", "MACD", "Bollinger Bands"],
    default=["RSI", "Bollinger Bands"]
)

# Function to calculate technical indicators
def calculate_technical_indicators(data, indicators):
    results = {}
    
    for ticker in data.columns:
        ticker_data = data[ticker].dropna()
        
        if "RSI" in indicators:
            results[f"{ticker}_RSI"] = talib.RSI(ticker_data, timeperiod=14)
        
        if "SMA" in indicators:
            results[f"{ticker}_SMA_20"] = talib.SMA(ticker_data, timeperiod=20)
            results[f"{ticker}_SMA_50"] = talib.SMA(ticker_data, timeperiod=50)
        
        if "EMA" in indicators:
            results[f"{ticker}_EMA_20"] = talib.EMA(ticker_data, timeperiod=20)
            results[f"{ticker}_EMA_50"] = talib.EMA(ticker_data, timeperiod=50)
        
        if "MACD" in indicators:
            macd, signal, hist = talib.MACD(ticker_data)
            results[f"{ticker}_MACD"] = macd
            results[f"{ticker}_MACD_Signal"] = signal
            results[f"{ticker}_MACD_Hist"] = hist
        
        if "Bollinger Bands" in indicators:
            upper, middle, lower = talib.BBANDS(ticker_data, timeperiod=20)
            results[f"{ticker}_BB_Upper"] = upper
            results[f"{ticker}_BB_Middle"] = middle
            results[f"{ticker}_BB_Lower"] = lower
    
    return pd.DataFrame(results, index=data.index)

# Calculate and display technical indicators
if tech_indicators:
    tech_data = calculate_technical_indicators(data, tech_indicators)
    
    # Create subplots for each ticker
    for ticker in tickers:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.03, 
                           subplot_titles=(f"{ticker} Price", "Indicators"),
                           row_heights=[0.7, 0.3])
        
        # Add price line
        fig.add_trace(go.Scatter(x=data.index, y=data[ticker], 
                               name="Price", line=dict(color='blue')),
                     row=1, col=1)
        
        # Add indicators
        if "RSI" in tech_indicators:
            fig.add_trace(go.Scatter(x=data.index, y=tech_data[f"{ticker}_RSI"],
                                   name="RSI", line=dict(color='purple')),
                         row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        if "Bollinger Bands" in tech_indicators:
            fig.add_trace(go.Scatter(x=data.index, y=tech_data[f"{ticker}_BB_Upper"],
                                   name="BB Upper", line=dict(color='gray')),
                         row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=tech_data[f"{ticker}_BB_Middle"],
                                   name="BB Middle", line=dict(color='black')),
                         row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=tech_data[f"{ticker}_BB_Lower"],
                                   name="BB Lower", line=dict(color='gray')),
                         row=1, col=1)
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

# Portfolio Optimization Section
st.subheader("📈 Portfolio Optimization")

# Calculate returns and covariance matrix
# Convert data to numeric values and ensure proper index handling
numeric_data = data.copy()
if 'Date' in numeric_data.columns:
    numeric_data = numeric_data.set_index('Date')
numeric_data = numeric_data.apply(pd.to_numeric, errors='coerce')

# Filter out tickers with missing data
valid_tickers = numeric_data.columns[numeric_data.notna().all()]
if len(valid_tickers) < 2:
    st.error("Not enough valid tickers for portfolio optimization. Need at least 2 tickers with complete data.")
    st.stop()

numeric_data = numeric_data[valid_tickers]
returns = numeric_data.pct_change().dropna()

if len(returns) < 2:
    st.error("Not enough data points for portfolio optimization. Need at least 2 data points.")
    st.stop()

cov_matrix = returns.cov() * 252  # Annualized covariance
mean_returns = returns.mean() * 252  # Annualized returns

# Function to calculate portfolio statistics
def portfolio_stats(weights, mean_returns, cov_matrix):
    portfolio_return = np.sum(mean_returns * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility != 0 else 0
    return portfolio_return, portfolio_volatility, sharpe_ratio

# Function to minimize negative Sharpe ratio with diversification constraint
def min_sharpe_ratio(weights, mean_returns, cov_matrix):
    # Calculate portfolio statistics
    portfolio_return, portfolio_volatility, sharpe_ratio = portfolio_stats(weights, mean_returns, cov_matrix)
    
    # Add penalty for concentration (Herfindahl index)
    herfindahl = np.sum(weights ** 2)
    concentration_penalty = 10 * max(0, herfindahl - 0.3)  # Penalize if HHI > 0.3
    
    # Add penalty for high volatility
    volatility_penalty = 5 * max(0, portfolio_volatility - 0.3)  # Penalize if volatility > 30%
    
    return -(sharpe_ratio - concentration_penalty - volatility_penalty)

# Constraints and bounds
constraints = [
    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
    {'type': 'ineq', 'fun': lambda x: 0.3 - np.sum(x ** 2)},  # Maximum concentration (HHI <= 0.3)
    {'type': 'ineq', 'fun': lambda x: 0.3 - np.sqrt(np.dot(x.T, np.dot(cov_matrix, x)))}  # Maximum volatility <= 30%
]

# Set maximum weight per asset to 40%
bounds = tuple((0, 0.4) for _ in range(len(valid_tickers)))
initial_weights = np.array([1/len(valid_tickers)] * len(valid_tickers))

# Optimize portfolio
optimal_weights = minimize(min_sharpe_ratio, initial_weights,
                         args=(mean_returns, cov_matrix),
                         method='SLSQP',
                         bounds=bounds,
                         constraints=constraints)

# Display optimization results
if optimal_weights.success:
    optimized_weights = optimal_weights.x
    portfolio_return, portfolio_volatility, sharpe_ratio = portfolio_stats(optimized_weights, mean_returns, cov_matrix)
    
    # Calculate current portfolio value
    current_portfolio_value = sum(portfolio.get(ticker, 0) for ticker in valid_tickers)
    
    # Create DataFrame for weights
    weights_df = pd.DataFrame({
        'Ticker': valid_tickers,
        'Current Weight': [portfolio.get(ticker, 0) / current_portfolio_value if current_portfolio_value > 0 else 0 
                          for ticker in valid_tickers],
        'Optimized Weight': optimized_weights,
        'Expected Return': mean_returns.values,
        'Volatility': np.sqrt(np.diag(cov_matrix))
    })
    
    # Display weights comparison
    st.write("### Portfolio Weights Comparison")
    st.dataframe(weights_df.style.format({
        'Current Weight': '{:.2%}',
        'Optimized Weight': '{:.2%}',
        'Expected Return': '{:.2%}',
        'Volatility': '{:.2%}'
    }))
    
    # Display portfolio statistics
    col1, col2, col3 = st.columns(3)
    col1.metric("Expected Return", f"{portfolio_return:.2%}")
    col2.metric("Volatility", f"{portfolio_volatility:.2%}")
    col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    
    # Plot efficient frontier
    st.write("### Efficient Frontier")
    num_portfolios = 1000
    results = np.zeros((3, num_portfolios))
    
    for i in range(num_portfolios):
        weights = np.random.random(len(valid_tickers))
        weights /= np.sum(weights)
        portfolio_return, portfolio_volatility, sharpe_ratio = portfolio_stats(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_return
        results[1,i] = portfolio_volatility
        results[2,i] = sharpe_ratio
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results[1,:], y=results[0,:],
                            mode='markers',
                            marker=dict(color=results[2,:],
                                      colorscale='Viridis',
                                      showscale=True,
                                      colorbar=dict(title='Sharpe Ratio')),
                            name='Random Portfolios'))
    
    # Add optimal portfolio
    fig.add_trace(go.Scatter(x=[portfolio_volatility], y=[portfolio_return],
                            mode='markers',
                            marker=dict(color='red', size=10),
                            name='Optimal Portfolio'))
    
    fig.update_layout(
        title='Efficient Frontier',
        xaxis_title='Volatility',
        yaxis_title='Expected Return',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# Risk Management Section
st.subheader("📊 Risk Management Analysis")

# Value at Risk (VaR) Calculation
def calculate_var(returns, confidence_level=0.95):
    """Calculate Value at Risk (VaR)"""
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    if len(returns) == 0:
        return 0.0
    
    # Sort returns and find the appropriate percentile
    sorted_returns = np.sort(returns)
    index = int((1 - confidence_level) * len(sorted_returns))
    return sorted_returns[index] if index < len(sorted_returns) else sorted_returns[-1]

# Maximum Drawdown Calculation
def calculate_max_drawdown(returns):
    """Calculate Maximum Drawdown"""
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    if len(returns) == 0:
        return 0.0
    
    # Calculate cumulative returns
    cumulative = np.cumprod(1 + returns)
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(cumulative)
    
    # Calculate drawdown
    drawdown = (cumulative - running_max) / running_max
    
    return np.min(drawdown) if len(drawdown) > 0 else 0.0

# Stop-loss Recommendation
def calculate_stop_loss(returns, confidence_level=0.95):
    """Calculate Stop-loss level based on historical volatility"""
    if isinstance(returns, pd.Series):
        returns = returns.values
    volatility = np.std(returns)
    stop_loss = -volatility * 2  # 2 standard deviations
    return stop_loss

# Risk-adjusted Performance Metrics
def calculate_risk_metrics(returns):
    """Calculate various risk-adjusted performance metrics"""
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    if len(returns) < 2:  # Need at least 2 points for meaningful calculations
        return {
            'Annual Return': 0.0,
            'Annual Volatility': 0.0,
            'Sharpe Ratio': 0.0,
            'Sortino Ratio': 0.0,
            'Calmar Ratio': 0.0
        }
    
    # Daily risk-free rate (assuming 2% annual)
    daily_rf = (1.02) ** (1/252) - 1
    
    # Calculate basic statistics
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    # Annualized return and volatility
    annual_return = (1 + mean_return) ** 252 - 1
    annual_volatility = std_return * np.sqrt(252)
    
    # Sharpe Ratio
    if annual_volatility > 0:
        sharpe_ratio = (annual_return - 0.02) / annual_volatility
    else:
        sharpe_ratio = 0.0
    
    # Sortino Ratio
    downside_returns = returns[returns < daily_rf]
    if len(downside_returns) > 1:
        downside_std = np.std(downside_returns) * np.sqrt(252)
        if downside_std > 0:
            sortino_ratio = (annual_return - 0.02) / downside_std
        else:
            sortino_ratio = 0.0
    else:
        sortino_ratio = 0.0
    
    # Calmar Ratio
    max_drawdown = calculate_max_drawdown(returns)
    if max_drawdown < 0:  # Only calculate if we have a drawdown
        calmar_ratio = annual_return / abs(max_drawdown)
    else:
        calmar_ratio = 0.0
    
    # Ensure ratios are reasonable
    sharpe_ratio = np.clip(sharpe_ratio, -5, 5)
    sortino_ratio = np.clip(sortino_ratio, -5, 5)
    calmar_ratio = np.clip(calmar_ratio, -5, 5)
    
    return {
        'Annual Return': annual_return,
        'Annual Volatility': annual_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Calmar Ratio': calmar_ratio
    }

# Calculate portfolio returns
# First ensure data is numeric and handle the index properly
numeric_data = data.copy()
if 'Date' in numeric_data.columns:
    numeric_data = numeric_data.set_index('Date')
numeric_data = numeric_data.apply(pd.to_numeric, errors='coerce')

# Calculate portfolio returns
if len(numeric_data.columns) > 0:
    # Calculate daily returns for each stock
    daily_returns = numeric_data.pct_change().dropna()
    
    if len(daily_returns) > 0:
        # Calculate portfolio weights based on current holdings
        current_values = {ticker: numeric_data[ticker].iloc[-1] * portfolio.get(ticker, 0) 
                         for ticker in numeric_data.columns}
        total_value = sum(current_values.values())
        
        if total_value > 0:
            # Calculate weights
            weights = {ticker: value/total_value for ticker, value in current_values.items()}
            
            # Calculate weighted portfolio returns
            portfolio_returns = pd.Series(index=daily_returns.index)
            for date in daily_returns.index:
                portfolio_returns[date] = sum(weights[ticker] * daily_returns.loc[date, ticker] 
                                            for ticker in weights.keys() 
                                            if ticker in daily_returns.columns)
            
            portfolio_returns = portfolio_returns.dropna()
            
            if len(portfolio_returns) > 0:
                # Calculate metrics
                try:
                    risk_metrics = calculate_risk_metrics(portfolio_returns)
                    var_95 = calculate_var(portfolio_returns, 0.95)
                    var_99 = calculate_var(portfolio_returns, 0.99)
                    max_dd = calculate_max_drawdown(portfolio_returns)
                    stop_loss = calculate_stop_loss(portfolio_returns)
                    
                    # Display Risk Metrics
                    st.write("### Portfolio Risk Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Volatility", f"{risk_metrics['Annual Volatility']:.2%}")
                    col2.metric("Maximum Drawdown", f"{max_dd:.2%}")
                    col3.metric("Value at Risk (95%)", f"{var_95:.2%}")
                    col4.metric("Value at Risk (99%)", f"{var_99:.2%}")
                    
                    # Display Risk-adjusted Performance
                    st.write("### Risk-adjusted Performance")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Sharpe Ratio", f"{risk_metrics['Sharpe Ratio']:.2f}")
                    col2.metric("Sortino Ratio", f"{risk_metrics['Sortino Ratio']:.2f}")
                    col3.metric("Calmar Ratio", f"{risk_metrics['Calmar Ratio']:.2f}")
                    
                    # Add explanation
                    st.info("""
                    Risk-adjusted ratios help evaluate investment performance relative to risk:
                    - **Sharpe Ratio**: Measures excess return per unit of total risk (higher is better)
                    - **Sortino Ratio**: Measures excess return per unit of downside risk (higher is better)
                    - **Calmar Ratio**: Measures return relative to maximum drawdown (higher is better)
                    
                    A ratio of 0.0 indicates insufficient data or no risk-adjusted return.
                    """)
                    
                    # Stop-loss Recommendations
                    st.write("### Stop-loss Recommendations")
                    st.write(f"Based on historical volatility, recommended stop-loss level: {stop_loss:.2%}")
                    st.info("""
                    Stop-loss recommendations are based on historical volatility and are meant as a guide only. 
                    Consider adjusting based on your risk tolerance and market conditions.
                    """)
                    
                    # Risk Visualization
                    st.write("### Risk Analysis Visualization")
                    
                    # Create subplots
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                      vertical_spacing=0.03, 
                                      subplot_titles=("Portfolio Value", "Drawdown"))
                    
                    # Calculate portfolio cumulative returns
                    portfolio_cumulative = (1 + portfolio_returns).cumprod()
                    
                    # Portfolio Value
                    fig.add_trace(go.Scatter(x=portfolio_returns.index, 
                                          y=portfolio_cumulative,
                                          name="Portfolio Value",
                                          line=dict(color='blue')),
                               row=1, col=1)
                    
                    # Drawdown
                    running_max = np.maximum.accumulate(portfolio_cumulative.values)
                    drawdown_series = (portfolio_cumulative.values - running_max) / running_max
                    
                    fig.add_trace(go.Scatter(x=portfolio_returns.index,
                                          y=drawdown_series,
                                          name="Drawdown",
                                          line=dict(color='red')),
                               row=2, col=1)
                    
                    # Add VaR lines
                    fig.add_hline(y=1 + var_95, line_dash="dash", line_color="orange",
                               annotation_text="95% VaR", row=1, col=1)
                    fig.add_hline(y=1 + var_99, line_dash="dash", line_color="red",
                               annotation_text="99% VaR", row=1, col=1)
                    
                    # Update layout
                    fig.update_layout(
                        height=800,
                        showlegend=True,
                        xaxis_title="Date",
                        yaxis_title="Value",
                        yaxis2_title="Drawdown",
                        title="Portfolio Risk Analysis"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error calculating risk metrics: {str(e)}")
            else:
                st.warning("No valid returns data available for risk analysis")
        else:
            st.warning("Portfolio value is zero. Please check your holdings.")
    else:
        st.warning("No valid daily returns data available")
else:
    st.error("No valid data available for risk analysis")

# Individual Stock Risk Analysis
st.write("### Individual Stock Risk Analysis")

# Calculate risk metrics for each stock
stock_risk_metrics = {}
for ticker in tickers:
    if ticker in numeric_data.columns:
        # Get stock returns
        stock_returns = numeric_data[ticker].pct_change().dropna()
        if not stock_returns.empty:
            # Calculate metrics
            volatility = stock_returns.std() * np.sqrt(252)  # Annualized volatility
            max_drawdown = calculate_max_drawdown(stock_returns)
            var_95 = calculate_var(stock_returns, 0.95)
            
            # Calculate beta
            common_dates = stock_returns.index.intersection(portfolio_returns.index)
            if len(common_dates) > 0:
                aligned_stock_returns = stock_returns[common_dates]
                aligned_portfolio_returns = portfolio_returns[common_dates]
                
                if len(aligned_stock_returns) > 1 and len(aligned_portfolio_returns) > 1:
                    covariance = np.cov(aligned_stock_returns, aligned_portfolio_returns)[0,1]
                    portfolio_variance = np.var(aligned_portfolio_returns)
                    beta = covariance / portfolio_variance if portfolio_variance != 0 else 0
    else:
                    beta = 0
                    stock_risk_metrics[ticker] = {
                    'Volatility': volatility,
                    'Max Drawdown': max_drawdown,
                    'VaR (95%)': var_95,
                    'Beta': beta
                }

# Display stock risk metrics
if stock_risk_metrics:
    risk_df = pd.DataFrame(stock_risk_metrics).T
    st.dataframe(risk_df.style.format({
        'Volatility': '{:.2%}',
        'Max Drawdown': '{:.2%}',
        'VaR (95%)': '{:.2%}',
        'Beta': '{:.2f}'
    }), use_container_width=True)
    
    # Add interpretation
    st.info("""
    Risk Metrics Interpretation:
    - **Volatility**: Annualized standard deviation of returns (higher = more risk)
    - **Max Drawdown**: Largest peak-to-trough decline (more negative = higher risk)
    - **VaR (95%)**: Maximum expected loss at 95% confidence level
    - **Beta**: Sensitivity to portfolio movements (higher = more correlated)
    """)
    
    # Add summary statistics
    st.write("### Risk Metrics Summary")
    summary_df = pd.DataFrame({
        'Metric': ['Volatility', 'Max Drawdown', 'VaR (95%)', 'Beta'],
        'Average': [
            risk_df['Volatility'].mean(),
            risk_df['Max Drawdown'].mean(),
            risk_df['VaR (95%)'].mean(),
            risk_df['Beta'].mean()
        ],
        'Maximum': [
            risk_df['Volatility'].max(),
            risk_df['Max Drawdown'].max(),
            risk_df['VaR (95%)'].max(),
            risk_df['Beta'].max()
        ],
        'Minimum': [
            risk_df['Volatility'].min(),
            risk_df['Max Drawdown'].min(),
            risk_df['VaR (95%)'].min(),
            risk_df['Beta'].min()
        ]
    })
    st.dataframe(summary_df.style.format({
        'Average': '{:.2%}',
        'Maximum': '{:.2%}',
        'Minimum': '{:.2%}'
    }), use_container_width=True)
else:
    st.warning("No valid risk metrics available for the selected stocks.")

# Market Analysis Section
st.subheader("📊 Market Analysis")

# Market Trends Analysis
st.write("### Market Trends")
market_indexes = ['^GSPC', '^DJI', '^IXIC']  # S&P 500, Dow Jones, NASDAQ

# Use existing data if available, otherwise fetch new data
if all(index in data.columns for index in market_indexes):
    market_data = data[market_indexes]
else:
    market_data = yf.download(market_indexes, start="2022-01-01")['Close']

# Calculate returns
market_returns = market_data.pct_change().dropna()
market_cumulative = (1 + market_returns).cumprod()

# Plot market performance
fig_market = go.Figure()
for index in market_indexes:
    fig_market.add_trace(go.Scatter(
        x=market_cumulative.index,
        y=market_cumulative[index],
        name=index.replace('^', ''),
        mode='lines'
    ))

fig_market.update_layout(
    title="Major Market Index Performance",
    xaxis_title="Date",
    yaxis_title="Cumulative Return",
    height=400
)
st.plotly_chart(fig_market, use_container_width=True)

# Sector Performance Analysis
st.write("### Sector Performance")
sector_etfs = {
    'Technology': 'XLK',
    'Financial Services': 'XLF',
    'Consumer Cyclical': 'XLY',
    'Healthcare': 'XLV',
    'Communication Services': 'XLC',
    'Industrials': 'XLI',
    'Consumer Defensive': 'XLP',
    'Energy': 'XLE',
    'Real Estate': 'XLRE',
    'Basic Materials': 'XLB',
    'Utilities': 'XLU'
}

try:
    # Fetch sector ETF data for the current year
    today = datetime.today()
    start_date = datetime(today.year, 1, 1)
    sector_data = yf.download(list(sector_etfs.values()), start=start_date, end=today)['Close']
    
    # Calculate YTD returns
    first_prices = sector_data.iloc[0]
    last_prices = sector_data.iloc[-1]
    ytd_returns = ((last_prices - first_prices) / first_prices) * 100
    
    # Calculate volatility
    sector_returns = sector_data.pct_change().dropna()
    volatility = sector_returns.std() * np.sqrt(252) * 100
    
    # Create sector performance table
    sector_performance = pd.DataFrame({
        'Sector': list(sector_etfs.keys()),
        'YTD Return (%)': ytd_returns.values,
        'Volatility (%)': volatility.values
    })
    
    # Sort by YTD return
    sector_performance = sector_performance.sort_values('YTD Return (%)', ascending=False)
    
    # Display sector performance
    st.dataframe(sector_performance.style.format({
        'YTD Return (%)': '{:.2f}%',
        'Volatility (%)': '{:.2f}%'
    }), use_container_width=True)
    
    # Add a note about the data source
    st.info("""
    Note: Sector performance is calculated using SPDR sector ETFs (XL* series) which track the S&P 500 sectors.
    Returns are calculated from the first trading day of the current year to the most recent price.
    """)
    
except Exception as e:
    st.error(f"Error calculating sector performance: {str(e)}")
    st.warning("Please try refreshing the data or check your internet connection.")

# Market Breadth Indicators
st.write("### Market Breadth Indicators")

# Calculate advance-decline line
def calculate_advance_decline(data):
    # Ensure we're working with numeric data
    numeric_data = data.select_dtypes(include=[np.number])
    if numeric_data.empty:
        return pd.Series()
    
    # Calculate daily returns
    daily_returns = numeric_data.pct_change()
    
    # Count advances and declines
    advances = (daily_returns > 0).sum(axis=1)
    declines = (daily_returns < 0).sum(axis=1)
    
    # Calculate net advances
    net_advances = advances - declines
    
    return net_advances

# Calculate market breadth
market_breadth = calculate_advance_decline(data)
if not market_breadth.empty:
    market_breadth_cumulative = market_breadth.cumsum()
    
    # Plot market breadth
    fig_breadth = go.Figure()
    fig_breadth.add_trace(go.Scatter(
        x=market_breadth_cumulative.index,
        y=market_breadth_cumulative,
        name="Advance-Decline Line",
        line=dict(color='blue')
    ))
    
    fig_breadth.update_layout(
        title="Market Breadth (Advance-Decline Line)",
        xaxis_title="Date",
        yaxis_title="Cumulative Advances - Declines",
        height=400
    )
    st.plotly_chart(fig_breadth, use_container_width=True)
else:
    st.warning("Insufficient data to calculate market breadth indicators.")

# Market Sentiment Analysis
st.write("### Market Sentiment")
sentiment_indicators = {
    'VIX': '^VIX',  # Volatility Index
    'Put/Call Ratio': '^CPC'  # CBOE Put/Call Ratio
}

# Fetch sentiment data
sentiment_data = yf.download(list(sentiment_indicators.values()), start="2022-01-01")['Close']

# Plot sentiment indicators
fig_sentiment = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=("VIX (Volatility Index)", "Put/Call Ratio"))

fig_sentiment.add_trace(
    go.Scatter(x=sentiment_data.index, y=sentiment_data['^VIX'],
              name="VIX", line=dict(color='red')),
    row=1, col=1
)

fig_sentiment.add_trace(
    go.Scatter(x=sentiment_data.index, y=sentiment_data['^CPC'],
              name="Put/Call Ratio", line=dict(color='green')),
    row=2, col=1
)

fig_sentiment.update_layout(
    height=600,
    showlegend=True,
    title="Market Sentiment Indicators"
)

st.plotly_chart(fig_sentiment, use_container_width=True)

# Add interpretation
st.info("""
Market Analysis Interpretation:
- **Market Trends**: Shows performance of major market indices
- **Sector Performance**: Displays YTD returns and volatility for different sectors
- **Market Breadth**: Advance-Decline line indicates market participation
- **Market Sentiment**: 
  - VIX: Higher values indicate increased market fear
  - Put/Call Ratio: Higher values suggest bearish sentiment
""") 
