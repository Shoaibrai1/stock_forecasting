from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from datetime import datetime

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    companies = ["AAPL", "MSFT", "GOOG", "GOOGL", "META", "TSLA", "NVDA", "ADBE", "PYPL", "INTC", "CMCSA", "NFLX", "PEP"]
    return render_template("index.html", companies=companies)

@app.route("/forecast", methods=["POST"])
def forecast():
    ticker = request.form["ticker"]
    start_date = request.form["start"]
    end_date = request.form["end"]
    column = request.form["column"]
    forecast_days = int(request.form["forecast_days"])
    p, d, q = int(request.form["p"]), int(request.form["d"]), int(request.form["q"])
    seasonal_p = int(request.form["seasonal_p"])

    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)

    if column not in data.columns:
        return f"Column {column} not found."

    selected_data = data[["Date", column]].dropna()
    p_value = adfuller(selected_data[column])[1]
    is_stationary = p_value < 0.05

    decomposition = seasonal_decompose(selected_data[column], model='additive', period=12)

    model = sm.tsa.statespace.SARIMAX(selected_data[column],
                                      order=(p, d, q),
                                      seasonal_order=(p, d, q, seasonal_p))
    model_fit = model.fit()

    future = model_fit.get_prediction(start=len(selected_data),
                                      end=len(selected_data) + forecast_days - 1).predicted_mean
    future_dates = pd.date_range(start=pd.to_datetime(end_date), periods=forecast_days)
    forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": future})

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=selected_data['Date'], y=selected_data[column], name="Actual", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecast'], name="Forecast", line=dict(color="red")))
    graph_html = fig.to_html(full_html=False)

    return render_template("result.html",
                           ticker=ticker,
                           start=start_date,
                           end=end_date,
                           column=column,
                           graph_html=graph_html,
                           is_stationary=is_stationary,
                           summary=model_fit.summary().as_text())

# ✅ This part must be **outside** of all functions:
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)

