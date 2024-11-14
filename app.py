from flask import Flask, render_template, request, jsonify
import yfinance as yf
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
from flask_sqlalchemy import SQLAlchemy


# Initialize Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] =  'sqlite:///stock_analysis.db' 
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class StockAnalysis(db.Model):
    __tablename__ = 'stock_analysis'

    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(10), nullable=False)
    prediction = db.Column(db.Float, nullable=False)
    insight = db.Column(db.Text, nullable=False)
    date_analyzed = db.Column(db.DateTime, nullable=False, default=db.func.current_timestamp())

    def __repr__(self):
        return f"<StockAnalysis {self.ticker}>"
    


# Load the fine-tuned LLM model and tokenizer
model_dir = "/Users/trayvoniouspendleton/Documents/Stock Interface/fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)
model.to("cuda" if torch.cuda.is_available() else "cpu")

# Helper function to fetch stock data and prepare it
def fetch_and_prepare_stock_data(ticker):
    data = yf.download(ticker, period="1y")  # Fetch one year of data
    if data.empty:
        return None
    
    # Calculate moving averages and daily percentage change
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['Price_Change'] = data['Close'].pct_change() * 100  # Daily percentage change
    return data

# Plot 1: Closing Price Trend
def plot_closing_price(data, ticker):
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Close'], label='Closing Price')
    plt.title(f'{ticker} Stock Closing Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plot_path = os.path.join("static", "closing_price.png")
    plt.savefig(plot_path)
    plt.close()
    return plot_path

# Plot 2: Moving Average Comparison
def plot_moving_averages(data, ticker):
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Close'], label='Close')
    plt.plot(data.index, data['SMA_20'], label='SMA 20')
    plt.plot(data.index, data['EMA_12'], label='EMA 12')
    plt.title(f'{ticker} Moving Average Comparison')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plot_path = os.path.join("static", "moving_average_comparison.png")
    plt.savefig(plot_path)
    plt.close()
    return plot_path

# Plot 3: Daily Percentage Change Distribution
def plot_percentage_change(data, ticker):
    plt.figure(figsize=(8, 6))
    plt.hist(data['Price_Change'].dropna(), bins=50, color='blue')
    plt.title(f'{ticker} Daily Percentage Change Distribution')
    plt.xlabel('Percentage Change (%)')
    plt.ylabel('Frequency')
    plot_path = os.path.join("static", "percentage_change_distribution.png")
    plt.savefig(plot_path)
    plt.close()
    return plot_path

# Function to get LLM insight
def get_llm_insight(text):
    inputs = tokenizer(text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(inputs.input_ids, max_length=100)
    insight = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return insight

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    ticker = request.json.get("ticker", "").upper()  # Get the ticker symbol from the user input

    # Step 1: Fetch stock data and prepare it
    data = fetch_and_prepare_stock_data(ticker)
    if data is None:
        return jsonify({"error": "Stock data not available"}), 404  # Return error if data is not found

    # Step 2: Generate all three plots
    closing_price_plot = plot_closing_price(data, ticker)          # Closing Price Trend plot
    moving_average_plot = plot_moving_averages(data, ticker)       # Moving Averages Comparison plot (SMA vs EMA)
    percentage_change_plot = plot_percentage_change(data, ticker)  # Daily Percentage Change Distribution plot

    # Step 3: Example prediction (using a simple moving average for demonstration purposes)
    prediction = data['Close'].rolling(window=5).mean().iloc[-1]
    if isinstance(prediction, pd.Series):
        prediction = prediction.iloc[-1]  # Ensure it's a single float value
    prediction = float(prediction)  # Convert to float


    # Step 4: Generate LLM insight based on the ticker and prediction and save the analysis
    insight_text = f"Analyze the stock {ticker} with historical data and predicted closing price of {prediction:.2f}"
    insight = get_llm_insight(insight_text)

    # Save the analysis to the database
    analysis = StockAnalysis(ticker=ticker, prediction=prediction, insight=insight)
    db.session.add(analysis)
    db.session.commit()

    # Step 5: Return JSON response with plot URLs, prediction, and insight
    return jsonify({
        "closing_price_plot": closing_price_plot,            # URL of Closing Price plot
        "moving_average_plot": moving_average_plot,          # URL of Moving Averages plot
        "percentage_change_plot": percentage_change_plot,    # URL of Percentage Change Distribution plot
        "prediction": prediction,                            # Predicted stock price
        "insight": insight                                   # LLM-generated insight
    })

@app.route("/history", methods=["GET"])
def history():
    # Query the StockAnalysis table for all entries, ordered by the most recent analysis
    analyses = StockAnalysis.query.order_by(StockAnalysis.date_analyzed.desc()).all()
    
    # Format the query results into a JSON-compatible structure
    history_data = [
        {
            "ticker": analysis.ticker,
            "prediction": analysis.prediction,
            "insight": analysis.insight,
            "date_analyzed": analysis.date_analyzed.strftime("%Y-%m-%d %H:%M:%S")
        }
        for analysis in analyses
    ]
    
    return jsonify(history_data)

if __name__ == "__main__":
    app.run(debug=True)