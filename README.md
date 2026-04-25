# 📈 Stock Price Trend Predictor

A Machine Learning web app that predicts whether a stock price will go **UP or DOWN** the next trading day — built with Python, Scikit-learn, and Streamlit.

> 🔗 **Live Demo:** [your-app-name.streamlit.app](https://your-app-name.streamlit.app) *(update after deploying)*

---

## 🖥️ App Screenshot

> *(Add screenshot here after deploying)*

---

## 🚀 Features

- 🔮 **Next-day trend prediction** (UP / DOWN) with confidence score
- 🇮🇳 **Indian + Global stocks** — supports NSE stocks (`.NS` suffix) and all major global tickers
- 📊 **Interactive charts** — price history, moving averages, RSI indicator
- ⚡ **Live data** — fetches real-time data from Yahoo Finance
- 🤖 **3 ML models compared** — Logistic Regression, Random Forest, KNN

---

## 🧠 Features Used for Prediction

| Feature | Description |
|---|---|
| `Return` | Daily % price change |
| `MA5`, `MA10`, `MA20` | Moving averages (5, 10, 20 days) |
| `MA5_above_MA20` | Bullish crossover signal |
| `RSI` | Relative Strength Index (overbought/oversold) |
| `Volume_Change` | % change in trading volume |
| `Price_vs_MA20` | How far price is from its 20-day average |
| `Volatility` | 5-day rolling standard deviation of returns |
| `HL_Range` | Daily high-low range as % of close |

---

## 🗂️ Project Structure

```
stock_predictor/
│
├── data_prep.py        # Data fetching + feature engineering
├── train_model.py      # Model training, evaluation, saving
├── app.py              # Streamlit web app
├── requirements.txt    # Python dependencies
├── models/             # Saved model files (auto-created)
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── feature_cols.pkl
└── README.md
```

---

## ⚙️ How to Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/stock-trend-predictor.git
cd stock-trend-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare data & train the model
```bash
# Step 1: Fetch data + create features (saves stock_data_features.csv)
python data_prep.py

# Step 2: Train model + save to models/ folder
python train_model.py
```

### 4. Launch the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## ☁️ Deploy on Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Click **"New app"** → connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **Deploy** → get a public URL in minutes!

> **Note:** Before deploying, run `train_model.py` locally and commit the `models/` folder to GitHub.

---

## 📊 Model Performance

| Model | Test Accuracy |
|---|---|
| Logistic Regression | ~54% |
| Random Forest | ~56% |
| KNN | ~52% |

> Stock prediction is inherently noisy — even 55%+ accuracy is meaningful and profitable if acted consistently. The models outperform random guessing (50%).

---

## ⚠️ Disclaimer

This project is for **educational purposes only** and is not financial advice. Stock prices are influenced by many unpredictable factors. Never make real investment decisions based solely on ML predictions.

---

## 🛠️ Tech Stack

- **Data:** `yfinance`, `pandas`, `numpy`
- **ML:** `scikit-learn`
- **Visualization:** `plotly`, `matplotlib`
- **App:** `streamlit`
- **Deployment:** Streamlit Community Cloud

---

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)

---

⭐ If you found this useful, consider giving it a star!
