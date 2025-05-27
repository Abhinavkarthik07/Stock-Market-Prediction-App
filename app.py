import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

# -------------------- Data Loader --------------------
def load_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            return None
        data.reset_index(inplace=True)
        return data
    except Exception:
        return None

# -------------------- Preprocessing --------------------
def prepare_lstm_data(data, time_step=60):
    if len(data) < time_step:
        return np.array([]), np.array([]), None
    close_prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_prices)

    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i - time_step:i, 0])
        y.append(scaled_data[i, 0])
    X = np.array(X)
    y = np.array(y)

    return X.reshape(X.shape[0], X.shape[1], 1), y, scaler

# -------------------- LSTM Model --------------------
def train_lstm(X, y, epochs=10):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)
    return model

# -------------------- Future Prediction --------------------
def predict_lstm(model, data, time_step, scaler, days=30):
    last_sequence = data['Close'].values[-time_step:].reshape(-1, 1)
    input_seq = scaler.transform(last_sequence).flatten().tolist()
    predictions = []

    for _ in range(days):
        x_input = np.array(input_seq[-time_step:]).reshape(1, time_step, 1)
        pred = model.predict(x_input, verbose=0)[0][0]
        predictions.append(pred)
        input_seq.append(pred)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions

# -------------------- Streamlit App --------------------
st.title("📈 Stock Market Price Predictor using LSTM")

ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, MSFT, INFY.NS):", "AAPL").upper()
start_date = st.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2024-01-01"))

if st.button("Predict"):
    st.subheader(f"Fetching data for: `{ticker}`")
    data = load_data(ticker, start_date, end_date)

    if data is None or len(data) < 100:
        st.error("❌ Error: Invalid ticker or not enough data. Please try another stock or date range.")
        st.stop()

    st.success(f"✅ Loaded {len(data)} rows of data")
    st.write(data.tail())

    X, y, scaler = prepare_lstm_data(data)
    if X.size == 0:
        st.error("❌ Not enough data after preprocessing. Choose a longer date range.")
        st.stop()

    st.subheader("🧠 Training LSTM Model...")
    model = train_lstm(X, y)

    st.subheader("📊 Predicting Next 30 Days...")
    predictions = predict_lstm(model, data, time_step=60, scaler=scaler, days=30)

    future_dates = pd.date_range(data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=30)
    pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': predictions.flatten()})

    st.line_chart(pred_df.set_index("Date"))
    st.dataframe(pred_df)
