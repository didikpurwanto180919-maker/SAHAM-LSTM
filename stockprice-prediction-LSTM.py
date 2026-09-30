import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import yfinance as yf

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Prediksi Saham ANTM Realtime LSTM", layout="wide")

st.title("📈 Aplikasi Prediksi Saham ANTM (Aneka Tambang) Realtime")
st.markdown("Aplikasi ini menarik data historis dan harga terbaru secara otomatis secara *realtime* dari bursa untuk melatih model LSTM.")

# Sidebar Pengaturan
st.sidebar.header("⚙️ Pengaturan Data & Model")
ticker = st.sidebar.text_input("Kode Saham", value="ANTM.JK")
period = st.sidebar.selectbox("Rentang Data Historis", ["6mo", "1y", "2y", "5y"], index=1)

# Fungsi untuk mengambil data secara otomatis dan melakukan caching
@st.cache_data(ttl=3600)
def load_data(symbol, per):
    df = yf.download(symbol, period=per)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    return df

try:
    with st.spinner(f"Mengambil data realtime untuk {ticker}..."):
        df = load_data(ticker, period)
        
    if df.empty:
        st.error("Data tidak ditemukan. Periksa kembali kode ticker saham.")
        st.stop()
        
    st.subheader("📊 Pratinjau Data Realtime Terbaru")
    st.dataframe(df.tail())
    
    # Memastikan kolom Tanggal dan Close terbaca dengan benar
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    close_col = 'Close'
    
    # Visualisasi Data Historis
    st.subheader("📉 Grafik Harga Historis Realtime")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['Date'], df[close_col], label='Harga Close', color='blue')
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('Harga (IDR)')
    ax.set_title(f'Pergerakan Harga Saham {ticker}')
    ax.legend()
    st.pyplot(fig)
    
    # Pengaturan Parameter LSTM di Sidebar
    st.sidebar.subheader("⚙️ Parameter Model LSTM")
    max_window = max(5, len(df) - 10)
    default_window = min(15, max_window)
    
    prediction_days = st.sidebar.slider("Jumlah Hari untuk Prediksi (Window)", min_value=5, max_value=max_window, value=default_window, step=1)
    epochs = st.sidebar.slider("Epochs Pelatihan", min_value=1, max_value=30, value=5, step=1)
    batch_size = st.sidebar.selectbox("Batch Size", [8, 16, 32], index=1)
    
    if st.sidebar.button("Jalankan Pelatihan & Prediksi"):
        if len(df) <= prediction_days + 5:
            st.error("Data terlalu sedikit untuk window tersebut.")
            st.stop()
            
        with st.spinner("Sedang melatih model LSTM dengan data realtime..."):
            data = df.filter([close_col]).values
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)
            
            training_data_len = int(np.ceil(len(scaled_data) * 0.8))
            train_data = scaled_data[0:training_data_len, :]
            
            x_train, y_train = [], []
            for i in range(prediction_days, len(train_data)):
                x_train.append(train_data[i - prediction_days:i, 0])
                y_train.append(train_data[i, 0])
                
            x_train, y_train = np.array(x_train), np.array(y_train)
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
            
            # Membangun Model LSTM
            model = Sequential()
            model.add(LSTM(units=32, return_sequences=True, input_shape=(x_train.shape[1], 1)))
            model.add(Dropout(0.1))
            model.add(LSTM(units=32, return_sequences=False))
            model.add(Dropout(0.1))
            model.add(Dense(units=16))
            model.add(Dense(units=1))
            
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
            
            test_data = scaled_data[training_data_len - prediction_days:, :]
            x_test = []
            for i in range(prediction_days, len(test_data)):
                x_test.append(test_data[i - prediction_days:i, 0])
                
            x_test = np.array(x_test)
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
            
            predictions = model.predict(x_test)
            predictions = scaler.inverse_transform(predictions)
            
            train = df[:training_data_len]
            valid = df[training_data_len:].copy()
            valid['Predictions'] = predictions
            
            st.subheader("📈 Hasil Prediksi vs Data Aktual")
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            ax2.plot(train['Date'], train[close_col], label='Data Training', color='orange')
            ax2.plot(valid['Date'], valid[close_col], label='Data Aktual', color='blue')
            ax2.plot(valid['Date'], valid['Predictions'], label='Hasil Prediksi', color='red', linestyle='--')
            ax2.set_title(f'Model LSTM - Prediksi Harga Saham {ticker}')
            ax2.set_xlabel('Tanggal')
            ax2.set_ylabel('Harga')
            ax2.legend()
            st.pyplot(fig2)
            
            st.success("Pelatihan dan prediksi selesai!")
            
            # Informasi Estimasi Prediksi Terakhir
            st.subheader("🔮 Estimasi Harga Penutupan Terakhir")
            if not valid.empty:
                last_row = valid.iloc[-1]
                last_date = last_row['Date'].strftime('%d-%m-%Y')
                last_pred = last_row['Predictions']
                st.info(f"Berdasarkan data perdagangan terakhir pada tanggal **{last_date}**, estimasi harga penutupannya adalah: **Rp {last_pred:,.2f}**")

            # Menampilkan tabel hasil prediksi
            st.subheader("📋 Detail Data Prediksi Terbaru")
            st.dataframe(valid[['Date', close_col, 'Predictions']])

except Exception as e:
    st.error(f"Terjadi kesalahan saat memproses data atau model: {e}")
