import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Prediksi Harga Saham LSTM", layout="wide")

st.title("📈 Aplikasi Prediksi Harga Saham dengan LSTM")
st.markdown("Unggah file CSV data historis saham Anda untuk melatih model LSTM dan melihat prediksi harga.")

# Sidebar untuk Unggah File
st.sidebar.header("Unggah Data")
uploaded_file = st.sidebar.file_uploader("Unggah file CSV data saham", type=["csv"])

if uploaded_file is not None:
    try:
        # Membaca file CSV
        df = pd.read_csv(uploaded_file)
        
        st.subheader("📊 Pratinjau Data Mentah")
        st.dataframe(df.head())
        
        # Membersihkan nama kolom dari spasi ekstra
        df.columns = df.columns.str.strip()
        
        # Deteksi otomatis kolom Tanggal
        date_col = next((col for col in df.columns if 'tanggal' in col.lower() or 'date' in col.lower()), df.columns[0])
        df['Date'] = pd.to_datetime(df[date_col], format='%d/%m/%Y', errors='coerce')
        if df['Date'].isna().all():
            df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
        
        # Deteksi otomatis kolom Harga Terakhir (Close)
        close_col = next((col for col in df.columns if 'terakhir' in col.lower() or 'close' in col.lower() or 'penutupan' in col.lower()), df.columns[1])
        st.info(f"Menggunakan kolom '{close_col}' sebagai Harga Acuan dan '{date_col}' sebagai Tanggal.")

        # Membersihkan format angka Indonesia (misal: "3.200" -> 3200)
        if df[close_col].dtype == object:
            df[close_col] = df[close_col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
            df[close_col] = pd.to_numeric(df[close_col], errors='coerce')

        # Visualisasi Data Historis
        st.subheader("📉 Grafik Harga Historis")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df['Date'], df[close_col], label='Harga Terakhir', color='blue', marker='o')
        ax.set_xlabel('Tanggal')
        ax.set_ylabel('Harga')
        ax.set_title('Pergerakan Harga Saham Historis')
        ax.legend()
        st.pyplot(fig)

        # Pengaturan Model di Sidebar dengan batasan aman untuk data kecil
        st.sidebar.subheader("⚙️ Pengaturan Model")
        max_window = max(3, len(df) - 5)
        default_window = min(5, max_window)
        
        prediction_days = st.sidebar.slider("Jumlah Hari untuk Prediksi (Window)", min_value=3, max_value=max_window, value=default_window, step=1)
        epochs = st.sidebar.slider("Epochs Pelatihan", min_value=1, max_value=50, value=10, step=1)
        batch_size = st.sidebar.selectbox("Batch Size", [1, 2, 4, 8, 16], index=2)

        if st.sidebar.button("Jalankan Pelatihan & Prediksi"):
            if len(df) <= prediction_days + 3:
                st.error(f"Data terlalu sedikit ({len(df)} baris) untuk window {prediction_days}. Harap kurangi nilai Window di sidebar.")
                st.stop()
                
            with st.spinner("Sedang memproses data dan melatih model LSTM..."):
                # Persiapan Data
                data = df.filter([close_col]).values
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(data)

                # Membagi data training (80%)
                training_data_len = int(np.ceil(len(scaled_data) * 0.8))

                train_data = scaled_data[0:training_data_len, :]
                x_train, y_train = [], []

                for i in range(prediction_days, len(train_data)):
                    x_train.append(train_data[i - prediction_days:i, 0])
                    y_train.append(train_data[i, 0])

                x_train, y_train = np.array(x_train), np.array(y_train)
                x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

                # Membangun Model LSTM yang ringan untuk data kecil
                model = Sequential()
                model.add(LSTM(units=16, return_sequences=True, input_shape=(x_train.shape[1], 1)))
                model.add(Dropout(0.1))
                model.add(LSTM(units=16, return_sequences=False))
                model.add(Dropout(0.1))
                model.add(Dense(units=8))
                model.add(Dense(units=1))

                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

                # Data Pengujian
                test_data = scaled_data[training_data_len - prediction_days:, :]
                x_test = []
                y_test = data[training_data_len:, :]

                for i in range(prediction_days, len(test_data)):
                    x_test.append(test_data[i - prediction_days:i, 0])

                x_test = np.array(x_test)
                x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

                # Prediksi Harga
                predictions = model.predict(x_test)
                predictions = scaler.inverse_transform(predictions)

                # Visualisasi Hasil
                train = df[:training_data_len]
                valid = df[training_data_len:].copy()
                valid['Predictions'] = predictions

                st.subheader("📈 Hasil Prediksi vs Data Aktual")
                fig2, ax2 = plt.subplots(figsize=(12, 6))
                ax2.plot(train['Date'], train[close_col], label='Data Training', color='orange', marker='o')
                ax2.plot(valid['Date'], valid[close_col], label='Data Aktual', color='blue', marker='o')
                ax2.plot(valid['Date'], valid['Predictions'], label='Hasil Prediksi', color='red', linestyle='--', marker='x')
                ax2.set_title('Model LSTM - Prediksi Harga Saham')
                ax2.set_xlabel('Tanggal')
                ax2.set_ylabel('Harga')
                ax2.legend()
                st.pyplot(fig2)

                st.success("Pelatihan dan prediksi selesai!")
                
                # Menampilkan tabel hasil prediksi
                st.subheader("📋 Detail Data Prediksi Terbaru")
                st.dataframe(valid[['Date', close_col, 'Predictions']])

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file atau model: {e}")
else:
    st.info("Silakan unggah file CSV data saham melalui panel di sebelah kiri untuk memulai.")
