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
        
        # Membersihkan nama kolom (menghapus spasi ekstra jika ada)
        df.columns = df.columns.str.strip()
        
        # Deteksi otomatis kolom tanggal
        date_col = None
        for col in df.columns:
            if 'date' in col.lower() or 'tanggal' in col.lower():
                date_col = col
                break
                
        if date_col is None:
            # Jika tidak ditemukan, ambil kolom pertama sebagai asumsi tanggal
            date_col = df.columns[0]
            st.warning(f"Kolom tanggal tidak terdeteksi secara spesifik. Menggunakan kolom pertama: '{date_col}'")
        else:
            st.info(f"Menggunakan kolom '{date_col}' sebagai kolom Tanggal.")
            
        # Konversi kolom tanggal dan urutkan data
        df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Deteksi otomatis kolom Harga Penutupan (Close)
        close_col = None
        for col in df.columns:
            if 'close' in col.lower() or 'closing' in col.lower() or 'adj close' in col.lower() or 'penutupan' in col.lower():
                close_col = col
                break
                
        if close_col is None:
            # Cari kolom numerik selain tanggal jika tidak ada yang bernama 'close'
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                close_col = numeric_cols[-1] # Biasanya kolom harga berada di dekat akhir
                st.warning(f"Kolom 'Close' tidak terdeteksi. Menggunakan kolom numerik: '{close_col}'")
            else:
                st.error("Tidak ditemukan kolom numerik yang valid untuk harga saham.")
                st.stop()
        else:
            st.info(f"Menggunakan kolom '{close_col}' sebagai Harga Penutupan (Close).")

        # Visualisasi Data Historis
        st.subheader("📉 Grafik Harga Historis")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df['Date'], df[close_col], label='Harga Close', color='blue')
        ax.set_xlabel('Tanggal')
        ax.set_ylabel('Harga')
        ax.set_title('Pergerakan Harga Saham Historis')
        ax.legend()
        st.pyplot(fig)

        # Konfigurasi Model di Sidebar
        st.sidebar.subheader("⚙️ Pengaturan Model")
        prediction_days = st.sidebar.slider("Jumlah Hari untuk Prediksi (Window)", min_value=10, max_value=100, value=60, step=5)
        epochs = st.sidebar.slider("Epochs Pelatihan", min_value=1, max_value=50, value=5, step=1)
        batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)

        if st.sidebar.button("Jalankan Pelatihan & Prediksi"):
            with st.spinner("Sedang memproses data dan melatih model LSTM..."):
                # Persiapan Data
                data = df.filter([close_col]).values
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(data)

                # Membagi data training
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
                model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
                model.add(Dropout(0.2))
                model.add(LSTM(units=50, return_sequences=False))
                model.add(Dropout(0.2))
                model.add(Dense(units=25))
                model.add(Dense(units=1))

                model.compile(optimizer='adam', loss='mean_squared_error')
                
                # Melatih model
                model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

                # Data Pengujian (Testing Data)
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

                # Evaluasi / Visualisasi Hasil
                train = df[:training_data_len]
                valid = df[training_data_len:].copy()
                valid['Predictions'] = predictions

                st.subheader("📈 Hasil Prediksi vs Data Aktual")
                fig2, ax2 = plt.subplots(figsize=(12, 6))
                ax2.plot(train['Date'], train[close_col], label='Data Training', color='orange')
                ax2.plot(valid['Date'], valid[close_col], label='Data Aktual (Actual)', color='blue')
                ax2.plot(valid['Date'], valid['Predictions'], label='Hasil Prediksi (Prediction)', color='red', linestyle='--')
                ax2.set_title('Model LSTM - Prediksi Harga Saham')
                ax2.set_xlabel('Tanggal')
                ax2.set_ylabel('Harga Penutupan')
                ax2.legend()
                st.pyplot(fig2)

                st.success("Pelatihan dan prediksi selesai!")
                
                # Menampilkan tabel hasil prediksi terakhir
                st.subheader("📋 Detail Data Prediksi Terbaru")
                st.dataframe(valid[['Date', close_col, 'Predictions']].tail(10))

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file atau model: {e}")
else:
    st.info("Silakan unggah file CSV data saham melalui panel di sebelah kiri untuk memulai.")
