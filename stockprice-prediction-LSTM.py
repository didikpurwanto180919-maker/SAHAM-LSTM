# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Harga Saham LSTM", page_icon="📈", layout="wide")

# Judul aplikasi
st.title("📈 Prediksi Harga Saham Unilever Indonesia menggunakan LSTM")
st.markdown("---")

# Sidebar untuk upload file
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload file CSV data saham", type=["csv"])

if uploaded_file is not None:
    # Membaca dataset
    df = pd.read_csv(uploaded_file)
    
    # Preprocessing data
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    
    # Tab untuk organisasi konten
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Eksplorasi Data", 
        "🔧 Preprocessing", 
        "🧠 Model LSTM", 
        "📈 Prediksi", 
        "📋 Evaluasi"
    ])
    
    with tab1:
        st.header("Eksplorasi Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Preview Data")
            st.dataframe(df.head(), use_container_width=True)
            
        with col2:
            st.subheader("Informasi Data")
            st.write(f"**Shape Data:** {df.shape}")
            st.write(f"**Tanggal Mulai:** {df.index.min()}")
            st.write(f"**Tanggal Akhir:** {df.index.max()}")
            st.write(f"**Missing Values:**")
            st.write(df.isnull().sum())
        
        # Visualisasi data
        st.subheader("Visualisasi Harga Saham")
        
        # Pilihan tipe visualisasi
        viz_option = st.selectbox(
            "Pilih tipe visualisasi:",
            ["High dan Low", "Open dan Close", "Close dan Adj Close", "Close Only"]
        )
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        
        if viz_option == "High dan Low":
            ax.plot(df.index, df['High'], label='High')
            ax.plot(df.index, df['Low'], label='Low')
        elif viz_option == "Open dan Close":
            ax.plot(df.index, df['Open'], label='Open')
            ax.plot(df.index, df['Close'], label='Close')
        elif viz_option == "Close dan Adj Close":
            ax.plot(df.index, df['Close'], label='Close')
            ax.plot(df.index, df['Adj Close'], label='Adj Close')
        else:
            ax.plot(df.index, df['Close'], label='Close')
        
        ax.set_xlabel('Tanggal')
        ax.set_ylabel('Harga (Rp)')
        ax.set_title(f"Harga Saham Unilever Indonesia - {viz_option}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with tab2:
        st.header("Preprocessing Data")
        
        # Feature Scaling
        st.subheader("Feature Scaling dengan MinMaxScaler")
        ms = MinMaxScaler()
        df['Close_ms'] = ms.fit_transform(df[['Close']])
        
        # Split data
        st.subheader("Pembagian Data Training dan Testing")
        train_size = st.slider("Persentase data training:", 0.7, 0.9, 0.8, 0.05)
        
        def split_data(data, train_size):
            size = int(len(data) * train_size)
            train, test = data.iloc[0:size], data.iloc[size:len(data)]
            return train, test
        
        train, test = split_data(df['Close_ms'], train_size)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Data Training:** {len(train)} samples ({train_size*100}%)")
            st.write(f"**Data Testing:** {len(test)} samples ({100-train_size*100}%)")
        
        with col2:
            # Plot pembagian data
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(train.index, train, label='Training')
            ax.plot(test.index, test, label='Testing')
            ax.set_title(f'Pembagian Data: {int(train_size*100)}% Training & {int(100-train_size*100)}% Testing')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig)
    
    with tab3:
        st.header("Pembangunan Model LSTM")
        
        # Parameter model
        st.subheader("Hyperparameter Model")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            look_back = st.slider("Look Back Period:", 1, 10, 1)
            lstm_units1 = st.slider("LSTM Layer 1 Units:", 32, 256, 128)
        
        with col2:
            dropout_rate = st.slider("Dropout Rate:", 0.1, 0.5, 0.2, 0.1)
            lstm_units2 = st.slider("LSTM Layer 2 Units:", 16, 128, 64)
        
        with col3:
            dense_units = st.slider("Dense Layer Units:", 16, 64, 32)
            learning_rate = st.selectbox("Learning Rate:", [0.001, 0.0005, 0.0001])
        
        # Persiapan data untuk LSTM
        def split_target(data, look_back=1):
            X, y = [], []
            for i in range(len(data) - look_back):
                a = data[i:(i + look_back), 0]
                X.append(a)
                y.append(data[i + look_back, 0])
            return np.array(X), np.array(y)
        
        X_train, y_train = split_target(train.values.reshape(len(train), 1), look_back)
        X_test, y_test = split_target(test.values.reshape(len(test), 1), look_back)
        
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        
        # Build model
        st.subheader("Arsitektur Model")
        
        model = Sequential([
            LSTM(lstm_units1, input_shape=(1, look_back), return_sequences=True),
            Dropout(dropout_rate),
            LSTM(lstm_units2),
            Dropout(dropout_rate),
            Dense(dense_units, activation='relu'),
            Dense(1)
        ])
        
        # Menampilkan summary model
        with st.expander("Lihat Summary Model"):
            model_summary = []
            model.summary(print_fn=lambda x: model_summary.append(x))
            st.text("\n".join(model_summary))
        
        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer,
                     metrics=["mae"],
                     loss=tf.keras.losses.Huber())
        
        # Training model
        st.subheader("Training Model")
        epochs = st.slider("Jumlah Epochs:", 50, 300, 200, 50)
        batch_size = st.selectbox("Batch Size:", [16, 32, 64], index=1)
        
        if st.button("Mulai Training"):
            with st.spinner("Training model dalam proses..."):
                # Callback untuk early stopping
                class Callback(tf.keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs={}):
                        if logs.get('val_mae') is not None and logs.get('val_mae') < 0.01:
                            self.model.stop_training = True
                
                history = model.fit(X_train,
                                  y_train,
                                  epochs=epochs,
                                  batch_size=batch_size,
                                  validation_data=(X_test, y_test),
                                  shuffle=False,
                                  verbose=0,
                                  callbacks=[Callback()])
                
                st.success("Training selesai!")
                
                # Plot training history
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                
                # Loss
                ax1.plot(history.history['loss'])
                ax1.plot(history.history['val_loss'])
                ax1.legend(['Loss','Validation Loss'])
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.set_title('Training dan Validation Loss')
                ax1.grid(True, alpha=0.3)
                
                # MAE
                ax2.plot(history.history['mae'])
                ax2.plot(history.history['val_mae'])
                ax2.legend(['MAE','Validation MAE'])
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Mean Absolute Error')
                ax2.set_title('Training dan Validation MAE')
                ax2.grid(True, alpha=0.3)
                
                st.pyplot(fig)
    
    with tab4:
        st.header("Hasil Prediksi")
        
        if 'model' in locals():
            # Prediksi
            pred = model.predict(X_test, verbose=0)
            y_pred = np.array(pred).reshape(-1)
            
            # Plot hasil prediksi
            fig, ax = plt.subplots(figsize=(15, 7))
            ax.plot(test.index[look_back-1:-1], y_test, color='blue', label='Actual', linewidth=2)
            ax.plot(test.index[look_back-1:-1], y_pred, color='red', label='Predicted', linewidth=2)
            
            # Menambahkan metric MAE pada plot
            mae = mean_absolute_error(y_test, y_pred)
            ax.text(0.02, 0.95, f"MAE = {mae:.4f}", transform=ax.transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7),
                   fontsize=12)
            
            ax.set_xlabel('Tanggal')
            ax.set_ylabel('Harga Saham (Scaled)')
            ax.set_title('Prediksi Harga Saham Unilever Indonesia\nModel LSTM')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            # Prediksi dalam skala asli
            st.subheader("Prediksi dalam Skala Asli")
            y_pred_original = ms.inverse_transform(np.array(y_pred).reshape(-1, 1))
            y_test_original = ms.inverse_transform(np.array(y_test).reshape(-1, 1))
            
            fig2, ax2 = plt.subplots(figsize=(15, 7))
            ax2.plot(df.index, df['Close'], color='blue', label='Actual Full Data', alpha=0.7)
            ax2.plot(test.index[look_back-1:-1], y_test_original, color='green', 
                    label='Actual Testing', linewidth=2)
            ax2.plot(test.index[look_back-1:-1], y_pred_original, color='red', 
                    label='Predicted', linewidth=2)
            
            ax2.set_xlabel('Tanggal')
            ax2.set_ylabel('Harga Saham (Rp)')
            ax2.set_title('Prediksi Harga Saham Unilever Indonesia\nModel LSTM (Skala Asli)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            plt.xticks(rotation=45)
            st.pyplot(fig2)
    
    with tab5:
        st.header("Evaluasi Model")
        
        if 'model' in locals() and 'y_pred' in locals():
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mape = mean_absolute_percentage_error(y_test, y_pred)
            
            # Tampilkan metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Mean Absolute Error (MAE)", f"{mae:.6f}")
            
            with col2:
                st.metric("Root Mean Square Error (RMSE)", f"{rmse:.6f}")
            
            with col3:
                st.metric("Mean Absolute Percentage Error (MAPE)", f"{mape:.6f}")
            
            # Interpretasi hasil
            st.subheader("Interpretasi Hasil")
            st.info("""
            **Penjelasan Metrik Evaluasi:**
            - **MAE (Mean Absolute Error)**: Rata-rata selisih absolut antara prediksi dan nilai aktual
            - **RMSE (Root Mean Square Error)**: Akar kuadrat dari rata-rata kuadrat error, lebih sensitif terhadap outlier
            - **MAPE (Mean Absolute Percentage Error)**: Rata-rata persentase error relatif terhadap nilai aktual
            
            **Nilai yang lebih kecil menunjukkan performa model yang lebih baik.**
            """)
            
            # Tabel perbandingan hasil prediksi
            st.subheader("Perbandingan Nilai Aktual vs Prediksi")
            comparison_df = pd.DataFrame({
                'Tanggal': test.index[look_back-1:-1],
                'Actual': y_test_original.flatten(),
                'Predicted': y_pred_original.flatten(),
                'Error': (y_test_original.flatten() - y_pred_original.flatten())
            })
            
            st.dataframe(comparison_df.tail(10), use_container_width=True)
            
            # Download hasil prediksi
            csv = comparison_df.to_csv(index=False)
            st.download_button(
                label="Download Hasil Prediksi (CSV)",
                data=csv,
                file_name="hasil_prediksi_saham.csv",
                mime="text/csv"
            )

else:
    st.info("👆 Silakan upload file CSV data saham untuk memulai analisis")
    st.markdown("""
    **Format file yang diharapkan:**
    - Kolom 'Date' (tanggal)
    - Kolom 'Open' (harga pembukaan)
    - Kolom 'High' (harga tertinggi)
    - Kolom 'Low' (harga terendah) 
    - Kolom 'Close' (harga penutupan)
    - Kolom 'Adj Close' (harga penutupan disesuaikan)
    - Kolom 'Volume' (volume perdagangan)
    """)

# Footer
st.markdown("---")
st.markdown("Dikembangkan dengan Streamlit • Prediksi Harga Saham menggunakan LSTM")