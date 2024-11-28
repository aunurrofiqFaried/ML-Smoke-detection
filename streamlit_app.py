import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib as jb

rf = jb.load('path/to/smoke-detection-rf.joblib')
sc = jb.load('path/to/standardscaler.joblib')

data = pd.read_csv('path/to/smoke_detection.csv')

option = st.sidebar.selectbox(
      'PILIH MENU :',
      ('Deskripsi','Dataset','Grafik','Prediksi')
)

if option == 'Deskripsi' or option == '' :
      st.write("# Halaman Deskripsi")
      st.write("Dataset ini digunakan untuk mendeteksi potensi kebakaran berdasarkan sensor data lingkungan seperti suhu, kelembaban, tekanan, dan gas kimia.")
      st.write("Berikut adalah statistik dari dataset:")
      st.write(data.describe())
elif option == 'Dataset' :
      st.write("# Halaman Dataset")
      st.write("### Tabel Dataset")
      st.dataframe(data)
elif option == 'Grafik' :
      st.write("# Halaman Grafik")
      st.write("### Grafik Distribusi Sensor")
      
      fig, ax = plt.subplots(figsize=(10, 6))
      data['Temperature[C]'].hist(ax=ax, bins=20, alpha=0.7, label='Temperature[C]')
      data['Humidity[%]'].hist(ax=ax, bins=20, alpha=0.7, label='Humidity[%]')
      ax.set_title('Distribusi Suhu dan Kelembaban')
      ax.set_xlabel('Nilai')
      ax.set_ylabel('Frekuensi')
      ax.legend()

      st.pyplot(fig)
elif option == 'Prediksi':
      st.header("Prediksi Kebakaran")
      st.write("Masukkan nilai-nilai sensor untuk memprediksi potensi kebakaran.")

      col1, col2, col3 = st.columns(3)

      with col1:
            temp = st.number_input('Temperature [C]', value=20.0, min_value=-10.0, max_value=50.0, help="Masukkan suhu dalam derajat Celcius.")
            hum = st.number_input('Humidity [%]', value=57.36, min_value=0.0, max_value=100.0, help="Masukkan kelembaban relatif dalam persen.")
      
      with col2:
            tvoc = st.number_input('TVOC [ppb]', value=0, min_value=0, max_value=10000, help="Masukkan konsentrasi Total Volatile Organic Compounds.")
            eco2 = st.number_input('eCO2 [ppm]', value=400, min_value=0, max_value=5000, help="Masukkan konsentrasi Carbon Dioxide equivalent.")

      with col3:
            h2 = st.number_input('Raw H2', value=12306, min_value=0, help="Masukkan nilai mentah sensor H2.")
            eth = st.number_input('Raw Ethanol', value=18520, min_value=0, help="Masukkan nilai mentah sensor Ethanol.")
            press = st.number_input('Pressure [hPa]', value=939.735, min_value=900.0, max_value=1100.0, help="Masukkan tekanan atmosfer dalam hPa.")

      if st.button('Predict'):
            data_input = np.array([[temp, hum, tvoc, eco2, h2, eth, press]])
            predict = rf.predict(sc.transform(data_input))
            if predict == 0:
                  st.write('Prediksi Alarm Kebakaran = :green[OFF]')
            else:
                  st.write('Prediksi Alarm Kebakaran = :red[ON]')
