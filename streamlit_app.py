import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib as jb

rf = jb.load('./smoke-detection-rf.joblib')
sc = jb.load('./standardscaler.joblib')

# with open('smoke-detection-rf.pkl', 'rb') as f:
#       rf = pickle.load(f)

# with open('standardscaler.pkl', 'rb') as f:
#       sc = pickle.load(f)

data = pd.read_csv('./smoke_detection.csv')

option = st.sidebar.selectbox(
      'PILIH MENU :',
      ('Deskripsi','Dataset','Grafik','Prediksi')
)

if option == 'Deskripsi' or option == '' :
      st.write("# Halaman Deskripsi")
      st.write("Fitur prediksi kebakaran dalam aplikasi ini dirancang untuk membantu menganalisis kemungkinan terjadinya kebakaran berdasarkan data yang dihasilkan oleh berbagai sensor lingkungan. Sensor-sensor ini mencakup suhu, kelembaban, tekanan atmosfer, gas kimia seperti H2 dan ethanol, serta indikator gas seperti TVOC (Total Volatile Organic Compounds) dan eCO2 (equivalent Carbon Dioxide). Model machine learning berbasis Random Forest yang telah dilatih digunakan untuk memprediksi apakah alarm kebakaran harus aktif (ON) atau tidak (OFF). Input yang diberikan pengguna akan melalui proses normalisasi menggunakan StandardScaler untuk memastikan prediksi yang akurat. Model ini memberikan notifikasi dini untuk potensi kebakaran, sehingga dapat membantu dalam pengambilan keputusan cepat untuk mitigasi bahaya.")
      st.write("Narmin Humbatli")
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
            temp = st.slider('Temperature [C]', min_value=-10.0, max_value=50.0, value=20.0, step=0.1, help="Atur suhu dalam derajat Celcius.")
            hum = st.slider('Humidity [%]', min_value=0.0, max_value=100.0, value=57.36, step=0.1, help="Atur kelembaban relatif dalam persen.")
      
      with col2:
            tvoc = st.slider('TVOC [ppb]', min_value=0, max_value=10000, value=0, step=1, help="Atur konsentrasi Total Volatile Organic Compounds.")
            eco2 = st.slider('eCO2 [ppm]', min_value=0, max_value=5000, value=400, step=1, help="Atur konsentrasi Carbon Dioxide equivalent.")

      with col3:
            h2 = st.slider('Raw H2', min_value=0, max_value=30000, value=12306, step=1, help="Atur nilai mentah sensor H2.")
            eth = st.slider('Raw Ethanol', min_value=0, max_value=30000, value=18520, step=1, help="Atur nilai mentah sensor Ethanol.")
            press = st.slider('Pressure [hPa]', min_value=900.0, max_value=1100.0, value=939.735, step=0.1, help="Atur tekanan atmosfer dalam hPa.")

      if st.button('Predict'):
            data_input = np.array([[temp, hum, tvoc, eco2, h2, eth, press]])
            predict = rf.predict(sc.transform(data_input))
            if predict == 0:
                  st.write('Prediksi Alarm Kebakaran = :green[OFF]')
            else:
                  st.write('Prediksi Alarm Kebakaran = :red[ON]')
