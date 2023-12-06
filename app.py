import streamlit as st
import joblib
import numpy as np

rf = joblib.load('./smoke-detection-rf.joblib')
sc = joblib.load('./standardscaler.joblib')

st.header("Prediksi Kebakaran")
temp = st.number_input('Temperature[C]', value=20.0)
hum = st.number_input('Humidity[%]', value=57.36)
tvoc = st.number_input('TVOC[ppb]', value=0)
eco2 = st.number_input('eCO2[ppm]', value=400)
h2 = st.number_input('Raw H2', value=12306)
eth = st.number_input('Raw Ethanol', value=18520)
press = st.number_input('Pressure[hPa]', value=939.735)

if st.button('Predict'):
      data_input = np.array([[temp,hum,tvoc,eco2,h2,eth,press]])
      predict = rf.predict(sc.transform(data_input))
      if predict == 0:
            st.write('Prediksi Alarm Kebakaran = :green[OFF]')
      else:
            st.write('Prediksi Alarm Kebakaran = :red[ON]')