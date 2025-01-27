import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Load model dan scaler
rf_model = joblib.load('rf_model.sav')  # Pastikan model sudah disimpan
min_max_scaler = joblib.load('minmax_scaler.sav')  # Pastikan scaler sudah disimpan

def main():
    st.title('Diabetes Prediction Web App')
    st.write('Masukkan data pasien untuk memprediksi apakah mereka memiliki diabetes atau tidak.')

    # Input data dari pengguna
    Pregnancies = st.number_input('Number of Pregnancies', min_value=0, step=1, value=0)
    Glucose = st.number_input('Glucose Level', min_value=0, step=1, value=0)
    BloodPressure = st.number_input('Blood Pressure Level', min_value=0, step=1, value=0)
    SkinThickness = st.number_input('Skin Thickness Value', min_value=0, step=1, value=0)
    Insulin = st.number_input('Insulin Level', min_value=0, step=1, value=0)
    BMI = st.number_input('BMI Value', min_value=0.0, step=0.1, value=0.0)
    DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0, step=0.001, value=0.0)
    Age = st.number_input('Age of the Person', min_value=0, step=1, value=0)

    # Tombol untuk melakukan prediksi
    if st.button('Predict Diabetes'):
        # Membuat data input dalam bentuk array 2D
        input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        # Normalisasi data menggunakan scaler yang telah disimpan
        input_data_normalized = min_max_scaler.transform(input_data)

        # Prediksi menggunakan model yang telah dilatih
        prediction = rf_model.predict(input_data_normalized)

        # Menampilkan hasil prediksi
        if prediction[0] == 0:
            st.success('Prediction: No diabetes')
        else:
            st.success('Prediction: Diabetes')

if __name__ == '__main__':
    main()
