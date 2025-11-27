import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

@st.cache_data
def load_model():
    df = pd.read_csv('hearts.csv')
    le_sex = LabelEncoder(); df['Sex'] = le_sex.fit_transform(df['Sex'])
    le_chest = LabelEncoder(); df['ChestPainType'] = le_chest.fit_transform(df['ChestPainType'])
    le_ecg = LabelEncoder(); df['RestingECG'] = le_ecg.fit_transform(df['RestingECG'])
    le_angina = LabelEncoder(); df['ExerciseAngina'] = le_angina.fit_transform(df['ExerciseAngina'])
    le_slope = LabelEncoder(); df['ST_Slope'] = le_slope.fit_transform(df['ST_Slope'])
    
    X = df.drop(columns=['HeartDisease']); y = df['HeartDisease']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
    
    model = GaussianNB(); model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy, [le_sex, le_chest, le_ecg, le_angina, le_slope]

model, accuracy, encoders = load_model()

st.title("🫀 Heart Disease Predictor")
st.info(f"✅ Accuracy: {accuracy:.1%} | Trained on {len(pd.read_csv('hearts.csv'))} patients")

with st.form("predict"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 20, 100, 40)
        sex = st.selectbox("Sex", ["M", "F"])
        chest = st.selectbox("Chest Pain", ["ATA", "NAP", "ASY", "TA"])
        bp = st.number_input("Resting BP", 80, 200, 140)
        chol = st.number_input("Cholesterol", 0, 600, 200)
    with col2:
        bs = st.selectbox("FastingBS (>120)", [0, 1])
        ecg = st.selectbox("RestingECG", ["Normal", "ST", "LVH"])
        hr = st.number_input("Max HR", 70, 220, 150)
        angina = st.selectbox("Exercise Angina", ["N", "Y"])
        oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 0.0)
        slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
    
    if st.form_submit_button("🔬 Predict"):
        input_data = [[age, 1 if sex=="F" else 0, ["ATA","NAP","ASY","TA"].index(chest),
                      bp, chol, bs, ["Normal","ST","LVH"].index(ecg), hr,
                      1 if angina=="Y" else 0, oldpeak, ["Up","Flat","Down"].index(slope)]]
        
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0]
        
        st.balloons()
        if pred == 1:
            st.error("🚨 **HEART DISEASE DETECTED** ⚠️ Consult doctor immediately!")
        else:
            st.success("✅ **NO HEART DISEASE** - Patient is healthy!")
        st.info(f"**Probability:** Positive: {prob[1]:.1%} | Negative: {prob[0]:.1%}")
