import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

model = data['model']

def chstPain(chestPainType):
    if chestPainType == "Typical Angina (TA)":chestPainType = "TA"
    elif chestPainType == "Atypical Angina (ATA)": chestPainType = "ATA"
    elif chestPainType == "Non-Anginal Pain (NAP)": chestPainType = "NAP"
    elif chestPainType == "Asymptomatic (ASY)": chestPainType = "ASY"
    return chestPainType
    

def show_predict_page():
    st.title('Heart Failure Prediction')
    age = st.number_input('Age',0)
    gender = st.radio("Gender",['M','F'])
    chestPainType = st.selectbox("Chest Pain Type",  ["Typical Angina (TA)", "Atypical Angina (ATA)", "Non-Anginal Pain (NAP)", "Asymptomatic (ASY)"])
    restingBP = st.number_input("Resting Blood Pressure [mm Hg]", 0, 200)
    cholesterol = st.number_input("Cholesterol",0)
    fastingBS = st.radio("Fasting Blood Sugar higher than 120 mg/dl ?", ["No", "Yes"])
    restingECG = st.selectbox("Resting ECG", ["Normal","ST","LVH"])
    maxHR = st.slider("Maximum Heart Rate",60,202)
    exerciseAngina = st.radio("Excercise Angina",["No","Yes"])
    oldpeak = st.slider('Oldpeak', 0.0,5.0)
    stSlope = st.selectbox("ST Slope",["Up","Flat","Down"])

    chestPainType = chstPain(chestPainType)
    fastingBS = 1 if fastingBS == "Yes" else 0
    exerciseAngina = 'Y' if exerciseAngina == "Yes" else 'N'

    ok = st.button("Predict")
    if ok:
        trial = np.array([[age,gender,chestPainType,restingBP,cholesterol,fastingBS,restingECG,maxHR,exerciseAngina,oldpeak,stSlope]])
        res = model.predict(trial)
        result = "Safe" if res[0] == 0 else "Not Safe"
        st.subheader(f"Result :{result}")




