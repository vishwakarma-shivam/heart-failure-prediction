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
    gender = st.radio("Gender",['M','F'],)
    chestPainType = st.selectbox("Chest Pain Type",  ["Typical Angina (TA)", "Atypical Angina (ATA)", "Non-Anginal Pain (NAP)", "Asymptomatic (ASY)"],help="Angina is chest pain or discomfort you feel when there is not enough blood flow to your heart muscle. Your heart muscle needs the oxygen that the blood carries. Angina may feel like pressure or a squeezing pain in your chest. It may feel like indigestion. You may also feel pain in your shoulders, arms, neck, jaw, or back. ")
    bpinfo ='''
    Diastolic blood pressure (the second number) – indicates how much pressure your blood is exerting against your artery walls while the heart is resting between beats.
    '''
    restingBP = st.number_input("Resting Blood Pressure [mm Hg]", 0, 200,help = bpinfo)
    cholesterol = st.number_input("Cholesterol",0)
    fastingBS = st.radio("Fasting Blood Sugar higher than 120 mg/dl ?", ["No", "Yes"])
    ecginfo = '''
    The resting electrocardiogram is a test that measures the electrical activity of the heart.'''
    restingECG = st.selectbox("Resting ECG", ["Normal","ST","LVH"],help = ecginfo)
    maxHR = st.slider("Maximum Heart Rate",60,202)
    eainfo ='''
    When you climb stairs, exercise or walk, your heart demands more blood, but narrowed arteries slow down blood flow.
    That will result in chest pain for some period of time.
    '''
    exerciseAngina = st.radio("Exercise Angina",["No","Yes"], help = eainfo)
    oldpeak = st.slider('Oldpeak', 0.0,5.0, help = "oldpeak = ST depression induced by exercise relative to rest")
    slopeinfo = '''
    The ST segment shift relative to exercise-induced increments in heart rate, the ST/heart rate slope (ST/HR slope), has been proposed as a more accurate ECG criterion for diagnosing significant coronary artery disease (CAD).'''
    stSlope = st.selectbox("ST Slope",["Up","Flat","Down"],help = slopeinfo)

    chestPainType = chstPain(chestPainType)
    fastingBS = 1 if fastingBS == "Yes" else 0
    exerciseAngina = 'Y' if exerciseAngina == "Yes" else 'N'

    ok = st.button("Predict")
    if ok:
        trial = np.array([[age,gender,chestPainType,restingBP,cholesterol,fastingBS,restingECG,maxHR,exerciseAngina,oldpeak,stSlope]])
        res = model.predict(trial)
        result = "Safe" if res[0] == 0 else "Not Safe"
        st.subheader(f"Result :{result}")
        
        




