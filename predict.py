import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

model = data['model']

chest_pain_help = '''Angina is chest pain or discomfort you feel when there is not enough blood flow to your heart muscle. 
    Your heart muscle needs the oxygen that the blood carries. Angina may feel like pressure or a squeezing pain in your chest. It may feel like indigestion.
         You may also feel pain in your shoulders, arms, neck, jaw, or back. 
         Go to Learn About Angina for more details.
         '''

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
    chestPainType = st.selectbox("Chest Pain Type",  ["Typical Angina (TA)", "Atypical Angina (ATA)", "Non-Anginal Pain (NAP)", "Asymptomatic (ASY)"],
    help= chest_pain_help)
    
    bpinfo ='''
    Diastolic blood pressure (the second number) – indicates how much pressure your blood is exerting against your artery walls while the heart is resting between beats.
    '''
    restingBP = st.number_input("Resting Blood Pressure [mm Hg]", 0, 200,help = bpinfo)

    cholesterol_help = '''
    The list below shows healthy levels of cholesterol by age, according to the National Institutes of Health (NIH). Doctors measure cholesterol in milligrams per deciliter (mg/dl).

    - Anyone 19 or younger: less than 170 mg/dl
    - Men aged 20 or over: 	125–200 mg/dl
    - Women aged 20 or over: 125–200 mg/dl
    '''
    cholesterol = st.number_input("Cholesterol",0, help = cholesterol_help)
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
        st.subheader("Results")
        st.markdown(f"You are _{result}_ .")

        st.markdown("##### Precautions: ")
        
        if(restingBP>90):
            bp_info = '''
            - A normal blood pressure level is less than 120/80 mmHg.
            No matter your age, you can [take steps each day to keep your blood pressure in a healthy range.](https://www.cdc.gov/bloodpressure/prevent.htm)
            '''
            st.markdown(bp_info)
        if ( (age<20 and cholesterol>170) or (age>=20 and cholesterol > 200)):
            cholesterol_info='''
            - Total cholesterol levels under 200 mg/dl are healthy for adults.
            Doctors treat readings of 200–239 mg/dl as borderline high, and readings of at least 240 mg/dl as high. 
            Go to this [link to learn more.](https://www.mayoclinic.org/diseases-conditions/high-blood-cholesterol/in-depth/reduce-cholesterol/art-20045935) 
            '''
            st.markdown(cholesterol_info)


        #st.subheader(f"Result :{result}")
        
        




