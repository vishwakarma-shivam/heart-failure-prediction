import streamlit as st

overview = '''
### Overview
Cardiovascular diseases (CVDs) are the leading cause of death globally, taking an estimated 17.9 million lives each year. CVDs are a group of disorders of the heart and blood vessels and include coronary heart disease, cerebrovascular disease, rheumatic heart disease and other conditions. More than four out of five CVD deaths are due to heart attacks and strokes, and one third of these deaths occur prematurely in people under 70 years of age.

The most important behavioural risk factors of heart disease and stroke are unhealthy diet, physical inactivity, tobacco use and harmful use of alcohol. The effects of behavioural risk factors may show up in individuals as raised blood pressure, raised blood glucose, raised blood lipids, and overweight and obesity. These “intermediate risks factors” can be measured in primary care facilities and indicate an increased risk of heart attack, stroke, heart failure and other complications.

Cessation of tobacco use, reduction of salt in the diet, eating more fruit and vegetables, regular physical activity and avoiding harmful use of alcohol have been shown to reduce the risk of cardiovascular disease. Health policies that create conducive environments for making healthy choices affordable and available are essential for motivating people to adopt and sustain healthy behaviours.

Identifying those at highest risk of CVDs and ensuring they receive appropriate treatment can prevent premature deaths. Access to noncommunicable disease medicines and basic health technologies in all primary health care facilities is essential to ensure that those in need receive treatment and counselling.
'''

symptoms = '''


### Heart attack and stroke

Often, there are no symptoms of the underlying disease of the blood vessels. A heart attack or stroke may be the first sign of underlying disease. Symptoms of a heart attack include:

    pain or discomfort in the centre of the chest; and/or
    pain or discomfort in the arms, the left shoulder, elbows, jaw, or back.

In addition the person may experience difficulty in breathing or shortness of breath; nausea or vomiting; light-headedness or faintness; a cold sweat; and turning pale. Women are more likely than men to have shortness of breath, nausea, vomiting, and back or jaw pain.

The most common symptom of a stroke is sudden weakness of the face, arm, or leg, most often on one side of the body. Other symptoms include sudden onset of:

    numbness of the face, arm, or leg, especially on one side of the body;
    confusion, difficulty speaking or understanding speech;
    difficulty seeing with one or both eyes;
    difficulty walking, dizziness and/or loss of balance or coordination;
    severe headache with no known cause; and/or
    fainting or unconsciousness.

People experiencing these symptoms should seek medical care immediately.

### Rheumatic heart disease

Symptoms of rheumatic heart disease include: shortness of breath, fatigue, irregular heartbeats, chest pain and fainting. Symptoms of rheumatic fever (which can cause rheumatic heart disease if not treated) include: fever, pain and swelling of the joints, nausea, stomach cramps and vomiting.

'''

treatment = '''

### Treatment

WHO supports governments to prevent, manage and monitor CVDs by developing global strategies to reduce the incidence, morbidity and mortality of these diseases. These strategies include reducing risk factors, developing standards of care, enhancing health system capacity to care for patients with CVD, and monitoring disease patterns and trends to inform national and global actions.

The risk factors for CVD include behaviours such as tobacco use, an unhealthy diet, harmful use of alcohol and inadequate physical activity. They also include physiological factors, including high blood pressure (hypertension), high blood cholesterol and high blood sugar or glucose, which are linked to underlying social determinants and drivers such as ageing, income and urbanization. 
'''

def showoverview():
    st.markdown(overview)

def showsymptoms():
    st.markdown(symptoms)

def showtreatement():
    st.markdown(treatment)

def showlearncvd():
    st.header("Learn about CVD's ")
    c1, c2 =st.columns(2)
    
    opt = ("Overview", "Symptoms", "Treatment")
    with c1:
        select = st.radio("",opt)
    with c2:
        st.image('./images/heartim.jpg')
    if select == opt[0]: showoverview()
    elif select == opt[1]:showsymptoms()
    elif select == opt[2]:showtreatement()

    st.markdown("Credit [WHO](https://www.who.int/health-topics/cardiovascular-diseases)")