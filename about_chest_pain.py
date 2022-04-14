import streamlit as st

angina_text = '''
# Typical and Atypical Angina: What to Look For

Angina is a common condition that affects several million people in the United States. Yet most people aren’t aware of the different symptoms and types of this condition for men and women.

Angina pectoris or typical angina is the discomfort that is noted when the heart does not get enough blood or oxygen. Typically, this is caused by blockage or plaque buildup in the coronary arteries. If one or more of the coronary arteries is partially or completely clogged, the heart will not get enough oxygen.

Usually, angina is a symptom that may feel like a tightness or heaviness in the central chest. It may be associated with shortness of breath and perspiration. The location of the discomfort will vary from person to person. Some people may have it in the central part of the chest, some may have it on the left side, some may have it on the right side, and some may have it radiating across the chest. It may also feel like the discomfort moves or radiates to the shoulder, arms, jaw, neck, and back. It usually does not radiate past the wrist into the hand.

Angina may occur during activities such as climbing stairs, carrying groceries, or becoming upset, angry, or going outside into the cold air. Exercise and sexual activity may also cause the symptom to occur.

Men commonly have the usual kind of angina as described above.

Women may have more of a subtle presentation called atypical angina. For example, in one study of over 500 women who suffered a heart attack, 71% had fatigue, 48% had sleep disturbances, 42% had shortness of breath, and 30% had chest discomfort in the month prior to the heart attack. At the time of their heart attack, 58% had shortness of breath, 55% had weakness, 43% had fatigue, and 43% had chest discomfort. The problem may present like an indigestion feeling and can even mimic a problem related to peptic ulcer disease or gallbladder disease.

Angina may also be quite localized, and the symptoms may be mistaken for a muscular pain or muscle pull. One of my patients presented with elbow pain as his sign of a heart attack; another patient had heart attack pain that felt like she had progressively tightening handcuffs on her wrists.

It is important to remember that angina may present both with the usual type of symptoms and also with these more subtle and unusual types of changes. If there has been any significant change in your health or symptoms like these, you should see your healthcare provider immediately.

_Sources_ : [Harington Hospital](https://www.harringtonhospital.org/typical-and-atypical-angina-what-to-look-for)
'''

def showAnginaInfo():
    st.markdown(angina_text)