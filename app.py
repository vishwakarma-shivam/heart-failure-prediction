import streamlit as st
from predict import show_predict_page
from dataset import showData
from explore import show_explore_page
from statistics import showstatistics
from features import showfeatures


options = ('Predict','Statistics','Learn about CVD\'s','Features','View Dataset','Explore Data','Know the Developer','Credits')
opt = st.sidebar.radio("What you want to do today?",options)

if opt == options[0]:show_predict_page()
elif opt == options[1]: showstatistics()
elif opt == options[2]: showstatistics()
elif opt == options[3]: showfeatures()
elif opt == options[4]: showData()
elif opt == options[5]: show_explore_page()
elif opt == options[6]: showfeatures()
elif opt == options[7]: showData()
