import streamlit as st
from predict import show_predict_page
from dataset import showData
from explore import show_explore_page
from statistics import showstatistics
from features import showfeatures
from notebook import shownotebook
from learn import showlearncvd
from aboutme import showaboutme


options = ('Predict','Statistics','Learn about CVD\'s','Features','View Dataset','Explore Data','View Notebook','Know the Developer','Credits')
st.sidebar.markdown("### Heart Failure Prediction Using Machine Learning")
opt = st.sidebar.radio("Menu",options)


if opt == options[0]:show_predict_page()
elif opt == options[1]: showstatistics()
elif opt == options[2]: showlearncvd()
elif opt == options[3]: showfeatures()
elif opt == options[4]: showData()
elif opt == options[5]: show_explore_page()
elif opt == options[6]: shownotebook()
elif opt == options[7]: showaboutme()
elif opt == options[8]: pass
