import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# import cufflinks as cf
# import plotly.offline
# cf.go_offline()
# cf.set_config_file(offline=False, world_readable=True)
# import plotly 
# import plotly.express as px
# import plotly.graph_objs as go
# import plotly.offline as py
# from plotly.offline import iplot
# from plotly.subplots import make_subplots
# import plotly.figure_factory as ff

@st.cache
def loaddata():
    with open('heart.csv','rb') as file:
        df = pd.read_csv('heart.csv')
    return df

df = loaddata()

def show_explore_page():
    st.title("Explore Data")
    st.markdown("### Data collected from [**Heart Failure Prediction Dataset**](https://www.kaggle.com/fedesoriano/heart-failure-prediction)")
    data = df["RestingECG"].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(data, labels=data.index, autopct="%1.1f%%", startangle=90)
    ax1.axis("equal")
    st.markdown("#### Types of Resting ECG paitent have")
    st.pyplot(fig1)
    st.markdown("---")
    
