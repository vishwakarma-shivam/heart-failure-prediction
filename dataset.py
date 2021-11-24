import streamlit as st
import pandas as pd
with open('heart.csv','rb') as file:
    data = pd.read_csv('heart.csv')

def showData():
    st.header("Dataset")
    st.dataframe(data)
    st.download_button("Download",data=pd.DataFrame.to_csv(data,index=False), mime="text/csv", file_name ="heart_failure_dataset.csv")