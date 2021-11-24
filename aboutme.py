import streamlit as st


text = '''
## Meet the Developer

### Shivam Vishwakarma
Computer Science student at **Global Engineering College, Jabalpur**

'''

def showaboutme():
    st.markdown(text)
    st.image('./hero.jpg', caption = "Shivam Vishwakarma",width = 100)