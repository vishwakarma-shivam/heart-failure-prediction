import streamlit as st


text = '''
## Meet the Developer

### Shivam Vishwakarma


'''

def showaboutme():
    st.markdown(text)
    st.image('./images/hero2.png', caption = "Shivam Vishwakarma",width = 150)
    st.markdown("Computer Science student at **Global Engineering College, Jabalpur**")
    st.markdown("`I am currently enrolled in *Bachelor's degree* course in Computer Science.I made a quite few projects that you can check in my github🐱‍🏍 profile.`")
    st.markdown("My public profiles: ")
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1:
        st.markdown("[Linkedin🔗](https://www.linkedin.com/in/shivamvishwakarma)")
    with c2:
        st.markdown("[Github👨‍💻](https://www.github.com/svshiva)")
    with c3:
        st.markdown("[Instagram📷](https://www.instagram.com/sv_shivam_)")
    with c4:
        st.markdown("[Google Cloud☁](https://www.qwiklabs.com/public_profiles/771a5045-4014-4a37-bc93-c9504e1b82fc )")
    with c5:
        st.markdown("[kaggle🐱‍🏍](https://www.kaggle.com/codershiv)")
        

        