# Sebelum import streamlit, install dulu, make sure environment nya sama!
# !pip install streamlit 

import streamlit as st
import eda
import prediction

navigation = st.sidebar.selectbox('Pilih Halaman:',('EDA','Predict a player'))

if navigation == 'EDA':
    eda.run()
else:
    prediction.run()