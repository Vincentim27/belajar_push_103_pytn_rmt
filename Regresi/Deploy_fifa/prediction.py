import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

# load the model!
with open('model_lin.pkl', 'rb') as file_1: # rb = read binary
    model_lin=pickle.load(file_1)

# load the scaler!
with open('std_scaler.pkl', 'rb') as file_2:
    std_scaler=pickle.load( file_2)

with open('rbs_scaler.pkl', 'rb') as file_3:
    rbs_scaler=pickle.load( file_3)

# load the encoder
with open('encoder.pkl', 'rb') as file_4:
    encoder=pickle.load( file_4)

# load the num col
with open('num_col_std.txt', 'r') as file_5:
    num_col_std=json.load( file_5)

with open('num_col_rbs.txt', 'r') as file_6:
    num_col_rbs=json.load( file_6)

# load the cat col
with open('cat_col.txt', 'r') as file_7:
   cat_col= json.load( file_7)

def run():
    # Membuat form
    with st.form(key='form parameters'):
        name = st.text_input('Name',value='Dwi')
        age = st.number_input('Age', min_value=12, max_value=60,value=12,step=1,help='Usia Pemain')
        weight = st.number_input('Weight', min_value=50, max_value=150,value=70)
        height = st.slider('Height', 50, 250, 170)
        ValueEUR = st.number_input('ValueEUR', min_value=0,max_value=1000000000,value=0)
        st.markdown('---')

        attackingworkrate = st.selectbox('Attacking Work Rate',('Low','Medium','High'))
        defensiveworkrate = st.selectbox('Defensive Work Rate',('Low','Medium','High'), index=1)
        st.markdown('---')

        pace = st.number_input('Pace', min_value=0, max_value=100, value=88)
        shooting = st.number_input('Shooting', min_value=0, max_value=100, value=88)
        passing = st.number_input('Passing', min_value=0, max_value=100, value=88)
        dribbling = st.number_input('Dribbling', min_value=0, max_value=100, value=88)
        defending = st.number_input('Defending', min_value=0, max_value=100, value=88)
        psysicality = st.number_input('Pysicality', min_value=0, max_value=100, value=88)

        submitted = st.form_submit_button('Predict')

    df_inf = {
        'Name': name, 
        'Age': age, 
        'Height': height, 
        'Weight': weight, 
        'ValueEUR': ValueEUR, 
        'AttackingWorkRate': attackingworkrate,
        'DefensiveWorkRate': defensiveworkrate, 
        'PaceTotal': pace, 
        'ShootingTotal': shooting, 
        'PassingTotal': passing,
        'DribblingTotal': dribbling, 
        'DefendingTotal': defending, 
        'PhysicalityTotal': psysicality
    }
    df_inf = pd.DataFrame([df_inf])
    st.dataframe(df_inf)

    if submitted:
        # Split between num col and cat col
        df_inf_num_rbs = df_inf[num_col_rbs] 
        df_inf_num_std = df_inf[num_col_std]
        df_inf_cat = df_inf[cat_col]

        # Feature scaling 
        df_inf_std_scaled = std_scaler.transform(df_inf_num_std)
        df_inf_rbs_scaled = rbs_scaler.transform(df_inf_num_rbs)
        
        # Feature Encoding
        df_inf_cat_scaled = encoder.transform(df_inf_cat)

        # Concat
        df_inf_final = np.concatenate([df_inf_std_scaled,df_inf_rbs_scaled,df_inf_cat_scaled],axis=1)
        
        # Predict using Linear Regression
        y_pred_inf = model_lin.predict(df_inf_final)

        st.write('# Rating :', str(int(y_pred_inf)))

if __name__ == '__main__':
    run()