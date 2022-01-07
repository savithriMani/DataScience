# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:01:17 2020

"""

import pandas as pd
import streamlit as st 
from sklearn.linear_model import LogisticRegression

st.title('Model Deployment: Logistic Regression')

st.sidebar.header('User Input Parameters')

def user_input_features():
    CLMSEX = st.sidebar.radio('Gender',('Female','Male'))
    CLMINSUR = st.sidebar.radio('Insurance',('Yes','No'))
    SEATBELT = st.sidebar.radio('SeatBelt',('Yes','No'))
    CLMAGE = st.sidebar.number_input("Insert the Age")
    LOSS = st.sidebar.number_input("Insert Loss")
    data = {'CLMSEX':CLMSEX,
            'CLMINSUR':CLMINSUR,
            'SEATBELT':SEATBELT,
            'CLMAGE':CLMAGE,
            'LOSS':LOSS}
    features = pd.DataFrame(data,index = [0])
    return features 

    
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

if df.at[0,'CLMSEX']=='Female':
    df.at[0,'CLMSEX']='1'
else :
    df.at[0,'CLMSEX']='0'
    
if df.at[0,'CLMINSUR']=='Yes':
    df.at[0,'CLMINSUR']='1'
else :
     df.at[0,'CLMINSUR']='0'
    
if df.at[0,'SEATBELT']=='Yes':
    df.at[0,'SEATBELT']='1'
else :
    df.at[0,'SEATBELT']='0'
    
    

claimants = pd.read_csv("C:\\Users\\Mani\\EXCELR_Codes\\ExcelR_Codes\\Deployment\\claimants.csv")
claimants.drop(["CASENUM"],inplace=True,axis = 1)
claimants = claimants.dropna()

X = claimants.iloc[:,[1,2,3,4,5]]
Y = claimants.iloc[:,0]
clf = LogisticRegression()
clf.fit(X,Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Predicted Result')
st.write('Yes' if prediction_proba[0][1] > 0.5 else 'No')

st.subheader('Prediction Probability')
st.write(prediction_proba)