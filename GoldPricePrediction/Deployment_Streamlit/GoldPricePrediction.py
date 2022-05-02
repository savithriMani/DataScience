# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:01:17 2020

"""

import pandas as pd
import streamlit as st 
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.title('Gold Price Prediction')

st.sidebar.header('User Input: ')

DAYS = st.sidebar.number_input("No of days to predict") 
button= st.sidebar.button("Enter")
st.subheader('Prediction for : ')
st.write(DAYS)

    
days=int(DAYS)
gold = pd.read_csv("Gold_data.csv")
gold.set_index('date',inplace=True)
gold = gold.dropna()

model=ExponentialSmoothing(gold["price"],seasonal="mul",trend="add",seasonal_periods=30).fit()
predict=model.predict(start = len(gold) ,end = len(gold)+days )
pred_df=pd.DataFrame()
pred_df['date']=pd.date_range('2021-12-22', periods=days)
pred_df['date']=pd.to_datetime(pred_df['date']).dt.date
pred_df.set_index('date',inplace=True)
pred_df['Predicted Price']=pd.DataFrame(predict)

if button:
    st.subheader('Gold price prediction  Values :')
    st.dataframe(pred_df)

    st.subheader('Plot of predictions :')
    st.line_chart(pred_df)

















