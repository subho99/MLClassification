#importing libraries for Streamlit, pycaret, pandas and pandas_profiling
from operator import index
import streamlit as st
import plotly.express as px
import pickle

import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os 

os.environ['NUMEXPR_MAX_THREADS'] = '16'

#Creating Global variable to have access to dataset throughout the app
if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

#Using Streamlit to create the sidebar and options to choose from
with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2019/12/shutterstock_1166533285-Converted-03-1024x564.png")
    st.title("ML Training Model App")
    choice = st.radio("Navigation", ["Upload","Profiling","Classification Modelling", "Regression Modelling", "Download", "View Model"])
    st.info("This application helps you build a ML model using classification or regression techniques")

#using the sidebar choices
if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Profiling": 
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

if choice == "Classification Modelling": 
    from pycaret.classification import setup, compare_models, pull, save_model, load_model
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'): 
        setup(df, target=chosen_target, silent=True, use_gpu=True)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')

if choice == "Regression Modelling": 
    from pycaret.regression import setup, compare_models, pull, save_model, load_model
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'): 
        setup(df, target=chosen_target, silent=True, use_gpu=True)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')
        
        
if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")
        
if choice == "View Model":
        pipeline = load_model("best_model")
        pipeline
        