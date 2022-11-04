import streamlit as st
from pycaret.classification import load_model

with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("ML Training Model App")
    st.info("Displaying the Trained ML Model")
    
pipeline = load_model("best_model") 
pipeline