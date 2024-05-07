# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 16:47:29 2024

@author: sceli
"""
import subprocess


import streamlit as st
import pandas as pd
segments = pd.read_csv(r"D:\CE 3.sınıf\Spring Semester\Ara Proje\Data\106_segments_v1.csv", names=['segment'])

day = None 
segment = None

st.title("Short and Long Holidays Traffic Speed Prediction")
    
# Description
st.text("Choose a Holiday and a Segment in the sidebar. Input your values and get a prediction.")
days = ["1Mayis", "15Tem", "19Mayis", "23Nisan", "29Ekim", "30Aug", "Kurban", "Ramazan"]
#sidebar
sideBar = st.sidebar
day = sideBar.selectbox('Which Holiday do you want to predict?',days)
segment = sideBar.selectbox('Which segment do you want to use?',segments)

result = st.button("Run Model")

if result:
    # Run the other script
    subprocess.run(["python", "main.py"])

def get_day():
    return day
def get_segment():
    return segment
