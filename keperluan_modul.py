import streamlit as st
import time
import pemanis
import pandas as pd
from datetime import datetime
from PIL import Image
import os
from streamlit_option_menu import option_menu
import base64
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error            
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

def gambar(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def bg():
    img1 = gambar("Documents/image/c2.png")
    img2 = gambar("Documents/image/c1.png")    

    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url(data:image/jpeg;base64,{img1});
        background-size: cover;
        width:100%;
        background-position: center;
    }}

    [data-testid="stSidebar"] {{
        background-image: url("data:image/png;base64,{img2}");
        background-size: cover;
        background-position: center;         
    }}       

    /* Buat header dan semua turunannya transparan */
    [data-testid="stAppHeader"],
    [data-testid="stHeader"],
    [data-testid="stAppHeader"] *,
    [data-testid="stHeader"] * {{
        background-color: rgba(0, 0, 0, 0) !important;
        backdrop-filter: none !important;
    }}

    .st-emotion-cache-12fmjuu,
    .st-emotion-cache-15ecox0,
    .st-emotion-cache-1p1m4ay,
    .st-emotion-cache-1dp5vir {{
        background-color: rgba(0, 0, 0, 0) !important;
    }}

    [data-testid="stToolbar"] {{
        right: 2rem;
    }}
    </style>
    """
    
    st.markdown(page_bg_img, unsafe_allow_html=True)