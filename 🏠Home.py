import streamlit as st
import numpy as np
import pandas as pd
import base64
from sklearn.model_selection import train_test_split

def image_to_base64(img_path):
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
st.logo("assets/logo.png")

st.set_page_config(
    page_title="AirCheck",
    page_icon="assets/logo.png"
)


with st.spinner("Wait for it...", show_time=True):
    #dataset
    if 'data' not in st.session_state:
      df = pd.read_csv('assets/final.csv')
      df_clean = df.copy()
      df_clean = df_clean.replace(['-', 'TIDAK ADA DATA', '---'], np.nan)
      df_clean = df_clean[['pm10', 'so2', 'co', 'o3', 'no2', 'kategori']]
    
      for col in ['pm10', 'so2', 'co', 'o3', 'no2']:
          df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
      df_clean = df_clean.dropna()
      st.session_state.data = df_clean
      X = df_clean.drop('kategori', axis=1)
      y = df_clean['kategori']
    
      #dataset mapping
      ordinal_mapping = {
          'BAIK': 0,
          'SEDANG': 1,
          'TIDAK SEHAT': 2,
          'SANGAT TIDAK SEHAT': 3
      }
    
      y_encoded = y.map(ordinal_mapping)
    
      #data splitting
      X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y)
      st.session_state.X_train = X_train
      st.session_state.X_test = X_test
      st.session_state.y_train = y_train
      st.session_state.y_test = y_test
    
    #ordinal logistic regression
    if 'ordinal_logistic' not in st.session_state:
      from statsmodels.miscmodels.ordinal_model import OrderedModel
      mod_log = OrderedModel(st.session_state.y_train,
                            st.session_state.X_train,
                            distr='logit')
      res_log = mod_log.fit(method='bfgs', disp=False)
      st.session_state.ordinal_logistic = res_log
    
    
    #XGBoost
    if 'xgboost' not in st.session_state:
      import xgboost as xgb
      xgboost = xgb.XGBClassifier(
          objective='multi:softmax',
          num_class=4,
          max_depth=25,
          learning_rate=0.001,
          n_estimators=100,
      )
      xgboost.fit(st.session_state.X_train, st.session_state.y_train)
      st.write('selesai trainning')
      st.session_state.xgboost = xgboost
      st.write('dah simpen')
        
      #tabnet
    if 'tabnet' not in st.session_state:
      from pytorch_tabnet.tab_model import TabNetClassifier
      tabnet = TabNetClassifier()
      tabnet.load_model('assets/full_tabnet_model.pth.zip')
      st.session_state.tabnet = tabnet





icon_url = f"data:image/png;base64,{image_to_base64('assets/logo.png')}"
title_text = "AirCheck"

st.markdown(f'<img src="{icon_url}" style="vertical-align:middle; display:inline; margin-right:10px; width:50px; height:50px;"> <span style="font-size: 40px; vertical-align:middle;"><strong>{title_text}</strong></span>', unsafe_allow_html=True)

# First section: What is AirCheck?
st.subheader("What is AirCheck?")
st.write("AirCheck adalah platform analisis kualitas udara yang menggunakan data konsentrasi polutan untuk memberikan wawasan mendalam tentang kondisi udara.")

# Second section: What do we provide?
st.subheader("What are we providing?")
st.markdown(
    """
    - **Data Insight**
     Kami menyediakan analisis data historis terkait konsentrasi polutan seperti PM10, CO, NO2, O3, dan SO2. Dengan EDA, Anda dapat memahami pola, tren, dan korelasi antar polutan. Anda juga bisa mengupload dataset anda sendiri untuk mendapatkan data insight dan prediksi yang lebih tepat untuk daerah anda.

    - **Air Quality Predictor**
     Kami menggunakan berbagai model statistik dan machine learning untuk menganalisis kualitas udara dan memprediksi perubahan kualitas udara di masa depan berdasarkan data polutan.
    """
)

