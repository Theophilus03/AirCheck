import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

st.logo("assets/logo.png")

st.set_page_config(
    page_title="Airlytics",
    page_icon="🌫️"
)


st.title("Welcome to AirCheck! 🏠")

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

#dataset
if 'data' not in st.session_state:
  df = pd.read_csv('assets/final.csv')
  st.session_state.data = df
  df_clean = df.copy()
  df_clean = df_clean.replace(['-', 'TIDAK ADA DATA', '---'], np.nan)
  df_clean = df_clean[['pm10', 'so2', 'co', 'o3', 'no2', 'kategori']]

  for col in ['pm10', 'so2', 'co', 'o3', 'no2']:
      df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
  df_clean = df_clean.dropna()

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
  #XGBoost
  import xgboost as xgb
  xgboost = xgb.XGBClassifier(
      objective='multi:softmax',
      num_class=4,
      max_depth=25,
      learning_rate=0.001,
      n_estimators=100,
  )
  xgboost.fit(X_train, y_train)
  st.session_state.xgboost = xgboost

  #tabnet
  from pytorch_tabnet.tab_model import TabNetClassifier
  tabnet = TabNetClassifier()
  tabnet.load_model('assets/full_tabnet_model.pth.zip')
  st.session_state.tabnet = tabnet

  #ordinal logistic regression
  from statsmodels.miscmodels.ordinal_model import OrderedModel
  mod_log = OrderedModel(y_train,
                        X_train,
                        distr='logit')

  res_log = mod_log.fit(method='bfgs', disp=False)
  st.session_state.ordianl_logistic = res_log
