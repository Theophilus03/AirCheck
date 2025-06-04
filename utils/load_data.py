import streamlit as st
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from pytorch_tabnet.tab_model import TabNetClassifier


def load_data():
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
    #naive bayes
    if 'naive_bayes' not in st.session_state:
        nb = GaussianNB()
        nb.fit( st.session_state.X_train,  st.session_state.y_train)
        st.session_state.naive_bayes = nb
    #ordinal logistic regression
    if 'ordinal_logistic' not in st.session_state:
        mod_log = OrderedModel(st.session_state.y_train,
                                st.session_state.X_train,
                                distr='logit')
        res_log = mod_log.fit(method='bfgs', disp=False)
        st.session_state.ordinal_logistic = res_log
    
    
    #XGBoost
    if 'xgboost' not in st.session_state:
        xgboost = xgb.XGBClassifier()
        xgboost.load_model('assets/xgb_model.json')
        st.session_state.xgboost = xgboost
        
      #tabnet
    if 'tabnet' not in st.session_state:
        tabnet = TabNetClassifier()
        tabnet.load_model('assets/full_tabnet_model.pth.zip')
        st.session_state.tabnet = tabnet


def explain_countplot(category_counts):
    total = category_counts.sum()
    max_cat = category_counts.idxmax()
    max_count = category_counts.max()
    min_cat = category_counts.idxmin()
    min_count = category_counts.min()
    
    prop_max = max_count / total
    prop_min = min_count / total
    
    explanation = []
    explanation.append(f"Dari total {total} data, kategori '{max_cat}' adalah yang paling banyak dengan jumlah {max_count} ({prop_max:.1%} dari total).")
    explanation.append(f"Kategori '{min_cat}' memiliki jumlah paling sedikit, yaitu {min_count} ({prop_min:.1%} dari total).")
    
    if prop_max - prop_min > 0.2:
        explanation.append("Distribusi kategori ini cukup tidak seimbang.")
    else:
        explanation.append("Distribusi kategori ini relatif seimbang.")
    
    return " ".join(explanation)
