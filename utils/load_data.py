import streamlit as st
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from pytorch_tabnet.tab_model import TabNetClassifier
import torch


def load_data():
    if 'data' not in st.session_state:
        with st.spinner("Loading Data...", show_time=False):
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
        with st.spinner("Training Gaussian Naive Bayes...", show_time=False):
            nb = GaussianNB()
            nb.fit( st.session_state.X_train,  st.session_state.y_train)
            st.session_state.naive_bayes = nb
    #ordinal logistic regression
    if 'ordinal_logistic' not in st.session_state:
        with st.spinner("Training Ordinal Logistic Regression...", show_time=False):
            mod_log = OrderedModel(st.session_state.y_train,
                                    st.session_state.X_train,
                                    distr='logit')
            res_log = mod_log.fit(method='bfgs', disp=False)
            st.session_state.ordinal_logistic = res_log
    
    
    #XGBoost
    if 'xgboost' not in st.session_state:
        with st.spinner("Training XGBoost...", show_time=False):
            xgboost = xgb.XGBClassifier()
            xgboost.load_model('assets/xgb_model.json')
            st.session_state.xgboost = xgboost
        
      #tabnet
    if 'tabnet' not in st.session_state:
        with st.spinner("Training TabNet...", show_time=False):
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



def explain_heatmap(corr_matrix):
    strong_corr_pairs = []
    moderate_corr_pairs = []
    weak_corr_pairs = []

    for col1 in corr_matrix.columns:
        for col2 in corr_matrix.columns:
            if col1 != col2:  # Menghindari korelasi antara variabel dengan dirinya sendiri
                if col1 < col2:  # Pastikan pasangan hanya diproses satu kali (col1 < col2)
                    corr_value = corr_matrix.loc[col1, col2]
                    if abs(corr_value) > 0.7:
                        strong_corr_pairs.append((col1, col2, corr_value))
                    elif 0.3 <= abs(corr_value) <= 0.7:
                        moderate_corr_pairs.append((col1, col2, corr_value))
                    elif abs(corr_value) < 0.3:
                        weak_corr_pairs.append((col1, col2, corr_value))

    # Menampilkan hasil korelasi dalam kategori
    if strong_corr_pairs:
        st.markdown("###- Pasangan Variabel dengan Korelasi Kuat (|Korelasi| > 0.7):")
        for pair in strong_corr_pairs:
            st.write(f"Variabel {pair[0]} dan {pair[1]} memiliki korelasi kuat ({pair[2]:.2f})")

    if moderate_corr_pairs:
        st.markdown("\n###- Pasangan Variabel dengan Korelasi Sedang (0.3 < |Korelasi| < 0.7):")
        for pair in moderate_corr_pairs:
            st.write(f"Variabel {pair[0]} dan {pair[1]} memiliki korelasi sedang ({pair[2]:.2f})")

    if weak_corr_pairs:
        st.markdown("\n###- Pasangan Variabel dengan Korelasi Lemah (|Korelasi| < 0.3):")
        for pair in weak_corr_pairs:
            st.write(f"Variabel {pair[0]} dan {pair[1]} memiliki korelasi lemah ({pair[2]:.2f})")
