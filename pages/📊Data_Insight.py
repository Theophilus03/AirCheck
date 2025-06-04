import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from scipy.stats import norm

from sklearn.metrics import classification_report
import torch

from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from pytorch_tabnet.tab_model import TabNetClassifier

from utils.load_data import load_data, explain_countplot, explain_heatmap

st.logo("assets/logo.png")

st.set_page_config(
    page_title="AirCheck",
    page_icon="assets/logo.png"
)

st.title("Data Insight")


def split_data(df):
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

# Upload CSV file
uploaded_file = st.sidebar.file_uploader("Upload CSV file for Data Insight", type="csv")
REQUIRED_COLUMNS = ['pm10', 'so2', 'co', 'o3', 'no2', 'kategori']

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
        else:
            st.session_state.data = df
            st.success("Data uploaded and validated successfully!")
            split_data(df)

            #naive bayes
            with st.spinner("Training Naive Bayes Model...", show_time=False):
                nb = GaussianNB()
                nb.fit( st.session_state.X_train,  st.session_state.y_train)
                st.session_state.naive_bayes = nb
            
            #ordinal logistic regression
            with st.spinner("Training Ordinal Logistic Regression Model...", show_time=False):
                mod_log = OrderedModel(st.session_state.y_train,
                                        st.session_state.X_train,
                                        distr='logit')
                res_log = mod_log.fit(method='bfgs', disp=False)
                st.session_state.ordinal_logistic = res_log
    
            #tabnet
            with st.spinner("Training TabNet Model...", show_time=False):
                tabnet = TabNetClassifier(
                                n_d=10,  # Dimension of the decision prediction layer
                                n_a=10,  # Dimension of the attention embedding layer
                                n_steps= 5,  # Number of steps in the architecture
                                lambda_sparse=1e-3,  # Sparsity regularization
                                optimizer_params=dict(lr=5e-2),  # Optimizer params
                                mask_type='entmax',  # Can also be 'sparsemax'
                                scheduler_params={"step_size":10, "gamma":0.5},  # Learning rate scheduler
                                scheduler_fn=torch.optim.lr_scheduler.StepLR
                                )
                tabnet.fit(
                            st.session_state.X_train.values, st.session_state.y_train.values,
                            eval_set=[(st.session_state.X_test.values, st.session_state.y_test.values)],
                            eval_name=['valid'],
                            eval_metric=['accuracy'],
                            max_epochs=100,  # Maximum number of epochs
                            patience=10,  # Early stopping patience
                            batch_size=128,  # Mini-batch size
                            virtual_batch_size=128,  # Virtual batch size
                            num_workers=0,
                            drop_last=False  # Drop last batch if it is incomplete
                            )
                st.session_state.tabnet = tabnet
            
            #XGBoost
            with st.spinner("Training XGBoost Model...", show_time=False):
                xgboost = xgb.XGBClassifier(
                  objective='multi:softmax',
                  num_class=4,
                  max_depth=25,
                  learning_rate=0.001,
                  n_estimators=100,
                )
                xgboost.fit( st.session_state.X_train,  st.session_state.y_train)
                st.write("test")
                st.session_state.xgboost = xgboost
            

                
                
    except Exception as e:
        st.error(f"Error reading file: {e}")

with st.spinner("Please Wait...", show_time=False):
    load_data()

df = st.session_state.data
st.write("Data preview:")
st.dataframe(df.head(), hide_index=True)  # Show a preview of the CSV data

category_counts = df['kategori'].value_counts()

# Count Plot
fig, ax = plt.subplots()
category_counts.plot(kind='bar', ax=ax, color='skyblue')

for p in ax.patches:
    count = int(p.get_height())
    ax.text(p.get_x() + p.get_width() / 2, p.get_height() + 1, 
            str(count), ha='center', va='bottom')

ax.set_title("Distribution of Air Quality Categories")
ax.set_xlabel("Category")
ax.set_ylabel("Count")
st.pyplot(fig)

explanation_text = explain_countplot(category_counts)
st.write(explanation_text)

# Box Plot
order = ['BAIK', 'SEDANG', 'TIDAK SEHAT', 'SANGAT TIDAK SEHAT']

df_long = pd.melt(df, id_vars='kategori', value_vars=['no2', 'so2', 'o3', 'pm10', 'co'],
                  var_name='pollutant', value_name='value')

fig, ax = plt.subplots()
sns.boxplot(x='pollutant', y='value', hue='kategori', data=df_long, hue_order=order)
ax.set_title('Boxplot Polutan berdasarkan Kategori')
st.pyplot(fig)
st.write("kita bisa membandingkan distribusi antar kelompok. Ini bisa menunjukkan apakah ada perbedaan yang signifikan antara kelompok-kelompok tersebut, misalnya perbedaan dalam median, rentang, atau adanya outlier ")

#Heatmap
numeric_df = df.select_dtypes(include='number')
corr_matrix = numeric_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
plt.title("Heatmap Korelasi Antar Variabel Numerik")
plt.tight_layout()
st.pyplot(plt)

explain_heatmap(corr_matrix)


##analisis model
labels = [0, 1, 2, 3]
target_names = ['BAIK', 'SEDANG', 'TIDAK SEHAT', 'SANGAT TIDAK SEHAT']

#Ordinal Logistic Regression
st.title("Ordinal Logistic Regression")
y_pred = st.session_state.ordinal_logistic.predict(st.session_state.X_test)
y_pred = np.argmax(y_pred, axis=1)
result_test = classification_report(st.session_state.y_test, y_pred, digits=4,
                                    labels=labels, target_names=target_names, output_dict=True)
metrics_df = pd.DataFrame({
    "Precision": [result_test['macro avg']['precision']],
    "Recall": result_test['macro avg']['recall'],
    "f1-score": result_test['macro avg']['f1-score'],
    "Accuracy": result_test['accuracy']
    })
st.dataframe(metrics_df, hide_index=True)

st.markdown("<b><u>Uji Signifikansi Serentak</u></b>", unsafe_allow_html=True)

st.markdown("<b><u>Uji Signifikansi Partial</u></b>", unsafe_allow_html=True)

st.markdown("<b><u>Tes VIF</u></b>", unsafe_allow_html=True)

#Naive Bayes
st.title("Gaussian Naive Bayes")
y_pred = st.session_state.naive_bayes.predict(st.session_state.X_test)
result_test = classification_report(st.session_state.y_test, y_pred, digits=4,
                                    labels=labels, target_names=target_names, output_dict=True)
metrics_df = pd.DataFrame({
    "Precision": [result_test['macro avg']['precision']],
    "Recall": result_test['macro avg']['recall'],
    "f1-score": result_test['macro avg']['f1-score'],
    "Accuracy": result_test['accuracy']
    })
st.dataframe(metrics_df, hide_index=True)

#Gaussian Distribution plot
num_classes = len(np.unique(st.session_state.y_test)) 

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for feature_index in range(X_train.shape[1]):
    ax = axes[feature_index]
    feature_name = target_names[feature_index]
    
    x_vals = np.linspace(st.session_state.X_test.iloc[:, feature_index].min(), st.session_state.X_test.iloc[:, feature_index].max(), 200)

    for cls in range(num_classes):
        mean = gnb.theta_[cls, feature_index]
        std = np.sqrt(gnb.var_[cls, feature_index]) 
        y_vals = norm.pdf(x_vals, mean, std) 
        ax.plot(x_vals, y_vals, label=f"Class {cls}")


    ax.set_title(f"Gaussian Distribution - {feature_name}")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Probability Density")
    ax.grid(True)
    ax.legend(fontsize='small')

axes[-1].axis('off')
plt.tight_layout()
st.pyplot(plt)
st.write(""""Gaussian distribution plot yang menggunakan PDF (Probability Density Function) untuk 
menunjukkan kurva distribusi probabilitas dari setiap kelas, yang digunakan untuk menghitung 
kemungkinan kelas berdasarkan data input yang diberikan.""")

#XGBoost
st.title("XGBoost")
y_pred = st.session_state.xgboost.predict(st.session_state.X_test)
result_test = classification_report(st.session_state.y_test, y_pred, digits=4,
                                    labels=labels, target_names=target_names, output_dict=True)
metrics_df = pd.DataFrame({
    "Precision": [result_test['macro avg']['precision']],
    "Recall": result_test['macro avg']['recall'],
    "f1-score": result_test['macro avg']['f1-score'],
    "Accuracy": result_test['accuracy']
    })
st.dataframe(metrics_df, hide_index=True)

st.markdown("<b><u>Feature Importance</u></b>", unsafe_allow_html=True)
#Important Feature
fig, ax = plt.subplots(figsize=(8,6))
plot_importance(st.session_state.xgboost, ax=ax)  # show top 10 features
st.pyplot(fig)


#TabNet
st.title("TabNet")
y_pred = st.session_state.tabnet.predict(st.session_state.X_test.values)
result_test = classification_report(st.session_state.y_test, y_pred, digits=4,
                                    labels=labels, target_names=target_names, output_dict=True)
metrics_df = pd.DataFrame({
    "Precision": [result_test['macro avg']['precision']],
    "Recall": result_test['macro avg']['recall'],
    "f1-score": result_test['macro avg']['f1-score'],
    "Accuracy": result_test['accuracy']
    })
st.dataframe(metrics_df, hide_index=True)


