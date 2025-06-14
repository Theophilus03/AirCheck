import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

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
    
    return df_clean
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
            st.session_state.data2 = split_data(df)
            st.success("Data uploaded and validated successfully!")
                         
    except Exception as e:
        st.error(f"Error reading file: {e}")

    
with st.spinner("Loading Data...", show_time=False):
    load_data()

if uploaded_file is not None:
    df = st.session_state.data2
else:
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
st.header("Ordinal Logistic Regression")
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

st.markdown("#### - Uji Signifikansi Serentak (Uji Likelihood Ratio)")
lr_stat = 2 * (st.session_state.ordinal_logistic.llf - st.session_state.ordinal_logistic.llnull)
df_diff = st.session_state.ordinal_logistic.df_model
p_value = stats.chi2.sf(lr_stat, df_diff)
lr_test_df = pd.DataFrame({
    'Likelihood Ratio Statistic': [lr_stat],
    'p-value': [p_value]
})
st.dataframe(lr_test_df, hide_index=True)
st.write("""Nilai p-value sebesar 0 yang lebih kecil dari 0.05 dalam uji statistik Likelihood Ratio menunjukkan bahwa 
ada bukti yang cukup kuat untuk menolak hipotesis nol yang artinya setidaknya ada satu parameter 
dalam model yang berpengaruh secara signifikan.""")

st.markdown("#### - Uji Signifikansi Partial (Uji Wald)")
p_vals = st.session_state.ordinal_logistic.pvalues
partial_test_df = pd.DataFrame({
    'Variable': p_vals.index,
    'p-value': p_vals.round(4)
})
partial_test_df = partial_test_df.iloc[:-3]
st.dataframe(partial_test_df, hide_index=True)
st.write("""Nilai p-value yang lebih kecil dari 0,05 dalam uji statistik Wald menunjukkan bahwa terdapat bukti yang 
cukup kuat untuk menolak hipotesis nol, yang berarti variabel tersebut berpengaruh secara signifikan. Berdasarkan 
hasil uji, hanya variabel CO yang tidak berpengaruh secara signifikan.""")


st.markdown("#### - Uji Multikolinearitas (Uji VIF)")
vif_data = pd.DataFrame()
X2 = sm.add_constant(st.session_state.X_train)

vif_data["Variable"] = X2.columns
vif_data["VIF"] = [variance_inflation_factor(X2.values, i) for i in range(X2.shape[1])]
vif_data = vif_data[vif_data["Variable"] != "const"]
st.dataframe(vif_data, hide_index=True)
st.write("""Nilai VIF yang lebih besar dari 10 mengindikasikan terjadinya multikolinearitas. 
Berdasarkan hasil uji, tidak ada variabel yang mengindikasikan terjadinya multikolinearitas""")

#Naive Bayes
st.header("Gaussian Naive Bayes")
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
st.markdown("#### - Gaussian Distribution Plot")
num_classes = len(np.unique(st.session_state.y_test)) 
variable = st.session_state.X_test.columns.values

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
gnb = st.session_state.naive_bayes
for feature_index in range(st.session_state.X_test.shape[1]):
    ax = axes[feature_index]
    feature_name = variable[feature_index]
    
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
st.write("""Gaussian distribution plot yang menggunakan PDF (Probability Density Function) untuk 
menunjukkan kurva distribusi probabilitas dari setiap kelas, yang digunakan untuk menghitung 
kemungkinan kelas berdasarkan data input yang diberikan.""")

#XGBoost
st.header("XGBoost")
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
#Important Feature
st.markdown("#### - Feature Importances")
fig, ax = plt.subplots(figsize=(8,6))
plot_importance(st.session_state.xgboost, ax=ax, title="XGBoost Feature Importances")
st.pyplot(fig)
st.write("""XGBoost memberikan bobot tertinggi pada konsentrasi sulfur dioksida dan ozon 
untuk memprediksi target, dengan nitrogen dioksida, karbon monoksida, dan partikel debu (PM10) 
memberikan kontribusi yang lebih kecil, tetapi tetap penting dalam prediksi keseluruhan.""")

#TabNet
st.header("TabNet")
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


#feature importnce
st.markdown("#### - Feature Importances")
explainability_matrix, masks = st.session_state.tabnet.explain(st.session_state.X_train.values)
feat_importances_loaded = explainability_matrix.sum(axis=0)
feat_importances_loaded = feat_importances_loaded / feat_importances_loaded.sum()
indices = np.argsort(feat_importances_loaded)

fig, ax = plt.subplots(figsize=(8,6))
plt.title("TabNet Feature Importances")
plt.barh(range(len(feat_importances_loaded)), feat_importances_loaded[indices], color="b", align="center")
features = list(st.session_state.X_test.columns) # Use the original feature names
plt.yticks(range(len(feat_importances_loaded)), [features[idx] for idx in indices])
st.pyplot(plt)

st.write("""TabNet memberikan bobot tertinggi pada konsentrasi partikel debu (PM10) dalam memprediksi target, 
diikuti oleh sulfur dioksida, nitrogen dioksida, dan ozon yang memberikan kontribusi yang cukup signifikan. 
Sayangnya, karbon monoksida tidak memberikan pengaruh yang berarti dalam model ini.""")
