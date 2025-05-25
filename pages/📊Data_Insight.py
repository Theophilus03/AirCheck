import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import plot_importance

from sklearn.metrics import classification_report

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
                
                
    except Exception as e:
        st.error(f"Error reading file: {e}")

df = st.session_state.data
st.write("Data preview:")
st.write(df.head())  # Show a preview of the CSV data

category_counts = df['kategori'].value_counts()

# Display as a bar chart using matplotlib
fig, ax = plt.subplots()
category_counts.plot(kind='bar', ax=ax, color='skyblue')
ax.set_title("Distribution of Air Quality Categories")
ax.set_xlabel("Category")
ax.set_ylabel("Count")
st.pyplot(fig)

# Display Box Plot
order = ['BAIK', 'SEDANG', 'TIDAK SEHAT', 'SANGAT TIDAK SEHAT']

df_long = pd.melt(df, id_vars='kategori', value_vars=['no2', 'so2', 'o3', 'pm10', 'co'],
                  var_name='pollutant', value_name='value')

fig, ax = plt.subplots()
sns.boxplot(x='pollutant', y='value', hue='kategori', data=df_long, hue_order=order)
ax.set_title('Boxplot Polutan berdasarkan Kategori')
st.pyplot(fig)

#analisis model
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

metrics_df.index = [''] * len(metrics_df)
st.table(metrics_df)

#Naive Bayes
st.title("Naive Bayes")
y_pred = st.session_state.naive_bayes.predict(st.session_state.X_test)
result_test = classification_report(st.session_state.y_test, y_pred, digits=4,
                                    labels=labels, target_names=target_names, output_dict=True)
metrics_df = pd.DataFrame({
    "Precision": [result_test['macro avg']['precision']],
    "Recall": result_test['macro avg']['recall'],
    "f1-score": result_test['macro avg']['f1-score'],
    "Accuracy": result_test['accuracy']
    })
metrics_df = metrics_df.reset_index(drop=True)
st.dataframe(metrics_df)  

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
metrics_df = metrics_df.reset_index(drop=True)
st.markdown(metrics_df.style.hide(axis="index").to_html(), unsafe_allow_html=True)

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
metrics_df = metrics_df.reset_index(drop=True)
st.dataframe(metrics_df)  




# Inject custom JavaScript to hide the index column
hide_index_js = """
<script>
    const tables = window.parent.document.querySelectorAll('table');
    tables.forEach(table => {
        const indexColumn = table.querySelector('thead th:first-child');
        if (indexColumn) {
            indexColumn.style.display = 'none';
        }
        const indexCells = table.querySelectorAll('tbody th');
        indexCells.forEach(cell => {
            cell.style.display = 'none';
        });
    });
</script>
"""

# Use components.html to inject the JavaScript
st.components.v1.html(hide_index_js, height=0)
