import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import plot_importance

from sklearn.metrics import classification_report

st.logo("assets/logo.png")

st.set_page_config(
    page_title="AirCheck",
    page_icon="assets/logo.png"
)

st.title("Data Insight")

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

st.title("Ordinal Logistic Regression")
y_pred = st.session_state.ordianl_logistic.predict(st.session_state.X_test)
y_pred = np.argmax(y_pred, axis=1)
result_test = classification_report(st.session_state.y_test, y_pred, digits=4,
                                    labels=labels, target_names=target_names)
st.text('Model Report:\n    ' +result_test)


st.title("XG Boost")
y_pred2 = st.session_state.xgboost.predict(st.session_state.X_test)
result_test2 = classification_report(st.session_state.y_test, y_pred2, digits=4,
                                    labels=labels, target_names=target_names,
                                    output_dict=True)
st.dataframe( pd.DataFrame(result_test2).transpose())

fig, ax = plt.subplots(figsize=(8,6))
plot_importance(st.session_state.xgboost, ax=ax)  # show top 10 features
st.pyplot(fig)

st.title("TabNet")
tabnet = st.session_state.tabnet
y_pred3 = tabnet.predict(st.session_state.X_test)
result_test3 = classification_report(st.session_state.y_test, y_pred3, digits=4,
                                    labels=labels, target_names=target_names,
                                    output_dict=True)
st.dataframe( pd.DataFrame(result_test3).transpose())
