import streamlit as st
import base64

def image_to_base64(img_path):
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
st.logo("assets/logo.png")

st.set_page_config(
    page_title="AirCheck",
    page_icon="assets/logo.png"
)

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
     Kami menyediakan analisis data historis terkait konsentrasi polutan seperti PM10, CO, NO2, O3, dan SO2. Dengan data visual, Anda dapat memahami pola, tren, dan korelasi antar polutan. Anda juga bisa mengupload dataset anda sendiri untuk mendapatkan data insight dan prediksi yang lebih tepat untuk daerah anda.

    - **Air Quality Predictor**
     Kami menggunakan berbagai model statistik dan machine learning untuk memprediksi kualitas udara berdasarkan data polutan.
    """
)

st.subheader("Data Source?")
st.markdown(
    """
Data kualitas udara diambil dari Provinsi DKI Jakarta dengan sampel yang digunakan mencakup data dari tahun 2018 hingga 2024, 
yang diperoleh melalui teknik non-probability sampling (convenience sampling), dengan total data sebanyak 7,076 yang menggunakan data sekunder yang diperoleh dari situs Satu Data Indonesia. 
Data sekunder merupakan data yang didapatkan secara tidak langsung dari objek penelitian yang dimana data tersebut didapatkan dari sebuah situs internet ataupun sebuah referensi.
    """
)
