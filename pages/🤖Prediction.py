
import streamlit as st
import numpy as np
import pandas as pd

st.logo("assets/logo.png")

st.set_page_config(
    page_title="AirCheck",
    page_icon="assets/logo.png"
)

def get_input_values():
    pm10 = st.slider('PM10', min_value=0, max_value=300, value=50)
    so2 = st.slider('SO2', min_value=0, max_value=300, value=50)
    co = st.slider('CO', min_value=0, max_value=300, value=50)
    o3 = st.slider('O3', min_value=0, max_value=300, value=50)
    no2 = st.slider('NO2', min_value=0, max_value=300, value=50)

    return np.array([[pm10, so2, co, o3, no2]])

#Main
def main():
    # Title
        st.title('Air Quality Predictor')
        st.subheader('Submit all variables to predict air quality')

        # Create sliders for user input
        model_input = get_input_values()
        model_prediction = 0


        # Model selection dropdown
        model_option = st.selectbox('Model Prediction', ('Ordinal Linear Regression', 'Naive Bayes','XG Boost', 'Tab Net'))

        # Predict button
        ordinal_mapping = {
            0: 'BAIK',
            1: 'SEDANG',
            2: 'TIDAK SEHAT',
            3: 'SANGAT TIDAK SEHAT'
        }
        if st.button('Predict'):
          if model_option == 'Ordinal Linear Regression':
            ordianl_logistic = st.session_state.ordinal_logistic
            y_prob = ordianl_logistic.predict(model_input)
            y_pred = np.argmax(y_prob, axis=1)
            prediction = ordinal_mapping[int(y_pred)]
              
          elif model_option == 'Naive Bayes':
            nb = st.session_state.naive_bayes
            y_pred = xgboost.predict(model_input)
            prediction = ordinal_mapping[int(y_pred[0])]

          elif model_option == 'XG Boost':
            xgboost = st.session_state.xgboost
            y_pred = xgboost.predict(model_input)
            prediction = ordinal_mapping[int(y_pred[0])]

          elif model_option == 'Tab Net':
            tabnet = st.session_state.tabnet
            y_pred = tabnet.predict(model_input)
            prediction = ordinal_mapping[int(y_pred[0])]

          if prediction:
            if prediction=='BAIK':
              st.success(f"Predicted Air Quality: {prediction}" )
            elif prediction=='SEDANG':
              st.info(f"Predicted Air Quality: {prediction}" )
            elif prediction=='TIDAK SEHAT':
              st.warning(f"Predicted Air Quality: {prediction}" )
            elif prediction=='SANGAT TIDAK SEHAT':
              st.error(f"Predicted Air Quality: {prediction}" )


if __name__ == '__main__':
    main()
