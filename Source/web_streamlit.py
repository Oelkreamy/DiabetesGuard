import pickle 
import streamlit as st 
import pandas as pd 


Data = pickle.load(open('Diabetes_prediction.sav','rb'))

st.title('DiabetesGuard: Early Prediction of Diabetes Using Machine Learning')
st.info('DiabetesGuard is a machine learning-powered web application designed to predict the likelihood of diabetes in individuals based on various health metrics. Utilizing the Pima Indians Diabetes Database, the model provides early diabetes risk detection, helping users assess their health status and take proactive measures. The app is built using Streamlit for an interactive, easy-to-use interface, making advanced machine learning accessible to non-technical users.')
st.sidebar.header('Feature selection')

Pregnancies = st.text_input('Pregnancies')
Glucose = st.text_input('Glucose')
BloodPressure = st.text_input('BloodPressure')
SkinThickness = st.text_input('SkinThickness')
Insulin = st.text_input('Insulin')
BMI = st.text_input('BMI')
DiabetesPedigreeFunction = st.text_input('DiabetesPedigreeFunction')
Age = st.text_input('Age')


df = pd.DataFrame({'pregnancies':[Pregnancies], 'Glucose':[Glucose], 'BloodPressure':[BloodPressure], 'SkinThickness':[SkinThickness], 'Insulin':[Insulin], 'BMI':[BMI], 'DiabetesPedigreeFunction':[DiabetesPedigreeFunction],  'Age':[Age]},index=[0])

con = st.sidebar.button('confirm')

if con:
    result = Data.predict(df)
    if result == 1:
        st.sidebar.write('Diabetes is  detected')
        st.sidebar.image('https://www.shutterstock.com/image-vector/cute-blood-drop-cartoon-character-600nw-2514685229.jpg',width=200)
    else:
        st.sidebar.write('No Diabetes')
        st.sidebar.image('https://miro.medium.com/v2/resize:fit:1000/1*2Q5DgTeU4XvKRYU71F7t9Q.jpeg',width=200)


