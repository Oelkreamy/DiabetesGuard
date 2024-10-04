import pickle
import streamlit as st
import pandas as pd

# Load the trained model
model = pickle.load(open('Diabetes_prediction.sav', 'rb'))

# Load the scaler
scaler = pickle.load(open('scaler.sav', 'rb'))

# Page Title and Info
st.title('DiabetesGuard: Early Prediction of Diabetes Using Machine Learning')
st.info("This app predicts the likelihood of diabetes using health metrics based on the Pima Indians Diabetes Dataset. Please enter the required values in the fields below.")

# Sidebar for feature input
st.sidebar.header('Input Features')

# Use number inputs and sliders for more intuitive user interaction
Pregnancies = st.sidebar.number_input('Number of Pregnancies', min_value=0, max_value=20, step=1, value=1)
Glucose = st.sidebar.slider('Glucose Level', min_value=0, max_value=200, value=110)
BloodPressure = st.sidebar.slider('Blood Pressure (mm Hg)', min_value=0, max_value=140, value=70)
SkinThickness = st.sidebar.slider('Skin Thickness (mm)', min_value=0, max_value=100, value=20)
Insulin = st.sidebar.slider('Insulin Level (mu U/ml)', min_value=0, max_value=900, value=80)
BMI = st.sidebar.number_input('BMI (kg/m²)', min_value=0.0, max_value=70.0, value=25.0, step=0.1)
DiabetesPedigreeFunction = st.sidebar.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.5, step=0.01)
Age = st.sidebar.number_input('Age', min_value=0, max_value=120, step=1, value=25)

# Button for making predictions
confirm = st.sidebar.button('Predict')

# Prepare DataFrame for prediction
input_data = pd.DataFrame({
    'Pregnancies': [Pregnancies], 
    'Glucose': [Glucose], 
    'BloodPressure': [BloodPressure], 
    'SkinThickness': [SkinThickness], 
    'Insulin': [Insulin], 
    'BMI': [BMI], 
    'DiabetesPedigreeFunction': [DiabetesPedigreeFunction], 
    'Age': [Age]
})

input_data_scaled = scaler.transform(input_data)

# Make prediction and show result
if confirm:
    with st.spinner('Predicting...'):
        prediction = model.predict(input_data_scaled)

    if prediction[0] == 1:
        st.success('🚨 **Diabetes Detected** 🚨')
        st.image('https://www.shutterstock.com/image-vector/cute-blood-drop-cartoon-character-600nw-2514685229.jpg', width=200)
    else:
        st.success('🎉 **No Diabetes Detected** 🎉')
        st.image('https://miro.medium.com/v2/resize:fit:1000/1*2Q5DgTeU4XvKRYU71F7t9Q.jpeg', width=200)

# Optional Footer with Information
st.sidebar.markdown("Developed with ❤️ using Streamlit and Scikit-learn.")


