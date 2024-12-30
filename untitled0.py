# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 14:19:09 2024

@author: mypc
"""

import numpy as np
import pickle 
import streamlit as st

# Loading the saved model
Loaded_model = pickle.load(open('C:/Users/mypc/Documents/Diabetes Prediction/trained_model.sav', 'rb'))

# Creating a function for prediction

def diabetes_prediction(input_data):
    input_data = (1,189,60,23,846,30.1,0.398,59)

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the np array as we are predicting for one instance

    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    

    prediction = Loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0]==0):
      return 'The patient is not diabetic'

    else:
      return 'The patient is diabetic'
  
def main():
    
    st.title('Diabetes Prediction web app')
    
    # Getting input from user
    
    Pregnancies = st.text_input('Number of Pregnencies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure Value')
    SkinThickness = st.text_input('Skin Thickness Value')
    Insulin = st.text_input('Insulin Value')
    BMI = st.text_input('BMI Value')
    DiabetesPedigreeFunction = st.text_input('Diabetes pedigree function Value')
    Age = st.text_input('Age OF the person')
    
    # Code for prediction
    Diagnosis = ''
    
    # Creating a button for prediction
    if st.button('Diabeties Test Result'):
        Diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    st.success(Diagnosis)
    
if __name__=='__main__':
    main()