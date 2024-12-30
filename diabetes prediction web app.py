# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

# Loading the saved model
Loaded_model = pickle.load(open('C:/Users/mypc/Documents/Diabetes Prediction/trained_model.sav', 'rb'))

input_data = (1,189,60,23,846,30.1,0.398,59)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the np array as we are predicting for one instance

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#standardize the input
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = Loaded_model.predict(std_data)
print(prediction)

if (prediction[0]==0):
  print('The patient is not diabetic')

else:
  print('The patient is diabetic')