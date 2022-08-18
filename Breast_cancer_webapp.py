# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 00:49:52 2022

@author: prati
"""

import numpy as np
import pandas as pd
import joblib
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

with open('style.css') as f:
    st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)



image=Image.open('photo.png')
st.image(image,caption="Predicting diagnosis results", width=(800))

# loading the saved model
ann_model = load_model('ANN.h5')
logistic_reg_model= joblib.load(open('LOGISTIC_REG.save','rb'))
decision_tree_model= joblib.load(open('D_TREE.save','rb'))
Random_forest_model= joblib.load(open('R_FOREST.save','rb'))
knn_model= joblib.load(open('KNN.save','rb'))

# loading the Scaler
loaded_scaler= joblib.load(open('StandardScaler.save','rb'))


st.subheader("Choose Classifier:")
Classifier=st.selectbox(label="Classifier",options=("ANN","Logistic regression","Decision Tree","Random Forest","KNN"))

if Classifier=="ANN":
    loaded_model=ann_model
if Classifier=="Logistic regression":
    loaded_model=logistic_reg_model
if Classifier=="Decision Tree":
    loaded_model=decision_tree_model
if Classifier=="Random Forest":
    loaded_model=Random_forest_model
if Classifier=="KNN":
    loaded_model=knn_model


def BC_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
   
    # colnames=["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"]
    # st.write(pd.DataFrame(input_data_reshaped,columns=colnames))
    
    prediction = loaded_model.predict(input_data_reshaped)
    
    
    if (prediction[0]>0.5):
      st.write('You have a malignant(cancerous) tumour')
    else:
      st.write('You have a benign(non-cancerous) tumour')
    
def main():
    
    st.subheader("Enter Your Values")
    radius_mean=st.number_input("radius_mean",format='%2f')
    texture_mean=st.number_input("texture_mean",format='%2f')
    perimeter_mean=st.number_input("perimeter_mean",format='%2f')
    area_mean=st.number_input("area_mean",format='%2f')
    smoothness_mean=st.number_input("smoothness_mean",format='%2f')
    compactness_mean=st.number_input("compactness_mean",format='%2f')
    concavity_mean=st.number_input("concavity_mean",format='%2f')
    concave_points_mean=st.number_input("concave points_mean",format='%2f')
    symmetry_mean=st.number_input("symmetry_mean",format='%2f')
    fractal_dimension_mean=st.number_input("fractal_dimension_mean",format='%2f')
    radius_se=st.number_input("radius_se",format='%2f')
    texture_se=st.number_input("texture_se",format='%2f')
    perimeter_se=st.number_input("perimeter_se",format='%2f')
    area_se=st.number_input("area_se",format='%2f')
    smoothness_se=st.number_input("smoothness_se",format='%2f')
    compactness_se=st.number_input("compactness_se",format='%2f')
    concavity_se=st.number_input("concavity_se",format='%2f')
    concave_points_se=st.number_input("concave points_se",format='%2f')
    symmetry_se=st.number_input("symmetry_se",format='%2f')
    fractal_dimension_se=st.number_input("fractal_dimension_se",format='%2f')
    radius_worst=st.number_input("radius_worst",format='%2f')
    texture_worst=st.number_input("texture_worst",format='%2f')
    perimeter_worst=st.number_input("perimeter_worst",format='%2f')
    area_worst=st.number_input("area_worst",format='%2f')
    smoothness_worst=st.number_input("smoothness_worst",format='%2f')
    compactness_worst=st.number_input("compactness_worst",format='%2f')
    concavity_worst=st.number_input("concavity_worst",format='%2f')
    concave_points_worst=st.number_input("concave points_worst",format='%2f')
    symmetry_worst=st.number_input("symmetry_worst",format='%2f')
    fractal_dimension_worst=st.number_input("fractal_dimension_worst",format='%2f')
    
    
    # code for Prediction
    
    # creating a button for Prediction
    colnames=["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"]
    if st.button('Test Result'):
        user_input=[[radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave_points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave_points_worst,symmetry_worst,fractal_dimension_worst]]
        st.subheader("Your inputs:")
        st.write(pd.DataFrame(user_input,columns=colnames))
        if (loaded_model == decision_tree_model) | (loaded_model == Random_forest_model):
            BC_prediction(user_input)
        if (loaded_model == ann_model) | (loaded_model ==  logistic_reg_model) |(loaded_model ==  knn_model):
            BC_prediction(loaded_scaler.transform(user_input))

if __name__ == '__main__':
    main()
