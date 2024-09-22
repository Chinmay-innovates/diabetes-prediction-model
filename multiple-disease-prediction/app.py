import streamlit as st
import numpy as np
import pickle
import os
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Multiple Disease Prediction",layout="wide",page_icon="🩺")

working_dir = os.path.dirname(os.path.abspath(__file__))
# models
diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes.pkl','rb'))
breast_cancer_model =  pickle.load(open(f'{working_dir}/saved_models/breast_cancer.pkl','rb'))

# variables to hold user inputs
NewBMI_Overweight=0
NewBMI_Underweight=0
NewBMI_Obesity_1=0
NewBMI_Obesity_2=0 
NewBMI_Obesity_3=0

NewInsulinScore_Normal=0 
NewInsulinScore_Abormal=0 

NewGlucose_Low=0
NewGlucose_Normal=0 
NewGlucose_Overweight=0
NewGlucose_Secret=0

with st.sidebar:
   selected = option_menu(menu_title="Multiple Disease Prediction",
               options=[
                    "Diabetes Prediction",
                    "Breast Cancer Prediction",
                    "Heart Disease Prediction",
                ],    
                menu_icon='activity', #hospital-fill , #droplet
                icons=['thermometer-half', 'gender-female', 'heart-pulse'],
                default_index=0)
if selected == 'Diabetes Prediction':

    st.title ('Diabetes Prediction Using Machine Learning')
    col1, col2,col3 = st.columns(3)
    # first column
    with col1:
        Pregnancies = st.text_input("No of Pregnancies")
        SkinThickness = st.text_input("SkinThickness value")
        DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction value")
    with col2:
        Glucose = st.text_input("Glucose Levels")
        Insulin = st.text_input("Insulin value")
        Age = st.text_input("Age")
    with col3:
        BloodPressure = st.text_input("Blood Pressure value")
        BMI = st.text_input("BMI value")
                
    diabetes_res=''
    if st.button("Diabetes Test Result"):
        if float(BMI)<=18.5:
            NewBMI_Underweight = 1
        elif 18.5 < float(BMI) <=24.9:
            pass
        elif 24.9<float(BMI)<= 29.9:
            NewBMI_Overweight = 1
        elif 29.9<float(BMI)<= 34.9:
            NewBMI_Obesity_1 = 1
        elif 34.9<float(BMI)<= 39.9:
            NewBMI_Obesity_2 = 1
        elif float(BMI)>39.9:
            NewBMI_Obesity_3 = 1
        
        if 16<=float(Insulin)<=166:
            NewInsulinScore_Normal = 1
        else:
            NewInsulinScore_Abormal = 1
        if float(Glucose)<=70:
            NewGlucose_Low = 1
        if 70<float(Glucose)<=99:
            NewGlucose_Normal = 1
        elif 99<float(Glucose)<=126:
            NewGlucose_Overweight = 1
        elif float(Glucose)>126:
            NewGlucose_Secret = 1
            user_input=[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,
                    BMI,DiabetesPedigreeFunction,Age, NewBMI_Underweight,
                    NewBMI_Overweight,NewBMI_Obesity_1,
                    NewBMI_Obesity_2,NewBMI_Obesity_3,NewInsulinScore_Normal,
                    NewInsulinScore_Abormal,
                    NewGlucose_Low,
                    NewGlucose_Normal, NewGlucose_Overweight,
                    NewGlucose_Secret,
                    ]
            user_input_as_np_arr = np.asarray(user_input) 
            input_data_reshaped = user_input_as_np_arr.reshape(1,-1)

            prediction = diabetes_model.predict(input_data_reshaped)
            print(prediction)
            
            if prediction[0] == 1:
                diabetes_res = 'You have Diabetes'
            else:
                diabetes_res = 'You do not have Diabetes'
    st.success(diabetes_res)
                
if selected == 'Breast Cancer Prediction':
    st.title ('Breast Cancer Prediction Using Machine Learning')
    col1, col2,col3 = st.columns(3)
   # First column inputs
    with col1:
        texture_mean = st.text_input("Mean Texture")
        concave_points_mean = st.text_input("Mean Concave Points")
        texture_se = st.text_input("Texture SE")
        compactness_se = st.text_input("Compactness SE")
        symmetry_se = st.text_input("Symmetry SE")
        area_worst = st.text_input("Worst Area")
        concavity_worst = st.text_input("Worst Concavity")
        fractal_dimension_worst = st.text_input("Worst Fractal Dimension")

    # Second column inputs
    with col2:
        smoothness_mean = st.text_input("Mean Smoothness")
        symmetry_mean = st.text_input("Mean Symmetry")
        area_se = st.text_input("Area SE")
        concavity_se = st.text_input("Concavity SE")
        fractal_dimension_se = st.text_input("Fractal Dimension SE")
        smoothness_worst = st.text_input("Worst Smoothness")
        concave_points_worst = st.text_input("Worst Concave Points")

    # Third column inputs
    with col3:
        compactness_mean = st.text_input("Mean Compactness")
        fractal_dimension_mean = st.text_input("Mean Fractal Dimension")
        concave_points_se = st.text_input("Concave Points SE")
        texture_worst = st.text_input("Worst Texture")
        smoothness_se = st.text_input("Smoothness SE")
        compactness_worst = st.text_input("Worst Compactness")
        symmetry_worst = st.text_input("Worst Symmetry")
        
    breast_cancer_res = '' 
    if st.button("Breast Cancer Test Result"):
        user_input = [
            texture_mean,smoothness_mean,compactness_mean,
                      concave_points_mean,symmetry_mean,
                      fractal_dimension_mean,texture_se,
                      area_se,smoothness_se,compactness_se,
                      concavity_se,concave_points_se,symmetry_se,
                      fractal_dimension_se,texture_worst,area_worst,
                      smoothness_worst,compactness_worst,concavity_worst,
                      concave_points_worst,symmetry_worst,fractal_dimension_worst
                    ]
       
        user_input = [float(x) for x in user_input]     
        user_input = [user_input] # converting into 2D array
        prediction = breast_cancer_model.predict(user_input)
    
        if prediction[0] == 0:
            breast_cancer_res = 'Breast cancer is Malignant'
        else:
            breast_cancer_res = 'Breast cancer is Benign'
        
    st.success(breast_cancer_res)

if selected == 'Heart Disease Prediction':
    pass