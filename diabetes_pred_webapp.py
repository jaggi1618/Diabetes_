import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm

testmodel=pickle.load(open("Svm_model.sav","rb"))

def diabetes_pred(input_data):
    input_np=np.asarray(input_data)
    input_re=input_np.reshape(1,-1)
    # std_=sc.transform(input_re)
    pred=testmodel.predict(input_re)
    print(pred)
    if (pred[0]==1):
        return 'diabetes confirmed!!!'
    else:
        return 'No diabetes confirmed !!!'

def main():
    st.title('DIABETIES PREDICTIVE TEST') 
    
    Pregnancies=st.text_input('number of pregnancies')
    Glucose=st.text_input('Glucose level')
    BloodPressure=st.text_input('blood pressure level')
    SkinThickness=st.text_input('skinthickness value')
    Insulin=st.text_input('insulin level in your body')
    BMI=st.text_input('BMI value of your body')
    DiabetesPedigreeFunction=st.text_input('diabetes pedigree function')
    Age=st.text_input('enter your age')
    diagnosis=''
    if st.button('diabetes test result'):
        diagnosis=diabetes_pred([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    st.success(diagnosis)

if __name__== '__main__':
    main()
