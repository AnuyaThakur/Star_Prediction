from fastapi import FastAPI
import pickle
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
app=FastAPI()

df=pd.read_csv("Star_encode.csv",header=0) #Location of the encoded file
print(df.shape)
print(df.head())
# Splitting the Data for Train test split
X=df.drop("Type",axis=1) #Feature Variables
y=df["Type"] #Target Variable

# Train test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=1)

# Building the Model using Random Forest Classifier (We had good accuracy thus we are using Random Forest Classifier)
rfc=RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred=rfc.predict(X_test)

#Testing accuracy of Random Forest Classifier
from sklearn.metrics import confusion_matrix, accuracy_score
cm1=confusion_matrix(y_test, y_pred)
ac1=accuracy_score(y_test, y_pred)
print(cm1,ac1)
ac_per=ac1*100

#Pickle is used for saving
pickle.dump(rfc,open('star.pkl','wb')) # wb: Write Binary so the file is not readable, Model: Name of the file
loaded_model=pickle.load(open('star.pkl','rb'))


#For deploying on the Website
def predict_input_page():
    loaded_model = pickle.load(open('star.pkl', 'rb'))
    st.title("Star Type Prediction System")
    Temperature=st.text_input("Temperature of the Star: ",0)
    L = st.text_input("Luminosity of the Star with Respect to Sun: ",0)
    R=st.text_input("Radius of the Star with respect to Sun: ",0)
    A_M = st.text_input("Absolute Magnitude of the Star: ", 0)
    Color=st.selectbox("Color of the Star: ",('Red', 'Blue White', 'White', 'Yellowish White', 'Yellowish', 'Blue', 'Orange'))
    Spectral_Class=st.selectbox("Spectral class of the Star",("M","B","O","A","F","K","G"))
    ok=st.button("Predict the Type")

    if Color == "Red" :
        Color = 3
    elif Color == "Blue White":
        Color = 1
    elif Color == "White":
        Color = 4
    elif Color == "Yellowish White":
        Color = 6
    elif Color == "Yellowish":
        Color = 5
    elif Color == "Blue":
        Color = 0
    elif Color == "Orange":
        Color = 2
    
    if Spectral_Class == "M":
        Spectral_Class = 5
    elif Spectral_Class == "B":
        Spectral_Class = 1
    elif Spectral_Class == "A":
        Spectral_Class = 0
    elif Spectral_Class == "F":
        Spectral_Class = 2
    elif Spectral_Class == "O":
        Spectral_Class = 6
    elif Spectral_Class == "K":
        Spectral_Class = 4
    elif Spectral_Class == "G":
        Spectral_Class = 3
    


    testdata=np.array([[Temperature, L, R, A_M, Color, Spectral_Class]])
    classi=loaded_model.predict(testdata)[0]
    try:
        if ok==True:
            if classi == 0:
                st.info("Red Dwarf")
            elif classi == 1:
                st.info("Brown Dwarf")
            elif classi ==2:
                st.info("White Dwarf")
            elif classi == 3:
                st.info("Main Sequence")
            elif classi == 4:
                st.info("Super Giant")
            elif classi == 5:
                st.info("Hyper Giant")
    except:
        st.info("Enter some Data")

