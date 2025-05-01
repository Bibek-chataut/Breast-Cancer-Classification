import streamlit as st
import numpy as np
import pandas as pd

st.title("Breast Cancer Detection using KNN")

breast_cancer_data=pd.read_csv(r"C:\Users\bibek\Downloads\breast-cancer.csv")

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

breast_cancer_data.drop(columns=['id'],inplace=True)

from sklearn.preprocessing import LabelEncoder
#Encoding the target variable using LabelEncoder
label_encoder=LabelEncoder()
breast_cancer_data['diagnosis']=label_encoder.fit_transform(breast_cancer_data['diagnosis'])

#Splitting features and target
X_bc=breast_cancer_data.drop(columns=['diagnosis'])
y_bc=breast_cancer_data['diagnosis']


#Splitting into training and testing sets
X_train_bc,X_test_bc,y_train_bc,y_test_bc=train_test_split(X_bc,y_bc,test_size=0.2,random_state=42)

#Standardizing the data
scaler_bc=StandardScaler()
X_train_bc=scaler_bc.fit_transform(X_train_bc)
X_test_bc=scaler_bc.transform(X_test_bc)

#Training KNN model
knn_bc_model=KNeighborsClassifier(n_neighbors=5,metric='euclidean')
knn_bc_model.fit(X_train_bc,y_train_bc)


# Sidebar for user input
st.header("Input Features")
input_data = {}
for column in X_bc.columns:
    input_data[column] = st.number_input(column)

# Convert input data to DataFrame
input_df =pd.DataFrame([input_data])

# Standardize the input data
input_df = scaler_bc.transform(input_df)

# Predict the result
if st.sidebar.button('Predict'):
    prediction = knn_bc_model.predict(input_df)
    st.sidebar.write(f"Prediction: {'Malignant' if prediction[0] == 1 else 'Benign'}")


