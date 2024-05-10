import numpy as np
import pickle
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

# Load the trained model
model_load = None

with open('AdaBoostClassifier.pkl', 'rb') as file:
    model_load = pickle.load(file)

# Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=12)

# Function to predict using the loaded model

pickle_urls = [
    'AdaBoostClassifier',
    'BaggingClassifier',
    'DecisionTreeClassifier',
    'GaussianNB',
    'GradientBoostingClassifier',
    'KNeighborsClassifier',
    'LogisticRegression',
    'MLPClassifier',
    'RandomForestClassifier',
    'SGDClassifier',
    'SVC',
    'XGBClassifier'
]


def predict(features, model):
    if model is None:
        model = model_load

    if model == 'AdaBoostClassifier':
        model = pickle.load(open('AdaBoostClassifier.pkl', 'rb'))

    elif model == 'BaggingClassifier':
        model = pickle.load(open('BaggingClassifier.pkl', 'rb'))

    elif model == 'DecisionTreeClassifier':
        model = pickle.load(open('DecisionTreeClassifier.pkl', 'rb'))

    elif model == 'GaussianNB':
        model = pickle.load(open('GaussianNB.pkl', 'rb'))

    elif model == 'GradientBoostingClassifier':
        model = pickle.load(open('GradientBoostingClassifier.pkl', 'rb'))

    elif model == 'KNeighborsClassifier':
        model = pickle.load(open('KNeighborsClassifier.pkl', 'rb'))

    elif model == 'LogisticRegression':
        model = pickle.load(open('LogisticRegression.pkl', 'rb'))

    elif model == 'MLPClassifier':
        model = pickle.load(open('MLPClassifier.pkl', 'rb'))

    elif model == 'RandomForestClassifier':
        model = pickle.load(open('RandomForestClassifier.pkl', 'rb'))

    elif model == 'SGDClassifier':
        model = pickle.load(open('SGDClassifier.pkl', 'rb'))

    elif model == 'SVC':
        model = pickle.load(open('SVC.pkl', 'rb'))

    elif model == 'XGBClassifier':
        model = pickle.load(open('XGBClassifier.pkl', 'rb'))

    prediction = model.predict(features)
    return prediction


def main():
    # Set up the title and layout
    st.title("Autism Diagnosis Predictor")
    st.markdown("### Enter the following information:")

    option = st.sidebar.selectbox("Select type of model", pickle_urls)

    # Collect user input for prediction
    gender = st.selectbox("Gender", ["Male", "Female"])
    ethnicity = st.selectbox(
        "Ethnicity", ["Asian", "Black", "Hispanic", "White", "Other"])
    jaundice = st.selectbox("Jaundice at birth", ["Yes", "No"])
    austim = st.selectbox("Has a family member with autism", ["Yes", "No"])
    age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)

    # Predict if the user clicks the button
    if st.button("Predict Autism Diagnosis"):
        # Convert categorical inputs to numerical values
        gender = 1 if gender == "Male" else 0
        ethnicity = ['Asian', 'Black', 'Hispanic',
                     'White', 'Other'].index(ethnicity)
        jaundice = 1 if jaundice == "Yes" else 0
        austim = 1 if austim == "Yes" else 0
        # used_app_before = 1 if used_app_before == "Yes" else 0

        # Create a feature array
        features = np.array(
            [[gender, ethnicity, jaundice, austim, age]])

        # Make prediction
        result = predict(features, model=option)

        # Display result
        if result[0] == 1:
            st.error("The predicted diagnosis is: Autism")
        else:
            st.success(f"The predicted diagnosis is: No Autism")


main()
