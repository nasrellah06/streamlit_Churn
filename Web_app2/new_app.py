import streamlit as st
import numpy as np
import joblib

st.title("Welcome to Churn")
st.write("**Veuillez remplir les champs suivants**")

model = joblib.load("knn_model.pkl")

CreditScore = st.number_input("CreditScore")
Geography = ["France", "Spain", "Germany"]
pays = st.selectbox("Sélectionnez un pays :", Geography)

Gender = ["Male", "Female"]
gender_choisi = st.selectbox("Sélectionnez votre Sexe:", Gender)

Age = st.number_input("Age")
Tenure = st.number_input("Tenure", min_value=1, max_value=10)
Balance = st.number_input("Balance")
NumOfProducts = st.number_input("NumOfProducts", min_value=1, max_value=4)

HasCrCard = [0, 1]
HasCrCard_choisi = st.selectbox("HasCrCard", HasCrCard)

IsActiveMember = [0, 1]
IsActiveMember_choisi = st.selectbox("ActiveMember", IsActiveMember)

EstimatedSalary = st.number_input("EstimatedSalary")

# One-hot encoding manuel (tout sous forme numérique)
geo_france = 1 if pays == "France" else 0
geo_spain = 1 if pays == "Spain" else 0
geo_germany = 1 if pays == "Germany" else 0

gender_female = 1 if gender_choisi == "Female" else 0
gender_male = 1 if gender_choisi == "Male" else 0

# Préparer le vecteur final (assurez-vous que l'ordre correspond au modèle)
features = np.array([[float(CreditScore),
                      float(Age),
                      float(Tenure),
                      float(Balance),
                      float(NumOfProducts),
                      int(HasCrCard_choisi),
                      int(IsActiveMember_choisi),
                      float(EstimatedSalary),
                      int(geo_france),
                      int(geo_spain),
                      int(geo_germany),
                      int(gender_female)]])  # ou gender_male selon modèle

if st.button('Submit'):
        prediction = model.predict(features)[0]
        if prediction == 1:
            st.success("Le client est **susceptible de quitter** la banque.")
        else:
            st.info("Le client **va probablement rester** dans la banque.")
