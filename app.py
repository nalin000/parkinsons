import streamlit as st
import numpy as np
import pickle

# Load the model and scaler
model = pickle.load(open('parkinsons_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("🧠 Parkinson's Disease Prediction App")
st.write("Enter voice measurements to check if the person is likely to have Parkinson's Disease:")

# Input fields for all 22 features
features = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
    'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
    'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
    'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
]

input_values = []

# Create input boxes
for feature in features:
    value = st.text_input(f"{feature}")
    input_values.append(value)

# Predict button
if st.button("Predict"):
    try:
        # Convert to numpy array and reshape
        input_data = np.array([float(val) for val in input_values]).reshape(1, -1)
        std_data = scaler.transform(input_data)
        prediction = model.predict(std_data)

        if prediction[0] == 1:
            st.error("🧪 The model predicts: Parkinson's Disease.")
        else:
            st.success("✅ The model predicts: No Parkinson's Disease.")
    except:
        st.warning("⚠️ Please enter valid numeric values for all fields.")
