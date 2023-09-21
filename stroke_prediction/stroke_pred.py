import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the KNN model
model = joblib.load('stroke_model.pkl')

# Load the scaler for age, avg_glucose_level, and bmi
scaler = StandardScaler()
#scaler.fit(your_data_here)  # You should fit the scaler on the training data

# Streamlit app
def main():
    st.title('Stroke Prediction App')
    
    # Input features
    gender = st.selectbox('Gender', ['Male', 'Female'])
    hypertension = st.selectbox('Hypertension', [0, 1])
    heart_disease = st.selectbox('Heart Disease', [0, 1])
    ever_married = st.selectbox('Ever Married', [0, 1])
    residence_type = st.selectbox('Residence Type', ['Urban', 'Rural'])
    smoking_status = st.selectbox('Smoking Status', ['never smoked', 'formerly smoked', 'smokes'])
    age = st.number_input('Age', value=30)
    avg_glucose_level = st.number_input('Average Glucose Level', value=100)
    bmi = st.number_input('BMI', value=25)
    
    # Convert categorical features to numerical
    work_type_encoded = 0  # Assuming not used in this version
    residence_type_encoded = 0 if residence_type == 'Urban' else 1
    smoking_status_encoded = 0 if smoking_status == 'never smoked' else (1 if smoking_status == 'formerly smoked' else 2)
    gender_encoded = 0 if gender == 'Male' else 1
    
    # Scale the input features
    # Scale the input features
    age_scaled = scaler.fit_transform([[age]])[0][0]
    avg_glucose_level_scaled = scaler.fit_transform([[avg_glucose_level]])[0][0]
    bmi_scaled = scaler.fit_transform([[bmi]])[0][0]

    
    # Prepare input data
    input_data = np.array([[gender_encoded, hypertension, heart_disease, ever_married,
                            residence_type_encoded, smoking_status_encoded,
                            age_scaled, avg_glucose_level_scaled, bmi_scaled]])
    
    if st.button('Predict'):
        prediction = model.predict(input_data)
            
      
    # Display prediction
        if prediction[0] == 0:
            st.write('Prediction: No Stroke')
        else:
            st.write('Prediction: Stroke')
        
if __name__ == '__main__':
    main()
