
import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

model = joblib.load('2602227775.pkl')

def main():
    st.title('UTS Model Deployment')

    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input('Credit Score', min_value=0, max_value=850)
        age = st.number_input('Age', min_value=18, max_value=100)
        tenure = st.selectbox('Tenure', ['0','1','2','3','4','5','6','7','8','9','10'])  
        num_of_products = st.number_input('Number of Products', min_value=1, max_value=4, step=1)

    with col2: 
        has_cr_card = st.selectbox('Has Credit Card?', ['Yes', 'No']) 
        is_active_member = st.selectbox('Is Active Member?', ['Yes', 'No'])  
    
    balance = st.slider('Balance', min_value=0.0, max_value=238387.56)
    estimated_salary = st.slider('Estimated Salary', min_value=0.0, max_value=199992.48)
    col3, col4 = st.columns(2)

    with col3:
        geography = st.radio('Geography', ['Spain', 'France', 'Germany'])
    with col4:
        gender = st.radio('Gender', ['Male', 'Female'])
    
    if st.button('Make Prediction'):
        label_encoder = LabelEncoder()
        geography_encoded = label_encoder.fit_transform([geography])[0]
        gender_encoded = label_encoder.fit_transform([gender])[0]
        tenure_encoded = label_encoder.fit_transform([tenure])[0]
        has_cr_card_encoded = label_encoder.fit_transform([has_cr_card])[0]
        is_active_member_encoded = label_encoder.fit_transform([is_active_member])[0]

        features = [
            credit_score, geography_encoded, gender_encoded, age, tenure_encoded, balance, 
            num_of_products, has_cr_card_encoded, is_active_member_encoded, estimated_salary
        ]

        result = make_prediction(features)

        if result == 0:
            output_text = "Customer not likely to churn" 
        else:
            output_text = "Customer likely to churn"

        st.success(f'{output_text}')

def make_prediction(features):
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
   # main()
   st.write('Hello world')
