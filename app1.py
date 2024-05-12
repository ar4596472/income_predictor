import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset (cached using st.cache)
@st.cache
def load_data():
    df = pd.read_csv('adult.csv', na_values='?')
    df.dropna(inplace=True)
    return df

# Preprocess input data
def preprocess_input_data(input_data, label_encoders):
    input_data_processed = input_data.copy()
    for col, le in label_encoders.items():
        input_data_processed[col] = le.transform([input_data[col]])[0]
    return input_data_processed

# Load the trained model
def load_model(model_path):
    model = joblib.load(model_path)
    return model

# Prediction function
def predict_income(input_data, model, label_encoders):
    input_data_processed = preprocess_input_data(input_data, label_encoders)
    input_data_processed = input_data_processed.values.reshape(1, -1)  # Reshape to 2D array
    prediction = model.predict(input_data_processed)[0]
    return prediction

# Main function to run the Streamlit app
def main():
    st.title('Income Prediction')
    st.write('This app predicts income based on input features.')

    # Load data
    df = load_data()

    # Define selected features for input
    selected_features = ['age', 'education', 'education.num', 'marital.status', 'capital.gain', 'capital.loss', 'hours.per.week', 'income']

    # Filter the dataframe based on selected features
    df = df[selected_features]

    # Split data into train and test sets
    X = df.drop('income', axis=1)
    y = df['income']
    X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)  # Only needed for label encoding

    # Encode categorical features
    label_encoders = {}
    for col in X_train.select_dtypes(include='object').columns:
        le = LabelEncoder()
        le.fit(X_train[col])  # Fit on X_train to avoid unseen categories in test input
        label_encoders[col] = le

    # Load the trained model
    model_path = 'z.pkl'  # Path to saved model (replace with your model path)
    dt_model_loaded = load_model(model_path)

    # Input feature values
    input_features = {}
    for feature in X.columns:
        if feature != 'income':
            if feature == 'age':
                age_option = st.select_slider('Select age', options=list(range(1, 101)), value=35)
                input_features[feature] = age_option
            elif feature == 'hours.per.week':
                hours_option = st.select_slider('Select hours per week', options=list(range(0, 101)), value=40)
                input_features[feature] = hours_option
            elif feature == 'education.num':
                education_num = st.number_input(f'Enter education.num (positive integer)', value=10, min_value=1, step=1)
                input_features[feature] = int(education_num)
            elif feature in ['capital.gain', 'capital.loss']:
                capital_value = st.number_input(f'Enter {feature} (positive number)', value=0.0, min_value=0.0, step=1.0)
                input_features[feature] = float(capital_value)
            else:
                input_features[feature] = st.selectbox(f'Select {feature}', [''] + X[feature].unique().tolist())

    # Validate input
    invalid_inputs = [feature for feature, value in input_features.items() if value == '']
    
    if invalid_inputs:
        error_message = "Invalid input. Please provide values for the following options:\n"
        for feature in invalid_inputs:
            error_message += f"- {feature}\n"
        st.error(error_message)
    else:
        # Check special condition: age between 0 to 15 and hours per week is 0
        age = input_features['age']
        hours_per_week = input_features['hours.per.week']

        if 0 <= age <= 15 and hours_per_week == 0:
            st.write("You don't earn.")
        else:
            # Make predictions when user clicks the 'Predict Income' button
            if st.button('Predict Income'):
                # Create a DataFrame from input features
                input_data = pd.DataFrame([input_features], columns=X.columns)

                # Predict income
                prediction = predict_income(input_data, dt_model_loaded, label_encoders)
                st.write(f'Prediction: {prediction}')

if __name__ == '__main__':
    main()
