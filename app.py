import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split


# Load the dataset (cached using st.cache)
@st.cache_data
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

    # Drop unnecessary columns from sample data
    drop_columns = ['workclass', 'fnlwgt', 'occupation', 'relationship', 'race', 'sex', 'native.country']
    df = df.drop(drop_columns, axis=1)

    # Display sample data
    st.subheader('Sample Data:')
    st.write(df.head(10))  # Displaying the first 10 rows of the dataset

    # Define selected features for input
    selected_features = ['age', 'education', 'education.num', 'marital.status', 'capital.gain', 'capital.loss', 'hours.per.week', 'income']

    # Filter the dataframe based on selected features
    df = df[selected_features]

    # Split data into train and test sets
    X = df.drop('income', axis=1)
    y = df['income']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Encode categorical features
    label_encoders = {}
    for col in X_train.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        label_encoders[col] = le

    # Load the trained model
    model_path = 'z.pkl'  # Path to saved model (replace with your model path)
    dt_model_loaded = load_model(model_path)

    # Input feature values
    input_features = {}
    for feature in X.columns:
         if feature != 'income':
            if feature in ['age', 'hours.per.week']:
                # Convert 'age' and 'hours.per.week' to int64
                if X[feature].dtype == 'object':  # Check if the column is of object type
                    # Convert values to int after user input
                    input_features[feature] = st.number_input(f'Enter {feature}', value=int(X[feature].mean()), step=1)
                elif X[feature].dtype == 'int64' or X[feature].dtype == 'float64':
                    # Use number input directly for numerical columns
                    input_features[feature] = st.number_input(f'Enter {feature}', value=X[feature].mean())
            else:
                # Keep other columns with their original data types
                if X[feature].dtype == 'object':  # Categorical feature
                    input_features[feature] = st.selectbox(f'Select {feature}', X[feature].unique())
                elif X[feature].dtype == 'int64' or X[feature].dtype == 'float64':  # Numerical feature
                    if feature == 'education.num':
                        input_features[feature] = st.number_input(f'Enter {feature}', value=int(X[feature].mean()))
                    else:
                        input_features[feature] = st.number_input(f'Enter {feature}', value=X[feature].mean())

    # Make predictions when user clicks the 'Predict Income' button
    if st.button('Predict Income'):
        # Create a DataFrame from input features
        input_data = pd.DataFrame([input_features], columns=X.columns)
        
        # Ensure input data has all required columns
        for col in X.columns:
            if col not in input_data.columns:
                input_data[col] = np.nan  # Set missing columns to NaN
        
        # Predict income
        prediction = predict_income(input_data, dt_model_loaded, label_encoders)
        st.write(f'Prediction: {prediction}')

if __name__ == '__main__':
    main()
