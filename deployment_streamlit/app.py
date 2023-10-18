import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the model
model = joblib.load('D:\\sunbase_task\\streamlit\\xgb_classifier_model.pkl')

# Load data for feature engineering (assuming modified_data_df)
# Replace this with your actual data loading code
data_df = pd.read_csv('churn.csv')

# Streamlit app UI
st.title('Customer Churn Prediction')
st.write('Enter customer information to predict churn.')

# Collect user inputs
user_inputs = {
    'age': st.number_input('Age', min_value=18, max_value=70, value=25),
    'gender': st.radio('Gender', ['Male', 'Female']),
    'location': st.selectbox('Location', ['Chicago', 'Houston', 'Los Angeles', 'Miami', 'New York']),
    'subscription_length': st.number_input('Subscription Length (Months)', min_value=1, max_value=24, value=12),
    'monthly_bill': st.number_input('Monthly Bill', min_value=0, max_value=500, value=100),
    'total_usage': st.number_input('Total Usage (GB)', min_value=0, max_value=1000, value=200)
}

# Apply feature engineering steps
age_bins = [18, 30, 40, 50, 60, 71]
age_labels = ['18-29', '30-39', '40-49', '50-59', '60-70']
# Create a new column 'Age_Bin' based on age bins
data_df['Age_Bin'] = pd.cut(data_df['Age'], bins=age_bins, labels=age_labels, right=False)
age_bin = pd.cut([user_inputs['age']], bins=age_bins, labels=age_labels)[0]
# Encode the age bin
bins_to_labels = {
    '18-29': 0,
    '30-39': 1,
    '40-49': 2,
    '50-59': 3,
    '60-79': 4
}
encoded_age = bins_to_labels[age_bin]

# Encode the gender
gender_to_label = {
    'Female': 0,
    'Male': 1
}
encoded_gender = gender_to_label[user_inputs['gender']]

# Encode the location
location_mapping = {
    'Chicago': 0,
    'Houston': 1,
    'Los Angeles': 2,
    'Miami': 3,
    'New York': 4
}
encoded_location = location_mapping[user_inputs['location']]

# Calculate derived features based on user inputs
age_subscription_length = user_inputs['subscription_length'] / user_inputs['age']
usage_per_bill = user_inputs['total_usage'] / user_inputs['monthly_bill']
total_bill = user_inputs['subscription_length'] * user_inputs['monthly_bill']
age_length = user_inputs['age'] * user_inputs['subscription_length']
usage_per_age = user_inputs['total_usage'] / user_inputs['age']
usage_ratio_to_total_bill = user_inputs['total_usage'] / total_bill
usage_per_subscription_length = user_inputs['total_usage'] / user_inputs['subscription_length']
usage_per_bill2 = user_inputs['total_usage'] / user_inputs['monthly_bill']
billing_efficiency = user_inputs['monthly_bill'] / user_inputs['subscription_length']

location_billing_avg = data_df.groupby('Location')['Monthly_Bill'].mean()
selected_location_avg_billing = location_billing_avg[user_inputs['location']]

location_usage_avg = data_df.groupby('Location')['Total_Usage_GB'].mean()
selected_location_usage_avg = location_usage_avg[user_inputs['location']]

usage_to_subscription_length_ratio = user_inputs['total_usage'] / user_inputs['subscription_length']

age_billing_avg = data_df.groupby('Age_Bin')['Monthly_Bill'].transform('mean')
age_billing_deviation = user_inputs['monthly_bill'] - age_billing_avg

location_subscription_length_avg = data_df.groupby('Location')['Subscription_Length_Months'].mean()
selected_location_subscription_length_avg = location_subscription_length_avg[user_inputs['location']]

total_bill_per_usage = total_bill / user_inputs['total_usage']

min_age = data_df['Age'].min()
max_age = data_df['Age'].max()
normalized_age = (user_inputs['age'] - min_age) / (max_age - min_age)

# Create a dictionary with input features
new_data = {
    'Age': user_inputs['age'],
    'Gender': encoded_gender,
    'Location': encoded_location,
    'Subscription_Length_Months': user_inputs['subscription_length'],
    'Monthly_Bill': user_inputs['monthly_bill'],
    'Total_Usage_GB': user_inputs['total_usage'],
    'Age_Bin': encoded_age,
    'Subscription_Length_Ratio_age': age_subscription_length,
    'Usage_per_Billing': usage_per_bill,
    'Total_Bill': total_bill,
    'Age_Subscription_Length': age_length,
    'Usage_per_Age': usage_per_age,
    'Usage_Ratio_to_Total_Bill': usage_ratio_to_total_bill,
    'Usage_per_Subscription_Length': usage_per_subscription_length,
    'Usage_per_Bill': usage_per_bill2,
    'Billing_Efficiency': billing_efficiency,
    'Location_Billing_Avg': selected_location_avg_billing,
    'Location_Usage_Avg': selected_location_usage_avg,
    'Usage_to_Subscription_Length_Ratio': usage_to_subscription_length_ratio,
    'Age_Billing_Deviation': age_billing_deviation,
    'Location_Subscription_Length_Avg': selected_location_subscription_length_avg,
    'Total_Bill_per_Usage': total_bill_per_usage,
    'Normalized_Age': normalized_age
}

# Create and fit StandardScaler
scaler = StandardScaler()

# Initialize the StandardScaler
scaler = StandardScaler()

# Extract numerical values from the dictionary
numerical_data = [value for value in new_data.values() if isinstance(value, (int, float))]

# Convert to numpy array and reshape
numerical_data = np.array(numerical_data).reshape(-1, 1)

# Create and fit StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_data)

# Assign scaled values back to the dictionary
for key, scaled_value in zip(new_data.keys(), scaled_data):
    new_data[key] = scaled_value[0]

input_list = [list(new_data.values())][0]
input_data_as_numpy_array = np.asarray(input_list)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Create a button to predict
if st.button('Predict Churn'):
    prediction = model.predict(input_data_reshaped)[0]
    st.write('Churn Prediction:', 'Churn' if prediction == 1 else 'No Churn')