import streamlit as st
import pandas as pd
import joblib

model = joblib.load(r'C:\Users\Nirupam Gangurde\Propensify_model.pkl')
scaler = joblib.load(r'C:\Users\Nirupam Gangurde\scaler.pkl')
with open('feature_names.pkl', 'rb') as f:
    feature_names = joblib.load(f)

train_data = pd.read_excel(r'D:\ML Projects\Propensify\train.xlsx')
categorical_features = [
    'profession', 'marital', 'schooling', 'default', 'housing', 
    'loan', 'contact', 'month', 'day_of_week', 'poutcome'
]

st.title('Insurance Marketing Propensity Model')
st.write('Predict whether a customer will respond to the marketing campaign.')

age = st.number_input('Age', min_value=18, max_value=100, value=30)
profession = st.selectbox('Profession', train_data['profession'].unique())
marital = st.selectbox('Marital Status', train_data['marital'].unique())
schooling = st.selectbox('Schooling', train_data['schooling'].unique())
default = st.selectbox('Default', train_data['default'].unique())
housing = st.selectbox('Housing Loan', train_data['housing'].unique())
loan = st.selectbox('Personal Loan', train_data['loan'].unique())
contact = st.selectbox('Contact Type', train_data['contact'].unique())
month = st.selectbox('Last Contact Month', train_data['month'].unique())
day_of_week = st.selectbox('Last Contact Day of the Week', train_data['day_of_week'].unique())
campaign = st.number_input('Number of Contacts during Campaign', min_value=1, max_value=50, value=1)
pdays = st.number_input('Days since last contact', min_value=0, max_value=999, value=999)
previous = st.number_input('Number of Contacts before Campaign', min_value=0, max_value=50, value=0)
poutcome = st.selectbox('Outcome of Previous Campaign', train_data['poutcome'].unique())
emp_var_rate = st.number_input('Employment Variation Rate', value=1.1)
cons_price_idx = st.number_input('Consumer Price Index', value=93.994)
cons_conf_idx = st.number_input('Consumer Confidence Index', value=-36.4)
euribor3m = st.number_input('Euribor 3 Month Rate', value=4.857)
nr_employed = st.number_input('Number of Employees', value=5191.0)
pmonths = st.number_input('Months since last contact in previous campaign', min_value=0, max_value=999, value=999)
pastEmail = st.number_input('Number of Emails sent to client', min_value=0, max_value=50, value=0)

if st.button('Predict'):
    user_data = pd.DataFrame([{
        'custAge': age,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'emp.var.rate': emp_var_rate,
        'cons.price.idx': cons_price_idx,
        'cons.conf.idx': cons_conf_idx,
        'euribor3m': euribor3m,
        'nr.employed': nr_employed,
        'pmonths': pmonths,
        'pastEmail': pastEmail,
        'profession': profession,
        'marital': marital,
        'schooling': schooling,
        'default': default,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'month': month,
        'day_of_week': day_of_week,
        'poutcome': poutcome
    }])

    user_data = pd.get_dummies(user_data)
    user_data = user_data.reindex(columns=feature_names, fill_value=0)

    user_data = user_data.drop(columns=['id', 'responded'], errors='ignore')

    user_data_scaled = scaler.transform(user_data)

    prediction = model.predict(user_data_scaled)

    st.write('Prediction:', 'Yes' if prediction[0] else 'No')
