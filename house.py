import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ---------- Page Config ----------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ---------- Background Image ----------
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1560518883-ce09059eeffa");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
[data-testid="stToolbar"] {
    right: 2rem;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ---------- Title ----------
st.markdown(
    "<h1 style='text-align:center; color:white;'>🏠 House Price Prediction App</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h4 style='text-align:center; color:white;'>Predict house prices using Machine Learning</h4>",
    unsafe_allow_html=True
)

# ---------- Load Data ----------
df = pd.read_csv("house_price.csv")

# Feature selection
X = df[['Size_sqft', 'Bedrooms', 'Age_years']]
y = df['Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# ---------- Sidebar Inputs ----------
st.sidebar.header("Enter House Details")

size = st.sidebar.number_input(
    "Size (sqft)",
    min_value=100,
    max_value=10000,
    value=1500
)

bedrooms = st.sidebar.number_input(
    "Number of Bedrooms",
    min_value=1,
    max_value=10,
    value=3
)

age = st.sidebar.number_input(
    "Age of House (years)",
    min_value=0,
    max_value=100,
    value=10
)

# ---------- Prediction ----------
if st.sidebar.button("Predict Price 💰"):

    new_data = [[size, bedrooms, age]]
    prediction = model.predict(new_data)[0]

    st.markdown(
        f"""
        <div style="
            background-color: rgba(0,0,0,0.6);
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            color: white;
            font-size: 28px;">
            💰 Predicted House Price: <br>
            <b>₹ {prediction:,.2f}</b>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------- Data Preview ----------
with st.expander("📊 View Dataset"):
    st.dataframe(df)

# ---------- Footer ----------
st.markdown(
    "<p style='text-align:center; color:white;'>Made by Akriti using Streamlit</p>",
    unsafe_allow_html=True
)