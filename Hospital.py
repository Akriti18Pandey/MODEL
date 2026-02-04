import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Hospital Disease Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- CUSTOM BACKGROUND --------------------
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #e0f7fa, #f1f8e9);
}
[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #004d40, #00695c);
    color: white;
}
h1, h2, h3 {
    color: #004d40;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# -------------------- APP TITLE --------------------
st.title("🏥 Hospital Disease Prediction App")
st.write("This app uses a Decision Tree Classifier to predict whether a patient has a disease based on medical parameters.")

# -------------------- LOAD DATA --------------------
uploaded_file = st.file_uploader("Upload Hospital Data CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # Grouped statistics
    st.subheader("📈 Average Values by Disease")
    st.write(df.groupby("Disease").mean())

    # -------------------- MODEL TRAINING --------------------
    x = df[["Age", "Fever", "BP", "Sugar"]]
    y = df["Disease"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y_enc, test_size=0.2, random_state=42
    )

    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, pred)
    st.success(f"✅ Model trained successfully with accuracy: {accuracy*100:.2f}%")

    # -------------------- USER INPUT --------------------
    st.subheader("🧑‍⚕️ Enter Patient Details for Prediction")

    age = st.number_input("Age", min_value=1, max_value=120, value=20)
    fever = st.number_input("Fever (°F)", min_value=90, max_value=110, value=98)
    bp = st.number_input("Blood Pressure (mmHg)", min_value=80, max_value=200, value=120)
    sugar = st.number_input("Sugar Level (mg/dL)", min_value=50, max_value=300, value=100)

    if st.button("🔍 Predict Disease"):
        new_data = [[age, fever, bp, sugar]]
        prediction = model.predict(new_data)

        if prediction[0] == 1:
            st.error("🩺 Prediction: NO Disease Detected")
        else:
            st.warning("🩺 Prediction: Disease Detected")

else:
    st.info("📂 Please upload a Hospital_data.csv file to proceed.")