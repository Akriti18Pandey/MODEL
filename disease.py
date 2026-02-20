import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🩺",
    layout="centered"
)

# ---------------- HOSPITAL BACKGROUND IMAGE ----------------
page_bg = """
<style>
.stApp {
    background-image: url("https://img.freepik.com/free-photo/blur-hospital_1203-7972.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

/* Make content area look like a professional card */
.main > div {
    background-color: rgba(255, 255, 255, 0.93);
    padding: 2rem;
    border-radius: 15px;
}

/* Transparent header */
header {
    background: rgba(0,0,0,0);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🩺 Disease Prediction System")
st.markdown("### AI-Based Medical Diagnosis using Machine Learning")
st.write("Select symptoms below to predict the possible disease.")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("disease_prediction.csv")

# Feature & Target
x = df.drop("Disease", axis=1)
y = df["Disease"]

# Train Model
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

# ---------------- USER INPUT ----------------
st.subheader("🧾 Select Symptoms")

inputs = []

for col in x.columns:
    value = st.checkbox(col)
    inputs.append(1 if value else 0)

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Disease"):

    prediction = model.predict([inputs])[0]

    st.success(f"🧬 Predicted Disease: {prediction}")

    st.warning(
        "⚠️ This is an AI prediction and not a medical diagnosis. "
        "Consult a doctor for professional advice."
    )

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Developed with ❤️ by Akriti using Streamlit & Machine Learning")