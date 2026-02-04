import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# ------------------- Page Config -------------------
st.set_page_config(
    page_title="NSTI Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------- Custom Background -------------------
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1523050854058-8df90110c9f1");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
[data-testid="stSidebar"] {
    background-color: #f0f2f6;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ------------------- Sidebar -------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/NSTI_logo.png/120px-NSTI_logo.png", use_column_width=True)
st.sidebar.title("📌 Navigation")
st.sidebar.info("Use this app to analyze and predict student performance using Decision Tree Machine Learning.")

# ------------------- Main Title -------------------
st.title("🎓 NSTI Student Performance Prediction App")
st.write("Data Mining & Machine Learning using Decision Tree")

# ------------------- Load Dataset -------------------
df = pd.read_csv("student_performance.csv")

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# ------------------- Pattern Finding -------------------
st.subheader("📈 Pattern Finding (Grouped by Result)")
grouped_data = df.groupby("Result").mean()
st.dataframe(grouped_data)

# ------------------- Prepare Data -------------------
X = df[["Attendance", "StudyHours", "PreviousMarks"]]
y = df["Result"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# ------------------- Train Model -------------------
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# ------------------- Prediction on Test Data -------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("✅ Model Performance")
st.metric(label="Accuracy", value=f"{accuracy * 100:.2f} %")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# ------------------- New Student Prediction -------------------
st.subheader("🧑‍🎓 Predict New Student Result")

attendance = st.slider("Attendance (%)", 0, 100, 75)
study_hours = st.slider("Study Hours per day", 0, 10, 3)
previous_marks = st.slider("Previous Marks", 0, 100, 65)

new_student = [[attendance, study_hours, previous_marks]]

if st.button("Predict Result"):
    prediction = model.predict(new_student)
    if prediction[0] == 1:
        st.success("🎉 Prediction: Student PASS")
    else:
        st.error("❌ Prediction: Student FAIL")

# ------------------- Interactive Visualization -------------------
st.subheader("📉 Interactive Student Performance Pattern")

fig = px.scatter(
    df,
    x="Attendance",
    y="PreviousMarks",
    color="Result",
    size="StudyHours",
    hover_data=["StudyHours", "PreviousMarks", "Attendance"],
    title="Interactive Student Performance Pattern"
)
st.plotly_chart(fig, use_container_width=True)

# ------------------- Footer -------------------
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | NSTI Project")
