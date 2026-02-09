# save this as app.py and run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("HR_comma_sep.csv")
    df_processed = pd.get_dummies(df, drop_first=True)
    X = df_processed.drop("left", axis=1)
    y = df_processed["left"]
    return X, y, df

X, y, df = load_data()

# Train model (with SMOTE)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=123)
smote = SMOTE(random_state=123)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

model = GradientBoostingClassifier()
model.fit(X_train_res, y_train_res)

# Streamlit UI
st.title("Employee Turnover Prediction App")
st.write("Predict the probability of an employee leaving the company.")

# Input fields
satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
last_evaluation = st.slider("Last Evaluation Score", 0.0, 1.0, 0.5)
number_project = st.number_input("Number of Projects", min_value=1, max_value=10, value=3)
average_montly_hours = st.number_input("Average Monthly Hours", min_value=50, max_value=400, value=200)
time_spend_company = st.number_input("Years at Company", min_value=1, max_value=20, value=3)
work_accident = st.selectbox("Work Accident", [0,1])
promotion_last_5years = st.selectbox("Promotion in Last 5 Years", [0,1])
department = st.selectbox("Department", df['Department'].unique())
salary = st.selectbox("Salary Level", df['salary'].unique())

# Prepare input row
input_dict = {
    "satisfaction_level": satisfaction_level,
    "last_evaluation": last_evaluation,
    "number_project": number_project,
    "average_montly_hours": average_montly_hours,
    "time_spend_company": time_spend_company,
    "Work_accident": work_accident,
    "promotion_last_5years": promotion_last_5years,
    "Department": department,
    "salary": salary
}

input_df = pd.DataFrame([input_dict])
input_processed = pd.get_dummies(input_df, drop_first=True)

# Align with training columns
input_processed = input_processed.reindex(columns=X.columns, fill_value=0)

# Prediction
prob = model.predict_proba(input_processed)[0][1]

# Categorize risk
def categorize(score):
    if score < 0.2:
        return "Safe Zone (Green)"
    elif score < 0.6:
        return "Low-Risk Zone (Yellow)"
    elif score < 0.9:
        return "Medium-Risk Zone (Orange)"
    else:
        return "High-Risk Zone (Red)"

risk_zone = categorize(prob)

# Display results
st.subheader("Prediction Results")
st.write(f"**Turnover Probability:** {prob:.2f}")
st.write(f"**Risk Zone:** {risk_zone}")

# Pie chart example (distribution from test data)
import matplotlib.pyplot as plt

y_prob = model.predict_proba(X_test)[:,1]
zones = pd.Series(y_prob).apply(categorize)
counts = zones.value_counts()

fig, ax = plt.subplots()
ax.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=["green","yellow","orange","red"], startangle=140)
ax.set_title("Turnover Risk Distribution (Test Data)")
st.pyplot(fig)
