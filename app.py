import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import plotly.graph_objects as go

model = joblib.load("models/credit_risk_tpot.pkl")
background_data = pd.read_csv("models/background_data.csv")
explainer = shap.KernelExplainer(model.predict_proba, background_data)
feature_names = background_data.columns.tolist()

def generate_narrative(instance, shap_values, feature_names):
    shap_instance = shap_values[0, :, 1]
    narrative = "Explanation for this prediction:\n"
    for feature, value, contrib in zip(feature_names, instance, shap_instance):
        if contrib > 0:
            narrative += f"- {feature} ({value:.2f}) increased risk by {contrib:.2f}\n"
        elif contrib < 0:
            narrative += f"- {feature} ({value:.2f}) decreased risk by {-contrib:.2f}\n"
    return narrative

def waterfall_chart(shap_values, feature_names, instance):
    shap_instance = shap_values[0, :, 1]
    fig = go.Figure(go.Waterfall(
        name="Risk Contribution", orientation="v",
        x=feature_names, y=shap_instance,
        text=[f"{val:.2f}" for val in shap_instance],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    fig.update_layout(title="Feature Impact on Risk", showlegend=False)
    return fig

st.set_page_config(layout="wide")
st.markdown("""
    <style>
    .neumorphic {background: #e0e0e0; border-radius: 15px; padding: 20px; box-shadow: 5px 5px 10px #bebebe, -5px -5px 10px #ffffff;}
    .title {text-align: center; color: #333;}
    </style>
""", unsafe_allow_html=True)

st.title("Credit Risk Dashboard", anchor="neumorphic-title")
col1, col2 = st.columns([1, 2])

with col1:
    
    with st.form("input_form"):
        input_data = {}
        
        # Numeric features
        input_data["duration"] = st.slider("Loan Duration (months)", 0, 60, 12)
        input_data["credit_amount"] = st.slider("Loan Amount (Birr)", 0, 100000, 10000, format="%d Birr")
        input_data["age"] = st.slider("Age (years)", 18, 100, 30)
        input_data["installment_rate"] = st.slider("Installment Rate (% of income)", 1, 4, 2)
        input_data["residence_since"] = st.slider("Years at Current Residence", 0, 4, 2)
        input_data["num_credits"] = st.slider("Number of Existing Credits", 1, 4, 1)
        input_data["num_dependents"] = st.slider("Number of Dependents", 0, 2, 1)

        # Categorical features
        checking_map = {"No Checking": 0, "<0 Birr": 1, "0-2000 Birr": 2, ">2000 Birr": 3}
        input_data["checking_status"] = st.selectbox("Checking Account Balance", list(checking_map.keys()), index=0)
        input_data["checking_status"] = checking_map[input_data["checking_status"]]

        savings_map = {"<500 Birr": 0, "500-5000 Birr": 1, "5000-10000 Birr": 2, ">10000 Birr": 3}
        input_data["savings_status"] = st.selectbox("Savings Account Balance", list(savings_map.keys()), index=0)
        input_data["savings_status"] = savings_map[input_data["savings_status"]]

        employment_map = {"Unemployed": 0, "<1 year": 1, "1-4 years": 2, "4-7 years": 3, ">7 years": 4}
        input_data["employment"] = st.selectbox("Employment Duration", list(employment_map.keys()), index=2)
        input_data["employment"] = employment_map[input_data["employment"]]

        credit_history_map = {"Poor": 0, "Fair": 1, "Good": 2, "Excellent": 3}
        input_data["credit_history"] = st.selectbox("Credit History", list(credit_history_map.keys()), index=2)
        input_data["credit_history"] = credit_history_map[input_data["credit_history"]]

        personal_map = {"Male Single": 0, "Female Divorced/Separated/Married": 1, "Male Divorced/Separated": 2, "Male Married/Widowed": 3}
        input_data["personal_status"] = st.selectbox("Personal Status", list(personal_map.keys()), index=0)
        input_data["personal_status"] = personal_map[input_data["personal_status"]]

        debtors_map = {"None": 0, "Co-Applicant": 1, "Guarantor": 2}
        input_data["other_debtors"] = st.selectbox("Other Debtors", list(debtors_map.keys()), index=0)
        input_data["other_debtors"] = debtors_map[input_data["other_debtors"]]

        property_map = {"Real Estate": 0, "Life Insurance": 1, "Car": 2, "No Property": 3}
        input_data["property"] = st.selectbox("Property Owned", list(property_map.keys()), index=0)
        input_data["property"] = property_map[input_data["property"]]

        job_map = {"Unemployed": 0, "Unskilled": 1, "Skilled": 2, "Highly Skilled": 3}
        input_data["job"] = st.selectbox("Job Type", list(job_map.keys()), index=2)
        input_data["job"] = job_map[input_data["job"]]

        input_data["telephone"] = 1 if st.checkbox("Has Telephone", value=False) else 0
        input_data["foreign_worker"] = 1 if st.checkbox("Foreign Worker", value=False) else 0

        # Purpose features
        purposes = ["purpose_car", "purpose_furniture", "purpose_radio_tv", "purpose_domestic", 
                    "purpose_repairs", "purpose_education", "purpose_vacation"]
        purpose_labels = ["Car", "Furniture", "Radio/TV", "Domestic Appliances", "Repairs", "Education", "Vacation"]
        for purpose, label in zip(purposes, purpose_labels):
            input_data[purpose] = 1 if st.checkbox(f"Loan for {label}", value=False) else 0

        submit = st.form_submit_button("Assess Risk")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    if submit:
        input_df = pd.DataFrame([input_data], columns=feature_names)
        st.write("Input Data:", input_df) 
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]
        st.write("Prediction:", prediction, "Probability:", prob)  
        shap_values = explainer.shap_values(input_df)
        
        st.subheader("Risk Outcome")
        st.metric("Risk Level", "Bad" if prediction == 0 else "Good", f"{prob:.2%}")
        st.plotly_chart(waterfall_chart(shap_values, feature_names, input_df.iloc[0]))
        st.text(generate_narrative(input_df.iloc[0], shap_values, feature_names))
        st.markdown("</div>", unsafe_allow_html=True)
