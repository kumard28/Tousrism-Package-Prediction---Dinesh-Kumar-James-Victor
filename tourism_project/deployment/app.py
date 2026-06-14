import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# ── Load the trained model from Hugging Face model hub ───────────────────────
model_path = hf_hub_download(
    repo_id="kumard28/Tourism-Package-Prediction-Model",
    filename="tourism_model.pkl"
)
model = joblib.load(model_path)

# ── App Title and Description ─────────────────────────────────────────────────
st.title("Tourism Package Prediction App")
st.write("""
This application predicts whether a customer is likely to purchase the
Wellness Tourism Package offered by 'Visit with Us'.
Please fill in the customer details below and click **Predict** to get a result.
""")

# ── User Inputs ───────────────────────────────────────────────────────────────

age = st.number_input("Age", min_value=18, max_value=100, value=35)

type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])

city_tier = st.selectbox("City Tier", [1, 2, 3])

duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=60, value=10)

occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])

gender = st.selectbox("Gender", ["Male", "Female"])

number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)

number_of_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)

product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])

preferred_property_star = st.selectbox("Preferred Property Star Rating", [3, 4, 5])

marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

number_of_trips = st.number_input("Number of Trips per Year", min_value=0, max_value=20, value=2)

passport = st.selectbox("Holds Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

pitch_satisfaction_score = st.number_input("Pitch Satisfaction Score (1-5)", min_value=1, max_value=5, value=3)

own_car = st.selectbox("Owns a Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

number_of_children_visiting = st.number_input("Number of Children Visiting (under 5)", min_value=0, max_value=5, value=0)

designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

monthly_income = st.number_input("Monthly Income (₹)", min_value=1000, max_value=100000, value=20000)

# ── Assemble raw inputs into a DataFrame ─────────────────────────────────────
input_df = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": number_of_person_visiting,
    "NumberOfFollowups": number_of_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": number_of_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": number_of_children_visiting,
    "Designation": designation,
    "MonthlyIncome": monthly_income
}])

# ── Apply the same One Hot Encoding used during training ─────────────────────
# drop_first=True matches the encoding applied in the Data Preparation step
input_encoded = pd.get_dummies(input_df, drop_first=True)

# ── Align columns to match the model's expected feature set ──────────────────
# The model was trained on 27 columns — we reindex to ensure exact match
# Any missing OHE columns (e.g. a category not selected) are filled with 0
model_features = model.get_booster().feature_names
input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)

# ── Predict on Button Click ───────────────────────────────────────────────────
if st.button("Predict"):
    prediction = model.predict(input_encoded)[0]
    result = "✅ Likely to Purchase" if prediction == 1 else "❌ Unlikely to Purchase"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
