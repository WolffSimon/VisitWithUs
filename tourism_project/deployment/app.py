# Streamlit app: predicts whether a customer will buy the Wellness Tourism Package
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

MODEL_REPO = "MatthewSimon/tourism-model"


@st.cache_resource
def load_model():
    """Download the registered model from the Hugging Face model hub (cached after first load)."""
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename="best_model.joblib")
    return joblib.load(model_path)


model = load_model()

st.title("Wellness Tourism Package: Purchase Prediction")
st.write(
    "Enter a customer's details to predict whether they are likely to buy the "
    "Wellness Tourism Package, so the sales team can prioritise who to contact."
)

# --- Customer details ---
st.subheader("Customer details")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=70, value=36)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital status", ["Married", "Single", "Unmarried", "Divorced"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly income (₹)", min_value=1000, max_value=100000,
                                     value=22000, step=500)
with col2:
    city_tier = st.selectbox("City tier", [1, 2, 3])
    passport = st.selectbox("Holds a passport?", ["No", "Yes"])
    own_car = st.selectbox("Owns a car?", ["No", "Yes"])
    number_of_trips = st.number_input("Trips per year", min_value=1, max_value=25, value=3)
    preferred_star = st.selectbox("Preferred hotel rating (stars)", [3, 4, 5])

# --- Trip details ---
st.subheader("Trip details")
col3, col4 = st.columns(2)
with col3:
    persons_visiting = st.number_input("People travelling", min_value=1, max_value=6, value=3)
with col4:
    children_visiting = st.number_input("Children under 5 travelling", min_value=0, max_value=4, value=1)

# --- Sales interaction ---
st.subheader("Sales interaction")
col5, col6 = st.columns(2)
with col5:
    type_of_contact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
    duration_of_pitch = st.number_input("Pitch duration (minutes)", min_value=1, max_value=130, value=14)
with col6:
    followups = st.number_input("Number of follow-ups", min_value=1, max_value=6, value=4)
    pitch_satisfaction = st.slider("Pitch satisfaction score", min_value=1, max_value=5, value=3)

# --- Assemble the inputs into a one-row DataFrame ---
# Column names and values match the raw training data; the saved pipeline
# handles one-hot encoding itself.
input_df = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": persons_visiting,
    "NumberOfFollowups": followups,
    "PreferredPropertyStar": preferred_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": number_of_trips,
    "Passport": 1 if passport == "Yes" else 0,
    "PitchSatisfactionScore": pitch_satisfaction,
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": children_visiting,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
}])

# Reorder columns to exactly match the order used in training
input_df = input_df[model.feature_names_in_]

# --- Predict ---
if st.button("Predict"):
    probability = model.predict_proba(input_df)[0, 1]
    if probability >= 0.5:
        st.success(f"Likely to purchase (probability {probability:.0%}). Recommended for contact.")
    else:
        st.info(f"Unlikely to purchase (probability {probability:.0%}).")
    with st.expander("Inputs used for this prediction"):
        st.dataframe(input_df)
