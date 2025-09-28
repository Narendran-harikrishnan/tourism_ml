import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# =============================
# Custom Page Config
# =============================
st.set_page_config(page_title="Tourism Prediction App", layout="wide")

# =============================
# Apply Custom Background and Styles
# =============================
custom_css = """
<style>
body {
    background: linear-gradient(to right, #f0f2f6, #dce3ea);
    font-family: 'Segoe UI', sans-serif;
}

h1, h2, h3 {
    color: #2c3e50;
}

.stButton>button {
    background-color: #007acc;
    color: white;
    border-radius: 8px;
    padding: 0.5em 1em;
    font-size: 1em;
}

.stButton>button:hover {
    background-color: #005f99;
}

div[data-testid="stSidebar"] {
    background-color: #f7f9fb;
}

.block-container {
    padding-top: 2rem;
}

.card {
    background-color: white;
    padding: 2rem;
    border-radius: 12px;
    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.05);
    margin-bottom: 2rem;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# =============================
# Load the trained model
# =============================
try:
    model_path = hf_hub_download(
        repo_id="Narendranh/Tourism",
        filename="fir_tourism_model_v1.joblib",
        repo_type="model"
    )
    model = joblib.load(model_path)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# =============================
# Title & Description
# =============================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.title(":luggage: Tourism Prediction App")
st.write("""
This application predicts whether a customer is likely to purchase a tourism package.
Please enter customer details below to get a prediction.
""")
st.markdown("</div>", unsafe_allow_html=True)

# =============================
# User Inputs in Two Columns
# =============================
st.markdown("<div class='card'>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    duration_pitch = st.number_input("Duration of Pitch", min_value=0, max_value=60, value=10)
    num_persons = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
    num_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=2)
    preferred_star = st.selectbox("Preferred Property Star Rating", [1, 2, 3, 4, 5])
    num_trips = st.number_input("Number of Trips", min_value=0, max_value=50, value=5)

with col2:
    passport = st.selectbox("Has Passport", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    own_car = st.selectbox("Owns a Car", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    num_children = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0)
    monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=20000)
    typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

st.markdown("</div>", unsafe_allow_html=True)

# =============================
# Assemble Input Data
# =============================
input_data = pd.DataFrame([{
    "Age": age,
    "CityTier": city_tier,
    "DurationOfPitch": duration_pitch,
    "NumberOfPersonVisiting": num_persons,
    "NumberOfFollowups": num_followups,
    "PreferredPropertyStar": preferred_star,
    "NumberOfTrips": num_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_score,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": num_children,
    "MonthlyIncome": monthly_income,
    "TypeofContact": typeof_contact,
    "Occupation": occupation,
    "Gender": gender,
    "ProductPitched": product_pitched,
    "MaritalStatus": marital_status,
    "Designation": designation
}])

# =============================
# Prediction Logic
# =============================
if st.button("Predict Purchase"):
    try:
        input_dummies = pd.get_dummies(input_data, drop_first=False)
        input_data_processed = input_dummies.copy()
        
        categorical_cols_to_restore = [
            'Designation', 'ProductPitched', 'MaritalStatus', 
            'TypeofContact', 'Gender', 'Occupation'
        ]
        
        for col in categorical_cols_to_restore:
            input_data_processed[col] = input_data[col]

        expected_features = [
            'Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups',
            'PreferredPropertyStar', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore',
            'OwnCar', 'NumberOfChildrenVisiting', 'MonthlyIncome',
            'Designation', 'ProductPitched', 'MaritalStatus', 'TypeofContact', 'Gender', 'Occupation',
            'TypeofContact_Company Invited', 'TypeofContact_Self Enquiry',
            'Occupation_Salaried', 'Occupation_Small Business', 'Occupation_Large Business', 'Occupation_Free Lancer',
            'Gender_Male', 'Gender_Female',
            'ProductPitched_Basic', 'ProductPitched_Standard', 'ProductPitched_Deluxe',
            'ProductPitched_Super Deluxe', 'ProductPitched_King',
            'MaritalStatus_Single', 'MaritalStatus_Married', 'MaritalStatus_Divorced',
            'Designation_Executive', 'Designation_Manager', 'Designation_Senior Manager',
            'Designation_AVP', 'Designation_VP'
        ]

        for col in expected_features:
            if col not in input_data_processed.columns:
                input_data_processed[col] = 0

        input_data_final = input_data_processed[expected_features]

        prediction = model.predict(input_data_final)[0]
        result = ":white_check_mark: Will Purchase Package" if prediction == 1 else ":x: Will Not Purchase Package"
        st.subheader("Prediction Result:")
        st.success(f"The model predicts: **{result}**")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
