import streamlit as st
import joblib
import pandas as pd 

# Load the model
model = joblib.load("models/churn_model.pkl")

feature_names = joblib.load("models/feature_names.pkl")

st.title("📱 Expresso Churn Prediction App")
st.markdown("Enter customer details below to predict churn:")

# Input fields
tenure = st.number_input("Tenure (1 to 10)", min_value=1, max_value=10, step=1)
montant = st.number_input("Recharge Amount (MONTANT)", min_value=0.0)
frequence_rech = st.number_input("Recharge Frequency")
revenue = st.number_input("Revenue")
arpu_segment = st.number_input("ARPU Segment")
frequence = st.number_input("Call Frequency")
data_volume = st.number_input("Data Volume")
on_net = st.number_input("On-Net Minutes")
orange = st.number_input("Orange Minutes")
regularity = st.number_input("Regularity Score")
mrg = st.selectbox("Is the user married?", ["Yes", "No"])
region_fe = st.slider("Region Score (Freq Encoded)", min_value=0.0, max_value=0.1)
top_pack_fe = st.slider("Top Pack Score (Freq Encoded)", min_value=0.0, max_value=0.1)
freq_top_pack = st.number_input("Freq Top Pack")

# Convert inputs to model format
input_dict = {
    "TENURE": tenure,
    "MONTANT": montant,
    "FREQUENCE_RECH": frequence_rech,
    "REVENUE": revenue,
    "ARPU_SEGMENT": arpu_segment,
    "FREQUENCE": frequence,
    "DATA_VOLUME": data_volume,
    "ON_NET": on_net,
    "ORANGE": orange,
    "REGULARITY": regularity,
    "MRG": 1 if mrg == "Yes" else 0,
    "REGION_FE": region_fe,
    "TOP_PACK_FE": top_pack_fe,
    "FREQ_TOP_PACK": freq_top_pack
}

input_df = pd.DataFrame([[input_dict[col] for col in feature_names]], columns=feature_names)



# Predict
if st.button("🧮 Predict Churn"):
    prediction = model.predict(input_df)[0]
    result = "❌ Customer is likely to churn." if prediction == 1 else "✅ Customer will likely stay."
    st.subheader("Prediction Result:")
    st.success(result)