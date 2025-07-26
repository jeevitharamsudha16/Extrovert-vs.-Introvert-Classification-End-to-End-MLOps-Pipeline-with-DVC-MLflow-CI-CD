import streamlit as st
import pandas as pd
import joblib
import os

# Load the best model
model_path = "artifacts/models/best_model.pkl"
model = joblib.load(model_path)
 
# Load label encoders
encoders_dir = "artifacts/encoders"
stage_fear_encoder = joblib.load(os.path.join(encoders_dir, "Stage_fear_encoder.pkl"))
drained_encoder = joblib.load(os.path.join(encoders_dir, "Drained_after_socializing_encoder.pkl"))

# Streamlit UI
st.title("🧠 Personality Type Predictor")
st.markdown("Predict whether someone is likely an **Extrovert** or **Introvert** based on behavioral traits.")

# User input
time_alone = st.slider("🕒 Time spent alone (hours/day)", 0, 12, step=1)
stage_fear_input = st.selectbox("🎤 Do you have stage fear?", stage_fear_encoder.classes_)
social_event_attendance = st.slider("🎉 Social events attended (per month)", 0, 15, step=1)
going_outside = st.slider("🚶 Going outside (days/week)", 0, 7, step=1)
drained_input = st.selectbox("😓 Do you feel drained after socializing?", drained_encoder.classes_)
friends_circle = st.slider("👥 Friends circle size", 0, 20, step=1)
post_frequency = st.slider("📱 Social media post frequency (per week)", 0, 20, step=1)

# On predict
if st.button("🔍 Predict Personality"):
    # Encode categorical
    stage_fear_encoded = stage_fear_encoder.transform([stage_fear_input])[0]
    drained_encoded = drained_encoder.transform([drained_input])[0]

    # Create input DataFrame with corrected feature order
    input_data = pd.DataFrame([{
        'Time_spent_Alone': time_alone,
        'Stage_fear': stage_fear_encoded,
        'Social_event_attendance': social_event_attendance,
        'Going_outside': going_outside,
        'Drained_after_socializing': drained_encoded,
        'Friends_circle_size': friends_circle,
        'Post_frequency': post_frequency
    }])

    # Predict
    prediction = model.predict(input_data)[0]
    label = "🌟 Extrovert" if prediction == 0 else "🔒 Introvert"

    st.success(f"**Predicted Personality:** {label}")
