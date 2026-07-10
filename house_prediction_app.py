import streamlit as st
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 California House Price Predictor")
st.markdown("### Predict Median House Values in California (1990)")

# Load Models
@st.cache_resource
def load_models():
    try:
        import os

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        model_path = os.path.join(BASE_DIR, "polynomial_house_model.pkl")
        scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
        poly_path = os.path.join(BASE_DIR, "poly_features.pkl")

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        poly = joblib.load(poly_path)

        return model, scaler, poly

    except Exception as e:
        st.error(f"❌ Error loading model files: {e}")
        st.stop()

model, scaler, poly = load_models()
st.success("✅ Models loaded successfully!")

# Sidebar - User Inputs
st.sidebar.header("Enter House Details")

med_inc = st.sidebar.number_input("Median Income (in $10,000s)", min_value=0.0, value=3.87, step=0.1)
house_age = st.sidebar.number_input("House Age (years)", min_value=0, value=28, step=1)
avg_rooms = st.sidebar.number_input("Average Rooms per Household", min_value=0.0, value=5.4, step=0.1)
avg_bedrooms = st.sidebar.number_input("Average Bedrooms per Household", min_value=0.0, value=1.1, step=0.1)
population = st.sidebar.number_input("Block Group Population", min_value=0, value=1425, step=10)
avg_occup = st.sidebar.number_input("Average Occupants per Household", min_value=0.0, value=3.0, step=0.1)
latitude = st.sidebar.number_input("Latitude", min_value=32.0, max_value=42.0, value=34.05, step=0.01)
longitude = st.sidebar.number_input("Longitude", min_value=-124.0, max_value=-114.0, value=-118.25, step=0.01)

# Prediction Button
if st.sidebar.button("🔍 Predict House Price", type="primary"):
    # Prepare input data
    input_data = pd.DataFrame([[
        med_inc, house_age, avg_rooms, avg_bedrooms,
        population, avg_occup, latitude, longitude
    ]], columns=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                 'Population', 'AveOccup', 'Latitude', 'Longitude'])

    # Transform
    input_scaled = scaler.transform(input_data)
    input_poly = poly.transform(input_scaled)

    # Make prediction
    prediction = model.predict(input_poly)[0]
    price_dollars = prediction * 100000

    # Display result
    st.success(f"### Predicted Median House Value: **${price_dollars:,.2f}**")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Price (in $100k)", f"{prediction:.2f}")
    with col2:
        st.metric("Approx. Range", f"${(price_dollars*0.9):,.0f} - ${(price_dollars*1.1):,.0f}")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Model: Polynomial Regression (Degree 2)\nDataset: California Housing")
st.caption("Built with Streamlit • House Price Prediction Project")
