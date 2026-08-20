import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(
    page_title="Airbnb Price Predictor",
    page_icon="🏠",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "airbnb_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "airbnb_preprocessor.pkl"), "rb") as f:
    preprocessor = pickle.load(f)

st.markdown("""
<style>
.stApp {
    background-color: #0B1F33;
}

.title {
    text-align: center;
    color: #FF385C;
    font-size: 40px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #D9E2EC;
    margin-bottom: 30px;
}

h1, h2, h3, label, p {
    color: #FFFFFF !important;
}

[data-testid="stSelectbox"],
[data-testid="stTextInput"],
[data-testid="stNumberInput"] {
    color: white;
}

.stButton > button {
    width: 100%;
    background-color: #FF385C;
    color: white;
    font-size: 18px;
    font-weight: bold;
    border-radius: 10px;
    border: none;
}

.stButton > button:hover {
    background-color: #E03150;
    color: white;
}

.result {
    background-color: #132F4C;
    padding: 25px;
    margin-top: 25px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 4px 15px #061522;
}

.result h2 {
    color: #FF385C !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="title">🏠 Airbnb Price Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Predict the price category of an Airbnb listing</div>',
    unsafe_allow_html=True
)

st.subheader("📍 Location")

col1, col2 = st.columns(2)

with col1:
    neighbourhood_group = st.selectbox(
        "Neighbourhood Group",
        ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
    )

with col2:
    neighbourhood = st.text_input(
        "Neighbourhood",
        "bedford-stuyvesant"
    )

st.subheader("🏠 Property")

col1, col2, col3 = st.columns(3)

with col1:
    room_type = st.selectbox(
        "Room Type",
        ["Entire home/apt", "Private room", "Shared room"]
    )

with col2:
    latitude = st.number_input(
        "Latitude",
        value=40.7128
    )

with col3:
    longitude = st.number_input(
        "Longitude",
        value=-74.0060
    )

st.subheader("📊 Listing Information")

col1, col2, col3 = st.columns(3)

with col1:
    minimum_nights = st.number_input(
        "Minimum Nights",
        min_value=1,
        value=3
    )

with col2:
    number_of_reviews = st.number_input(
        "Number of Reviews",
        min_value=0,
        value=10
    )

with col3:
    reviews_per_month = st.number_input(
        "Reviews Per Month",
        min_value=0.0,
        value=1.0
    )

col1, col2 = st.columns(2)

with col1:
    calculated_host_listings_count = st.number_input(
        "Host Listings Count",
        min_value=1,
        value=1
    )

with col2:
    availability_365 = st.number_input(
        "Availability 365",
        min_value=0,
        max_value=365,
        value=100
    )

if st.button("🔮 Predict Price Category"):

    data = pd.DataFrame({
        "neighbourhood_group": [neighbourhood_group],
        "neighbourhood": [neighbourhood.lower().strip()],
        "room_type": [room_type],
        "latitude": [latitude],
        "longitude": [longitude],
        "minimum_nights": [minimum_nights],
        "number_of_reviews": [number_of_reviews],
        "reviews_per_month": [reviews_per_month],
        "calculated_host_listings_count": [calculated_host_listings_count],
        "availability_365": [availability_365]
    })

    processed = preprocessor.transform(data)

    prediction = model.predict(processed)[0]

    st.markdown(
        f"""
        <div class="result">
            <h3>🎯 Predicted Price Category</h3>
            <h2>{prediction}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
