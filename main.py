import streamlit as st
import pandas as pd
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(
    open(os.path.join(BASE_DIR, "airbnb_model.pkl"), "rb")
)

preprocessor = pickle.load(
    open(os.path.join(BASE_DIR, "airbnb_preprocessor.pkl"), "rb")
)

st.title(" Airbnb Price Category Predictor")

st.write(
    "Enter the Airbnb listing information to predict its price category."
)


# User Inputs

neighbourhood_group = st.selectbox(
    "Neighbourhood Group",
    [
        "Manhattan",
        "Brooklyn",
        "Queens",
        "Bronx",
        "Staten Island"
    ]
)


neighbourhood = st.text_input(
    "Neighbourhood",
    "bedford-stuyvesant"
)


room_type = st.selectbox(
    "Room Type",
    [
        "Entire home/apt",
        "Private room",
        "Shared room"
    ]
)


latitude = st.number_input(
    "Latitude",
    value=40.7128
)


longitude = st.number_input(
    "Longitude",
    value=-74.0060
)


minimum_nights = st.number_input(
    "Minimum Nights",
    min_value=1,
    value=3
)


number_of_reviews = st.number_input(
    "Number of Reviews",
    min_value=0,
    value=10
)


reviews_per_month = st.number_input(
    "Reviews Per Month",
    min_value=0.0,
    value=1.0
)


calculated_host_listings_count = st.number_input(
    "Calculated Host Listings Count",
    min_value=1,
    value=1
)


availability_365 = st.number_input(
    "Availability 365",
    min_value=0,
    max_value=365,
    value=100
)


# Prediction

if st.button("Predict"):

    user_data = pd.DataFrame({
        "neighbourhood_group": [neighbourhood_group],
        "neighbourhood": [neighbourhood.lower().strip()],
        "room_type": [room_type],
        "latitude": [latitude],
        "longitude": [longitude],
        "minimum_nights": [minimum_nights],
        "number_of_reviews": [number_of_reviews],
        "reviews_per_month": [reviews_per_month],
        "calculated_host_listings_count": [
            calculated_host_listings_count
        ],
        "availability_365": [availability_365]
    })


    # Apply the same preprocessing
    user_data_processed = preprocessor.transform(user_data)


    # Prediction
    prediction = model.predict(user_data_processed)[0]


    st.success(f"Predicted Price Category: {prediction}")
