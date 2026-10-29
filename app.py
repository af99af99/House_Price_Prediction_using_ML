import streamlit as st
import pandas as pd
import numpy as np
import joblib

#  PAGE
st.set_page_config(
    page_title="DeshGhar Vision | House Price Predictor",
    page_icon="🏠",
    layout="wide"
)


st.markdown("""
<style>

/* BACKGROUND */
.stApp {
    background: linear-gradient(135deg, #120021, #2a0a4a, #3b176b);
    color: #ffffff;
}

/* GLASS CARD */
.glass {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(22px);
    padding: 30px;
    border-radius: 22px;
    border: 1px solid rgba(255, 255, 255, 0.15);
    box-shadow: 0 8px 35px rgba(0,0,0,0.35);
}

/* TITLE */
.title {
    font-size: 44px;
    font-weight: 900;
    color: #f5d0fe;
}

/* SUBTITLE */
.subtitle {
    color: #d8b4fe;
    font-size: 16px;
}

/* LABEL */
label {
    color: #f3e8ff !important;
    font-weight: 600 !important;
}

/* INPUT TEXT */
input, textarea {
    color: #000000 !important;
}

/* number input */
.stNumberInput input {
    color: #000000 !important;
    background-color: rgba(255,255,255,0.85) !important;


}

/* SELECTBOX LABEL WHITE  */
.stSelectbox label {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/*  SELECTBOX VALUE BLACK */
div[data-baseweb="select"] span {
    color: #000000 !important;
    font-weight: 600 !important;
}

/* dropdown menu items */
div[data-baseweb="popover"] * {
    color: #000000 !important;
}

/* RADIO */
.stRadio label {
    color: #ffffff !important;
    font-weight: 700 !important;
}

.stRadio div[role="radiogroup"] label {
    color: #ffffff !important;
}

/* BUTTON */
.stButton>button {
    background: linear-gradient(90deg, #a855f7, #7c3aed);
    color: white;
    font-weight: bold;
    border-radius: 14px;
    height: 3.2em;
    border: none;
    box-shadow: 0 4px 25px rgba(168,85,247,0.5);
}

/* RESULT */
.result {
    background: rgba(168,85,247,0.25);
    padding: 22px;
    border-radius: 16px;
    color: #ffffff;
    text-align: center;
    border: 1px solid #a855f7;
    font-size: 18px;
    font-weight: bold;
}

.block-container {
    padding-top: 2rem;
}

.stRadio [role="radiogroup"] label {
    color: #ffffff !important;
}

.stRadio [role="radiogroup"] label * {
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# MODEL
model = joblib.load("xgb_house_model.pkl")
feature_names = joblib.load("model_features.pkl")

# HEADER 
st.markdown('<div class="title">🏠 DeshGhar Vision</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI House Price Predictor (Bangladesh Style)</div>', unsafe_allow_html=True)

st.markdown("---")

# INPUT 
st.markdown('<div class="glass">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider("Overall Quality", 1, 10, 7)
    gr_liv_area = st.number_input("Living Area (sq ft)", 500, 8000, 1500)
    total_bsmt_sf = st.number_input("Basement Area", 0, 5000, 800)
    garage_cars = st.slider("Garage Cars", 0, 4, 2)

with col2:
    house_age = st.number_input("House Age", 0, 150, 10)
    total_bath = st.number_input("Bathrooms", 1.0, 6.0, 2.0)

    neighborhood = st.selectbox(
        "Neighborhood",
        ['CollgCr','NridgHt','Somerst','OldTown','Edwards','Other']
    )

    bldg_type = st.selectbox(
        "Building Type",
        ['1Fam','TwnhsE','Duplex','Twnhs','2fmCon']
    )

    has_fireplace = st.radio("Fireplace", ["Yes", "No"])

st.markdown('</div>', unsafe_allow_html=True)

#PREDICTION 
if st.button("🔮 Predict Price"):

    input_data = pd.DataFrame([{
        'OverallQual': overall_qual,
        'GrLivArea': gr_liv_area,
        'TotalBsmtSF': total_bsmt_sf,
        'GarageCars': garage_cars,
        'HouseAge': house_age,
        'TotalBath': total_bath,
        'Neighborhood': neighborhood,
        'BldgType': bldg_type,
        'HasFireplace': 1 if has_fireplace == "Yes" else 0
    }])

    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=feature_names, fill_value=0)

    pred = model.predict(input_data)[0]
    price = np.expm1(pred)

    st.markdown(
        f'<div class="result">💰 Estimated Price: ${price:,.0f}</div>',
        unsafe_allow_html=True
    )

st.markdown("---")