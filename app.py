import streamlit as st
from model import hybrid_recommend
import pandas as pd

st.set_page_config(page_title="Food Recommender 🍔", layout="centered")

st.title("🍔 Food Recommendation System")
st.write("Find similar food based on your taste!")

# Load dataset for dropdown
food = pd.read_csv("1662574418893344.csv")

food_list = food['Name'].values

selected_food = st.selectbox("Select a food", food_list)

if st.button("Recommend"):
    results = hybrid_recommend(selected_food)

    st.subheader("🍽 Recommended Foods:")
    for i, item in enumerate(results):
        st.write(f"{i+1}. {item}")