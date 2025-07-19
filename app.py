import streamlit as st
import pandas as pd

from recommender import load_data
from preprocessor import clean_data

st.title("🎬 Movie Mind")

# Load and clean the movie data
with st.spinner("Loading and cleaning data..."):
    movies = load_data()
    movies = clean_data(movies)

# Display the shape and a few sample rows
st.subheader("📊 Cleaned Movie Dataset")
st.write(f"Shape: {movies.shape}")
st.dataframe(movies.head())
