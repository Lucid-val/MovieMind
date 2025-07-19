import streamlit as st
from recommender import load_data, clean_data, create_tags_column, recommend
from sklearn.feature_extraction.text import CountVectorizer

# Page config
st.set_page_config(page_title="Movie Mind", layout="centered")

# Title
st.title("🎬 Movie Mind - Your Movie Recommendation Buddy")

# Load and process data
df = load_data()
df = clean_data(df)
df = create_tags_column(df)

# Vectorization
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(df['tags']).toarray()

# --- Movie Search Input ---
search_query = st.text_input("🔎 Search for a movie you like", "")

# --- Auto-Suggest Matching Titles ---
matching_titles = df[df['title'].str.contains(search_query, case=False, na=False)]['title'].tolist()

if search_query:
    if matching_titles:
        selected_movie = st.selectbox("🎯 Select from suggestions", matching_titles)

        if st.button("🎬 Recommend Similar Movies"):
            recommendations = recommend(selected_movie, df, vectors)

            if recommendations:
                st.subheader("📌 Top 5 Recommendations:")
                for i, title in enumerate(recommendations, 1):
                    st.markdown(f"{i}. {title}")
            else:
                st.warning("No recommendations found. Try a different movie.")
    else:
        st.warning("⚠️ No matching titles found.")
