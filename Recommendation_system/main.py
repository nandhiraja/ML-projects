import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import openpyxl

# Load datasets
@st.cache
def load_data():
    try:
        combined_features = pd.read_csv("cleaned_dataset.csv")["0"]
        recom_data = pd.read_csv("recom_dataset.csv")
        return combined_features, recom_data
    except FileNotFoundError:
        st.error("Dataset files not found. Please ensure 'cleaned_dataset.csv' and 'recom_dataset.csv' are in the project folder.")
        st.stop()

# Process features and compute similarity
@st.cache
def process_features(features):
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(features)
    return cosine_similarity(feature_vectors)

# Recommend movies
def recommend_movies(similarity, data, random_names):
    st.sidebar.subheader('Random Movie Suggestions')
    for name in random_names:
        st.sidebar.write(f"- {name}")

    st.markdown("### Enter a Movie Name to Get Recommendations:")
    movie_name = st.text_input('Movie Name', key="movie_name")

    if st.button('Find Recommendations'):
        list_of_all_titles = data['names'].tolist()
        close_matches = difflib.get_close_matches(movie_name, list_of_all_titles)

        if not close_matches:
            st.error("No match found. Please try another movie.")
            return

        close_match = close_matches[0]
        index = data[data.names == close_match]['index'].values[0]
        similarity_scores = list(enumerate(similarity[index]))
        sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        st.markdown("### Recommended Movies:")
        for idx, (movie_index, _) in enumerate(sorted_movies[:10], start=1):
            movie_title = data[data.index == movie_index]['names'].values[0]
            st.write(f"{idx}. {movie_title}")

# Main function
def main():
    st.title("🎥 Movie Recommendation System")
    st.markdown("**Using IMDb data to recommend movies tailored to your preferences.**")

    combined_features, recom_data = load_data()
    random_names = recom_data["names"].sample(4).tolist()

    st.sidebar.title("Movie Recommendation System")
    st.sidebar.markdown("Choose a genre or explore random suggestions.")
    
    genres = ["Action", "Comedy", "Horror", "Lifestyle"]
    selected_genre = st.sidebar.selectbox("Select a Genre", genres)

    similarity = process_features(combined_features)
    recommend_movies(similarity, recom_data, random_names)

    st.sidebar.markdown("---")
    st.sidebar.button("Exit", on_click=lambda: st.success("Thank you for visiting!"))

if __name__ == "__main__":
    main()
