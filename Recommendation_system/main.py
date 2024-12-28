import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
#import openpyxl
import os


# Load datasets
@st.cache_resource
def load_data():
    try:
       script_dir = os.path.dirname(os.path.abspath(__file__))

       cleaned_dataset = os.path.join(script_dir, "cleaned_dataset.csv")
       combined_features = pd.read_csv(cleaned_dataset)


       recom_dataset = os.path.join(script_dir, "recom_dataset.csv")
       recom_data = pd.read_csv(recom_dataset)

       return combined_features, recom_data
    except FileNotFoundError:
        st.error("Dataset files not found. Please ensure 'cleaned_dataset.csv' and 'recom_dataset.csv' are in the project folder.")
        st.stop()

# Process features and compute similarity
@st.cache_data
def process_features(features):
    """Process the features using TF-IDF vectorizer and compute cosine similarity."""
    # Ensure features is a Series (a single column of text)
    if isinstance(features, pd.DataFrame):
        if "0" in features.columns:
            features = features["0"]
        else:
            st.error("The expected column '0' is not found in the dataset.")
            st.stop()

    # Drop NaN values and remove empty strings
    features = features.dropna()
    features = features[features.str.strip() != ""]

    # Check if features is empty after cleaning
    if features.empty:
        st.error("All documents are empty or contain only stop words.")
        st.stop()

    # Apply TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(features)
    similarity = cosine_similarity(feature_vectors)
    return similarity

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
    """Main function to run the Streamlit app."""
    st.title("🎥 Movie Recommendation System")
    st.markdown("**Using IMDb data to recommend movies tailored to your preferences.**")

    combined_features, recom_data = load_data()
    
    if combined_features.empty or recom_data.empty:
        st.error("One of the datasets is empty. Please check the data files.")
        st.stop()

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
