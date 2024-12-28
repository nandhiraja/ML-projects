import numpy as np 
import pandas as pd
import streamlit as st
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import openpyxl

# Load datasets
combined_features = pd.read_csv("cleaned_dataset.csv")
recom_data = pd.read_csv("recom_dataset.csv")

# Extract the combined features column
combined_features = combined_features["0"]

# Randomly select some movie names to display
random_names = random.sample(recom_data["names"].tolist(), 4)

# Display the title and description
st.title('Welcome to the Movie Recommendation System')
st.markdown('''
**Here we use the IMDb movie dataset to build the recommendation system**
### _______________________ **Nandhiraja..**
''')

# Movie genre selection
user_value = st.selectbox('Choose the type of movie you like', ["Action", "Comedy", "Horror", "Life Style"])

def process_features(features):
    """Process the features using TF-IDF vectorizer and compute cosine similarity."""
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(features)
    similarity = cosine_similarity(feature_vectors)
    return similarity

def recommend_movies(similarity, data):
    """Recommend movies based on user input."""
    st.subheader('Some Random Movie Names')
    st.markdown('---')
    for name in random_names:
        st.write(name)
    st.markdown('---')

    movie_name = st.text_input('Now Enter the Movie Name Below', key="movie_name_1")

    if st.button('Find Recommendations'):
        list_of_all_titles = data['names'].tolist()
        find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

        if not find_close_match:
            st.write("No match found, try another")
            return

        close_match = find_close_match[0]
        index_of_the_movie = data[data.names == close_match]['index'].values[0]
        similarity_score = list(enumerate(similarity[index_of_the_movie]))
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

        st.subheader('Movies Suggested for You')
        st.markdown('---')

        movies = []
        for idx, movie in enumerate(sorted_similar_movies[:10], 1):
            index = movie[0]
            title_from_index = data[data.index == index]['names'].values[0]
            movies.append(f"{idx}. {title_from_index}")

        for movie in movies:
            st.write(movie)

def main():
    """Main function to run the Streamlit app."""
    similarity = process_features(combined_features)
    recommend_movies(similarity, recom_data)

    if st.button('Exit'):
        st.write('Thanks for visiting')
        st.stop()

if __name__ == "__main__":
    main()
