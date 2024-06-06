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

def processing(combined_features):
    # Initialize the model  
    vectorizer = TfidfVectorizer()

    # Fit the data into the model
    feature_vectors = vectorizer.fit_transform(combined_features)

    # Getting the similarity scores using cosine similarity
    similarity = cosine_similarity(feature_vectors)
    
    return similarity

def recommendation(similarity, recom_data):
    # Display some random movie names
    st.subheader('Some Random Movie Names')
    st.markdown('---')
    for ran_names in random_names:
        st.write(ran_names)
    st.markdown('---')
    
    # Use a unique key for the text input widget
    movie_name = st.text_input('Now Enter the Movie Name Below', key="movie_name_1")
    
    if st.button('Find Recommendations'):
        # Get all movie names as a list
        list_of_all_titles = recom_data['names'].tolist()
        
        # Find the closest match
        find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
        
        if not find_close_match:
            st.write("No match found, try another")
            return

        close_match = find_close_match[0]  # Take the first closest match
        index_of_the_movie = recom_data[recom_data.names == close_match]['index'].values[0]  # Get the index of the movie
        similarity_score = list(enumerate(similarity[index_of_the_movie]))  # Generate similarity score for the given movie
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)  # Sort by similarity score

        st.subheader('Movies Suggested for You')
        st.markdown('---')
        
        movies = []
        for idx, movie in enumerate(sorted_similar_movies[:10], 1):
            index = movie[0]
            title_from_index = recom_data[recom_data.index == index]['names'].values[0]
            movies.append(f"{idx}. {title_from_index}")

        # Display the recommended movies
        for movie in movies:
            st.write(movie)

def main():
    similarity = processing(combined_features)
    recommendation(similarity, recom_data)

    if st.button('Exit'):
        st.write('Thanks for visiting')
        st.stop()  # This stops the execution of the script

if __name__ == "__main__":
    main()
