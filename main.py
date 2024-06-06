import numpy as np 
import pandas as pd
import streamlit as st
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

combined_features = pd.read_csv("cleaned_dataset.csv")
recom_data = pd.read_csv("recom_dataset.csv")

combined_features = combined_features["0"]

random_names = random.sample(recom_data["names"].tolist(), 4)

st.write('''# Welcome to movie recommendation system 
 ** Here we use the Imdb movie dataset to build the recommendation system**''')
user_value = st.selectbox('choose which type of movie you like', ["Action", "Comedy", "Horror", "Life Style"])

def processing(combined_features):
    # Initialize the model  
    vectorizer = TfidfVectorizer()

    # Fit the data into the model
    feature_vectors = vectorizer.fit_transform(combined_features)

    # Getting the similarity scores using cosine similarity
    similarity = cosine_similarity(feature_vectors)
    
    return similarity

def recommendation(similarity, recom_data):
    # Make the recommendation based on your movie which you like 
    # Get input of the movie_name
    st.write(''' # SOME OF RANDOM MOVIE NAMES''')
    st.write(random_names)
    movie_name = st.text_input('Enter the movie name', key="movie_name")
    
    list_of_all_titles = recom_data['names'].tolist()  # Get all movie names as list
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)  # Find closest match

    st.write(find_close_match)
    if not find_close_match:
        st.write("No match found, try another")
        return

    close_match = find_close_match[0]  # Take the first closest match
    index_of_the_movie = recom_data[recom_data.names == close_match]['index'].values[0]  # Get the index of the movie
    similarity_score = list(enumerate(similarity[index_of_the_movie]))  # Generate similarity score for the given movie
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)  # Sort by similarity score

    st.write('''
    # Movies suggested for you 
    You may also like these movies:
    ''')
    i = 1
    movies = []
    for movie in sorted_similar_movies:
        index = movie[0]
        title_from_index = recom_data[recom_data.index == index]['names'].values[0]
        if i <= 10:
            movies.append(title_from_index)
            i += 1

    for idx, movie in enumerate(movies, 1):
        st.markdown(f"**{idx}. {movie}**")

def main():
    similarity = processing(combined_features)
    while True:
        recommendation(similarity, recom_data)
        val = st.text_input("Do you want to continue? yes/no", key="continue")
        if val.lower() in ['n', 'no']:
            break

main()
