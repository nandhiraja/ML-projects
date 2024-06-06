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
st.write('''# Welcome to movie recommendation system 
**Here we use the IMDb movie dataset to build the recommendation system**
### _______________________ **Kaviya..**''')

user_value = st.selectbox('Choose which type of movie you like', ["Action", "Comedy", "Horror", "Life Style"])

def processing(combined_features):
    # Initialize the model  
    vectorizer = TfidfVectorizer()

    # Fit the data into the model
    feature_vectors = vectorizer.fit_transform(combined_features)

    # Getting the similarity scores using cosine similarity
    similarity = cosine_similarity(feature_vectors)
    
    return similarity

def recommendation(similarity, recom_data):
    # Make the recommendation based on your movie preference
    st.write(''' ## SOME RANDOM MOVIE NAMES
    ---------------------------------------------------------------''')
    for ran_names in random_names:
        st.write(ran_names)
    st.write('''
    ---------------------------------------------------------------''')
    
    # Use a unique key for the text input widget
    movie_name = st.text_input('**Now Enter the movie name down**', key="movie_name_1")

    if movie_name:
        # Record the user input in an Excel sheet
        record_user_input(movie_name)
    
    # Get all movie names as a list
    list_of_all_titles = recom_data['names'].tolist()
    
    # Find the closest match
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    
    #st.write('''## More top 3 similar movies''')
    #for i in find_close_match:
    #    st.write(i) 
    
    if not find_close_match:
        st.write("No match found, try another")
        return

    close_match = find_close_match[0]  # Take the first closest match
    index_of_the_movie = recom_data[recom_data.names == close_match]['index'].values[0]  # Get the index of the movie
    similarity_score = list(enumerate(similarity[index_of_the_movie]))  # Generate similarity score for the given movie
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)  # Sort by similarity score

    st.write('''
    ## Movies suggested for you. You may also like these movies ....
    ''')
    
    movies = []
    for idx, movie in enumerate(sorted_similar_movies[:10], 1):
        index = movie[0]
        title_from_index = recom_data[recom_data.index == index]['names'].values[0]
        movies.append(f"{idx}. {title_from_index}")

    # Display the recommended movies
    for movie in movies:
        st.write(movie)
def record_user_input(movie_name):
    # Define the filename
    filename = "user_movie_inputs.xlsx"
    
    try:
        # Load existing data if the file exists
        df_existing = pd.read_excel(filename)
    except FileNotFoundError:
        # Create a new DataFrame if the file does not exist
        df_existing = pd.DataFrame(columns=["Movie Name"])
    
    # Append the new input
    df_new = pd.DataFrame([[movie_name]], columns=["Movie Name"])
    df_updated = pd.concat([df_existing, df_new], ignore_index=True)
    
    # Save the updated DataFrame to the Excel file
    df_updated.to_excel(filename, index=False)

def main():
    similarity = processing(combined_features)
    recommendation(similarity, recom_data)

    val = st.text_input(''' ## Do you want to continue?   Yes / No''', key="continue")
    if val.lower() in ['n', 'no']:
        st.write(''' ## **No va Sollura...! poo da chips mandayaa** ''')
        st.stop()  # This stops the execution of the script

main()
