import numpy as np 
import pandas as pd
import streamlit as st
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random



combined_features=pd.read_csv("/cleaned_dataset.csv")
recom_data=pd.read_csv("/recom_dataset.csv")

combined_features=combined_features["0"]


random_names = random.sample(recom_data["names"].tolist(), 4)

st.write('''## Welcome to movie recommendation system 
### Here we use the Imdb movie dataset to build the  recommendation system''')
user_value = st.selectbox('choose which type of movie you like ', ["Action","comedy","Horror","life style"])

def processing(combined_features):
    
    # initialize the model  
    vectorizer = TfidfVectorizer()

    # fit the data into the  model
    feature_vectors = vectorizer.fit_transform(combined_features)

    # getting the similarity scores using cosine similarity
    similarity = cosine_similarity(feature_vectors)
    
    return similarity


def recommendation(similarity,recom_data):
        # make the recommendation based on your movie which you like 
    # get input of the movie_name
    st.write(random_names)
    movie_name = st.text_input('Enter the movie name')
    #movie_name =input('Enter your favourite movie name : ')

    # here we define some default name  -----> otherwise you can get it from user 

    #movie_name = "Iron man"
    #print('Enter your favourite movie name : Iron man')

    list_of_all_titles = recom_data['names'].tolist()       # get all movie name as list

    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)  # find closest match --list of movie-- form oru given movie
    print(" similiar names  :  ",  find_close_match ,"\n\n")

    close_match = find_close_match[0]             # we take first one which is given by cloest movie

    index_of_the_movie = recom_data[recom_data.names == close_match]['index'].values[0]   #  geting the index of the movie

    similarity_score = list(enumerate(similarity[index_of_the_movie]))            # generate the similarity score for the given movie
        
    sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True)     # sort the score to get top 10 more similiarty movies

    print('Movies suggested for you : \n')
    st.write('''# Movies suggested for you 
    ## you may also like this movies ..''')
    i = 1
    
    movie=[]
    # print the top 10 movies base on similiarity score 
  
    for movie in sorted_similar_movies:
        index = movie[0]
        title_from_index = recom_data[recom_data.index==index]['names'].values[0]
        if (i<10):
            movie.append(title_from_index)
            print(i, '.',title_from_index)
            i+=1
    st.write(movie)


    

def main():
    print(random_names)
    similarity = processing(combined_features)
    while True:
        
        call= recommendation(similarity,recom_data)
        key=input("Are you continue....Y/N : ")
        
        if key in ['n','N','no','No','NO']:
            break
        
main()   
    
