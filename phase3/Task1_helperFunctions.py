import csv
import os
import math
from datetime import datetime
import numpy as np

def dotsimi_normalize(all_movie_id, sim_dict):
    sim_dict_new = {}
    max_simi = max(sim_dict.items(), key = lambda x:x[1])[1]
    if max_simi==0:
        max_simi = 1
    for (key, value) in sim_dict.items():
        id = all_movie_id[key]
        value = value / max_simi
        sim_dict_new[id] = value
    return sim_dict_new

def get_user_watched_movies(user_id):
    movie_rows = get_file_content('mlmovies')
    tag_rows = get_file_content('mltags')
    rating_rows = get_file_content('mlratings')

    # Get all Movie ids
    all_movie_id = [row[0] for row in movie_rows]

    # Get Moive ids watched by the given use 
    user_movie_id = []

    # Tag ids related the movies watched by the given user
    user_tag_id = []
    
    # Get movie ids watched by the given user according to ratings
    for row in rating_rows:
        if row[1] == user_id:
            user_movie_id.append(row[0])

    # Get movie ids watched by the given user according to tags
    for row in tag_rows:
        if row[0] == user_id and row[1] not in user_movie_id:
            user_movie_id.append(row[1])
        if row[1] in user_movie_id and row[2] not in user_tag_id:
            user_tag_id.append(row[2])
    return all_movie_id, user_movie_id, user_tag_id

# Remove the movies given user watched from matrix
def remove_movie_watched(input_matrix, all_movie, user_movie):
    column = len(input_matrix[0])
    # Matrix has rows with watched movie related to the given user
    watched_matrix = []
    # Add rows to the matrix
    # First get the rows user watched in the input matrix
    # Save the watched movie rows into a new matrix
    for movie_id in user_movie:
        index = all_movie.index(movie_id)
        watched_matrix.append(input_matrix[index].tolist())
        all_movie.remove(movie_id)
        # Delete the watched movie rows from the input matrix
        input_matrix = np.delete(input_matrix, (index), axis=0)
         
    watched_matrix = np.array(watched_matrix)
    return watched_matrix, input_matrix, all_movie

def get_file_content(file_name):
    # Get path of the csv file
    file_path = os.path.join("..", "phase3_test_data/" + file_name + ".csv")
    # List containted all the rows of the csv file
    content_rows = []
    with open(file_path, 'r') as f:
        f.seek(0)
        reader = csv.reader(f)
        next(reader, None)
        content_rows = [row for row in reader]
    return content_rows

def get_genre_types():
    movie_rows = get_file_content('mlmovies')
    # All Genres with each movieid
    all_movie_genres = [row[3] for row in movie_rows]
    # All Genre Types
    all_genres = []

    for genre in all_movie_genres:
        if '|' in genre:
            for item in genre.split('|'):
                if not item in all_genres:
                    all_genres.append(item)
        else:
            if genre not in all_genres:
                all_genres.append(genre)

    return all_movie_genres, all_genres 

def time_section_calculate():
    tag_rows = get_file_content('mltags')
    # List containted all the timestamps
    timestamp_list = [row[3] for row in tag_rows]

    # Normalize timestamp to [0, 1], map timestamp weight to this range
    # Convert the str to datetime
    newest_time = datetime.strptime(max(timestamp_list), '%Y-%m-%d %H:%M:%S')
    oldest_time = datetime.strptime(min(timestamp_list), '%Y-%m-%d %H:%M:%S')
    # Get the timestamp range in seconds
    time_section = newest_time - oldest_time
    if newest_time == oldest_time:
        time_section = newest_time
    return time_section, oldest_time

def movie_tag_calculator(movie_id, N, time_section, oldest_time):
    tag_rows = get_file_content('mltags')

    # Tags related to the given movie id
    movie_tags = []
    # Timestamps related to the movie_tags
    movie_ts = []


    for row in tag_rows:
        if row[1] == movie_id:
            # Get tags related to the given movie id
            # Get ts related to the given movie id
            movie_tags.append(row[2])
            movie_ts.append(row[3])

    # Dictionary with tag id as key and weight as value
    tag_weight = {}

    # Use raw count as weight and considerd ts-weight
    for i in range(len(movie_tags)):
        tag_ts = datetime.strptime(movie_ts[i], '%Y-%m-%d %H:%M:%S')
        tag_section = tag_ts - oldest_time
        ts_weight = tag_section / time_section
        weight = 1 + ts_weight
        
        # Get tag:weight dict 
        # Add up the weight of same tags
        if movie_tags[i] in tag_weight:
            tag_weight[movie_tags[i]] += weight
        else:
            tag_weight[movie_tags[i]] = weight

    for (tag, weight) in tag_weight.items():
        # Count the number of movies related with the given tag
        movie_count = []
        for row in tag_rows:
            if row[2] == tag and row[1] not in movie_count:
                movie_count.append(row[1])

        # Get the number of movies related to the given tag
        n = len(movie_count)
        # Calculate idf-weight
        idf_weight = math.log10(N/(float)(n))
        tag_weight[tag] = weight * idf_weight

    return tag_weight