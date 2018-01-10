# Matrix Generator for
# 1. Movie-Tag-Genre Tensor
# 2. Movie - tag Matrix

import numpy as np
import csv
import os
import Task1_helperFunctions as hel

def movie_tag_num_matrix(all_movie_id, all_tag_id):
    all_tags = hel.get_file_content("mltags")

    # Movie-tag Matrix
    movie_tag_matrix = np.zeros((len(all_movie_id), len(all_tag_id)))

    for row in all_tags:
        # Get the index of row
        i = all_movie_id.index(row[1])
        # Get the index of index
        j = all_tag_id.index(row[2])

        movie_tag_matrix[i][j] += 1

    movie_tag_matrix = transition_matrix_word_pro(movie_tag_matrix)

    # print(movie_tag_matrix)
    return movie_tag_matrix
    

# Get movie-genre-tag tensor
def movie_genre_tag_tensor(all_movie_id, movie_tag_matrix):
    all_movie_genres, all_genres = hel.get_genre_types()

    # Value will be put into tensor
    tensor_list = []

    # tensor genre sequence as all_genres list
    for genre in all_genres:
        matrix_list = []
        for i in range(len(all_movie_id)):
            row_list = []
            # If movie is this genre, get the tag value
            # Otherwise, value equals 0
            if genre not in all_movie_genres[i]:
                row_list = [0.0 for i in range(len(all_tag_id))]
            else:
                row_list = movie_tag_matrix[i].tolist()
            matrix_list.append(row_list)
        tensor_list.append(matrix_list)
    movie_genre_tag_tensor = np.array(tensor_list)
    return movie_genre_tag_tensor


# Get movie-genre matrix
def movie_genre_matrix(all_movie_id):
    all_movie_genres, all_genres = hel.get_genre_types()

    # Movie-Genre Matrix
    movie_genre_matrix = []

    for i in range(len(all_movie_id)):
        # Value list for each row in atrix
        row_list = []
        # Movie belongs to genre value get 1, otherwise value get 0
        for genre in all_genres:
            if genre in all_movie_genres[i]:
                row_list.append(1)
            else:
                row_list.append(0)

        movie_genre_matrix.append(row_list)

    movie_genre_matrix = np.array(movie_genre_matrix)
    return movie_genre_matrix

# Get movie-tag matrix
def movie_tag_matrix(all_movie_id, all_tag_id):

    # Movie-Tag Matrix
    movie_tag_matrix = np.zeros((len(all_movie_id), len(all_tag_id)))
    time_section, oldest_time = hel.time_section_calculate()

    for i in range(len(all_movie_id)):
        # Tagid:weight dictionary 
        # Tag ids are related to the given movie id 
        tag_weight_dict = hel.movie_tag_calculator(all_movie_id[i], len(all_movie_id), time_section, oldest_time)

        for (tag_id, weight) in tag_weight_dict.items():
            j = all_tag_id.index(tag_id)
            movie_tag_matrix[i][j] = weight

    return movie_tag_matrix

# Transition Matrix Generator
# 1. Transition Matrix representing the Out-Degree of each vertex
#    Each column represents each vertex's outgoing edges and its value 1/N,
#    Where N is the number of Outgoing Edges of this vertex

def transition_matrix_word_pro(input_matrix):
    # Get the row number and column number
    row_num = len(input_matrix)

    # Input Matrix is Empty
    if row_num == 0:
        return None

    col_num = len(input_matrix[0])

    # Transition Matrix
    result_matrix = []

    # Create Transition Matrix based on Input Matrix
    # Traverse each Row of Input Matrix Twice
    # O(RowNum * ColNum) = O(N^2)
    for r in range(0, row_num):
        # Count Non-zero value in each Row
        # Used to Calculate values in Transition Matrix
        counter = 0
        # Value List for each Row
        row_list = []

        # Count the num of Non-zero values in each Row
        for c in range(0, col_num):
            if input_matrix[r][c] != 0:
                counter = counter + input_matrix[r][c]

        # Form Transition Matrix Row by Row
        for c in range(0, col_num):
            if input_matrix[r][c] != 0 :
                row_list.append(input_matrix[r][c]/counter)
            else:
                row_list.append(0)

        # Append each row
        result_matrix.append(row_list)
    # --- End of Out-Most Loop ---

    result_matrix = np.array(result_matrix)

    # Return Transition Matrix
    return result_matrix
# ----- End of Transition Matrix Generation -----



# @Param - "Adjacency" Matrix (Symmetric, Square)
# Return - Transition Matrix in the form of 2D List
def transition_matrix_out_degree(input_matrix):
    # Get the row number and column number
    row_num = len(input_matrix)

    # Input Matrix is Empty
    if row_num == 0:
        return None

    col_num = len(input_matrix[0])

    # Transition Matrix
    result_matrix = []

    # Create Transition Matrix based on Input Matrix
    # Traverse each Row of Input Matrix Twice
    # O(RowNum * ColNum) = O(N^2)
    for r in range(0, row_num):
        # Count Non-zero value in each Row
        # Used to Calculate values in Transition Matrix
        counter = 0
        # Value List for each Row
        row_list = []

        # Count the num of Non-zero values in each Row
        for c in range(0, col_num):
            # Diagonal values should not be counted
            if input_matrix[r][c] != 0 and r != c:
                counter += 1

        # Form Transition Matrix Row by Row
        for c in range(0, col_num):
            # Diagonal values in Transition Matrix are all zero
            if input_matrix[r][c] != 0 and r != c:
                row_list.append(1/counter)
            else:
                row_list.append(0)

        # Append each row
        result_matrix.append(row_list)
    # --- End of Out-Most Loop ---

    # Transpose the Matrix (Make Columns to represent the Transition Info)
    result_matrix = [list(i) for i in zip(*result_matrix)]

    # Return Transition Matrix
    return result_matrix
# ----- End of Transition Matrix Generation -----
