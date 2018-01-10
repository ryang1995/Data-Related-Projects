import math
import heapq
import numpy as np

def similarity_euc_calculate(first_matrix, second_matrix, movie_list):
	sim_dict = {}
	# data_1 nonwatched movie
	for index, data_1 in enumerate(second_matrix):
		sim_row = 0
		# data_2 : watched movie
		for data_2 in first_matrix:
			sim = np.sqrt(np.sum((data_1 - data_2)**2))
			sim_row += sim
		sim_row = 1/(1+sim_row)
		sim_dict[movie_list[index]] = sim_row
	return sim_dict

def similarity_dot_calculate(u_matrix, vT, user_movie_index, user_tag_index):
	sim_dict = {}
	for i in range(len(u_matrix)):
		if i in user_movie_index:
			continue
		else:
			sim = 0
			for j in range(len(vT[0])):
				if j in user_tag_index:
					sim += np.dot(u_matrix[i], vT[:,j])
			sim_dict[i] = sim
	return sim_dict

# Get the index of five nearest data in the movie_list 
def movie_recommendate(sim_dict):
	rec_list = []
	for i in range(5):
		key = max(sim_dict.items(), key = lambda x:x[1])[0]
		value = max(sim_dict.items(), key = lambda x:x[1])[1]
		rec_list.append([key, value])
		sim_dict.pop(key)

	return rec_list

# Calculate the similarity between other movies with the movies user has watched
# Fist_matrix is the data point represent movies the given user watched
# Second_matrix is the data point represent other movies
def similarity_calculate(first_matrix, second_matrix, movie_list):
	sim_dict = {}
	# data_1 nonwatched movie
	for index, data_1 in enumerate(second_matrix):
		sim_row = 0
		# data_2 : watched movie
		for data_2 in first_matrix:
			sim = 0
			# Calculates the KL distance between 2 vector
			for i in range(0, len(data_1)):
				sim += data_1[i] * math.log10(data_1[i] / data_2[i])
			sim_row = sim_row + sim
			sim_row = 1/(1+sim_row)
		sim_dict[movie_list[index]] = sim_row
	return sim_dict

