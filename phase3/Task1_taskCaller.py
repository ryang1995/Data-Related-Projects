import Task1_helperFunctions as hel
import Task1_MatrixGenerator as mat
import Task1_SimilarityCalculator as sc
import Task1_PageRankRWR as pRR
import os
import numpy as np
import sys
import tensorly as tl
from operator import itemgetter
from tensorly.decomposition import parafac
from sklearn.decomposition import LatentDirichletAllocation as LDA

def task_1_a(user_id, movie_tag_matrix):
	movie_rows = hel.get_file_content('mlmovies')
	tagname_rows = hel.get_file_content('genome-tags')

	# Get all the tag ids
	all_tag_id = [row[0] for row in tagname_rows]
	all_movie_name = [row[1] for row in movie_rows]
	all_movie_id, user_movie_id, user_tag_id = hel.get_user_watched_movies(user_id)

	user_tag_index = []
	for tag in user_tag_id:
		i = all_tag_id.index(tag)
		user_tag_index.append(i)

	user_movie_index = []
	for movie in user_movie_id:
		i = all_movie_id.index(movie)
		user_movie_index.append(i)

	# If user_movie_id list is empty means user didnt's watch a movie
	if len(user_movie_id) == 0:
		print("The User Have Not Watched Any Movie!")
		for i in range(5):
			print(all_movie_id[i], all_movie_name[i])
	# Use tag as feature and TF-IDF weight as value
	else:
		if len(user_tag_id) == 0:
			matrix = mat.movie_genre_matrix(all_movie_id)
		else:
			# matrix = mat.movie_tag_matrix(all_movie_id, all_tag_id)
			matrix = movie_tag_matrix
		# Apply svd algorithm
		u_matrix, s, vT = np.linalg.svd(matrix, full_matrices=False)
		value_list = []

		for eigen_value in s:
			if eigen_value > 10:
				value_list.append(eigen_value)
		length = len(value_list)

		s = np.array(value_list)
		u_matrix = u_matrix[:, :length]
		sigma = np.diag(s)
		vT = vT[:length, :]
		# key(movie_index) value(dot product similarity)
		sim_dict = sc.similarity_dot_calculate(u_matrix, vT, user_movie_index, user_tag_index)
		sim_dict = hel.dotsimi_normalize(all_movie_id, sim_dict)
		# print(sim_list)
		rec_list = sc.movie_recommendate(sim_dict)
	print(">>Result(similarity normalized to range(0, 1))--------------")
	for rec in rec_list:
		print(rec[0], movie_rows[all_movie_id.index(rec[0])][1], rec[1])
	return sim_dict

def task_1_b(user_id):
	movie_rows = hel.get_file_content('mlmovies')
	tagname_rows = hel.get_file_content('genome-tags')
	# List of all the tags, saved as id
	all_tag_id = [row[0] for row in tagname_rows]

	# Get the list of all movie ids, movie id that user watched 
	all_movie_id, user_movie_id, user_tag_id = hel.get_user_watched_movies(user_id)

	movie_tag_num_matrix = mat.movie_tag_num_matrix(all_movie_id, all_tag_id)
	# np.random.seed(SOME_FIXED_SEED)
	lda = LDA(n_components = 500)
	result_matrix = lda.fit_transform(movie_tag_num_matrix)
	# for row in result_matrix:
	# 	print(row)
	# print(result_matrix)

	watched_matrix, result_matrix, movie_list= hel.remove_movie_watched(result_matrix, all_movie_id, user_movie_id)

	sim_dict = sc.similarity_calculate(watched_matrix, result_matrix, movie_list)
	rec_list = sc.movie_recommendate(sim_dict)
	print(">>Result--------------")
	for rec in rec_list:
		print(rec[0], movie_rows[all_movie_id.index(rec[0])][1], rec[1])
	return sim_dict

def task_1_c(user_id, movie_tag_matrix):
	movie_rows = hel.get_file_content('mlmovies')
	# tagname_rows = hel.get_file_content('genome-tags')

	movie_rows = [row for row in movie_rows if int(row[2]) >= 2004]
	
	# Get all the tag ids
	# all_tag_id = [row[0] for row in tagname_rows]

	all_movie_id, user_movie_id, user_tag_id = hel.get_user_watched_movies(user_id)

	# Get movie-genre-tag-tensor
	movie_genre_tag_tensor = mat.movie_genre_tag_tensor(all_movie_id, movie_tag_matrix)

	# Apply Cp decomposition on the tensor
	tl.set_backend('numpy')
	movie_genre_tag_tensor = tl.tensor(movie_genre_tag_tensor)
	print(">Performing CP decomposition...")
	result_matrix = parafac(movie_genre_tag_tensor, rank = 500)[1]

	watched_matrix, result_matrix, movie_list= hel.remove_movie_watched(result_matrix, all_movie_id, user_movie_id)
	sim_dict = sc.similarity_euc_calculate(watched_matrix, result_matrix, movie_list)
	rec_list = sc.movie_recommendate(sim_dict)
	print(">>Result--------------")
	for rec in rec_list:
		print(rec[0], movie_rows[all_movie_id.index(rec[0])][1], rec[1])
	return sim_dict

def task_1_d(user_id, alpha):
	movie_rows = hel.get_file_content('mlmovies')
	tagname_rows = hel.get_file_content('genome-tags')

	# Get all the tag ids
	all_tag_id = [row[0] for row in tagname_rows]
	all_movie_name = [row[1] for row in movie_rows]
	# Get the list of all movie ids, movie id that user watched 
	# Use user_movie_id as seed_list
	all_movie_id, user_movie_id, user_tag_id = hel.get_user_watched_movies(user_id)

	# If user_movie_id list is empty means user didnt's watch a movie
	if len(user_movie_id) == 0:
		print("The User Have Not Watched Any Movie!")
		rec_list = [all_movie_id[i] for i in range(5)]
		
	# Use tag as feature and TF-IDF weight as value
	else:
		if len(user_tag_id) == 0:
			matrix = mat.movie_genre_matrix(all_movie_id)
		else:
			matrix = mat.movie_tag_matrix(all_movie_id, all_tag_id)

		mm_matrix = np.dot(matrix, matrix.T)

		# Get Transition Matrix
		trans_list = mat.transition_matrix_out_degree(mm_matrix)

		# Get Index of movie id the given user watched
		index_list = []
		for movie in user_movie_id:
			index_list.append(all_movie_id.index(movie))

		# Call Personalized PageRank
		# If alpha is not specified, run with predefined probability
		if alpha < 0.0 or alpha > 1.0:
			print("Alpha not specified, running with predefined value of 0.9")
			pr_result = pRR.rwr_page_rank(seeds=index_list, trans_2d_list=trans_list)
		else:
			pr_result = pRR.rwr_page_rank(seeds=index_list, trans_2d_list=trans_list, alpha=alpha)

		sim_dict = {}
		for i in range(0, len(pr_result)):
			if i not in index_list:
				sim_dict[all_movie_id[i]] = pr_result[i]

		rec_list = sc.movie_recommendate(sim_dict)
		print(">>Result--------------")
		for rec in rec_list:
			print(rec[0], movie_rows[all_movie_id.index(rec[0])][1], rec[1])
		# output_view = [(v, k) for k, v in pr_dict.items()]
		# output_view.sort(reverse=True)
		# output_list = []
		# counter = 0
		# for v, k in output_view:
		# 	if counter < 5:
		# 	    output_list.append([all_movie_id[k], all_movie_name[k], v])
		# 	    print(all_movie_id[k], all_movie_name[k], v)
		# 	    counter += 1
		return sim_dict

def task_1_e(user_id):
	a_sim_dict = task_1_a(user_id)
	b_sim_dict = task_1_b(user_id)
	c_sim_dict = task_1_c(user_id)
	d_sim_dict = task_1_d(user_id)
	# fuzz ranking
	sim_dict = {}
	for (movie_id, simi_value) in c_sim_dict:
		sim_dict[movie_id] = simi_value + a_sim_dict[movie_id] + b_sim_dict[movie_id] + d_sim_dict[movie_id]

	rec_list = sc.movie_recommendate(sim_dict)
	print(">>Result--------------")
	for rec in rec_list:
		print(rec[0], movie_rows[all_movie_id.index(rec[0])][1], rec[1])
	return sim_dict

