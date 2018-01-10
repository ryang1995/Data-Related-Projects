import numpy as np
import csv
import sys
import math

with open('./Phase2_data/movie-actor.csv', 'rb') as f:
		reader = csv.reader(f)
		movie_actor_data = [row for row in reader]

with open('./Phase2_data/imdb-actor-info.csv', 'rb') as f:
	reader = csv.reader(f)
	actor_info_data = [row for row in reader]
	actor_id_data = [row[0] for row in actor_info_data[1:]]

def get_sim_actors(coactor_matrix, seeds):
	# Get transition matrix
	tran_matrix = coactor_matrix.astype(float)
	for i, row in enumerate(tran_matrix):
		count = 0
		for j, col in enumerate(row):
			if col != 0: count += 1
		for j, col in enumerate(row):
			if col != 0: tran_matrix[i][j] = 1/float(count)

	tran_matrix = tran_matrix.T

	c = raw_input("Please input the restart probability:\n")
	if c == '': c = 0.2
	else: c = float(c)
	uq_total = np.zeros((len(actor_id_data), 1))
	for seed in seeds:
		# fist get index of the seed actor
		index = actor_id_data.index(seed)
		v = [[0.0] for i in range(0, len(actor_id_data))]
		v[index] = [1.0]
		restart_v = np.array(v)
		uq = restart_v
		i = 1
		while i:
			uq_t = np.dot((1-c) * tran_matrix, uq) + restart_v * c
			if np.array_equal(uq, uq_t):
				i = 0
			uq = uq_t

		uq_total = uq_total + uq

	actor_uq = []
	index_list = []
	for seed in seeds:
		index_list.append(actor_id_data.index(seed))
	for i, row in enumerate(uq_total):
		if i not in index_list:
			actor_uq.append((row[0], actor_id_data[i]))
	
	actor_uq = sorted(actor_uq, reverse=True)

	print "\n\nThe 10 most related actors are as follow:\n"
	for i in range(10):
		print actor_uq[i]


def get_coactor_matrix():
	# reverse the list to dict with actorid as key and the movies the actor played as value
	# movie_actor = {actorid:[movieid]}
	movie_actor = {}
	for row in movie_actor_data[1:]:
		actor_id = row[1]
		if actor_id in movie_actor:
			movie_actor[actor_id].append(row[0])
		else:
			movie_actor[actor_id] = [row[0]]

	matrix_data = []
	for i in range(0, len(actor_id_data)):
		for j in range(0, len(actor_id_data)):
			key1 = actor_id_data[i]
			key2 = actor_id_data[j]
			set1 = set(movie_actor[key1])
			set2 = set(movie_actor[key2])
			matrix_data.append(len(set1 & set2))
	
	coactor_matrix = np.array(matrix_data)
	coactor_matrix = coactor_matrix.reshape(len(actor_id_data), len(actor_id_data))

	return coactor_matrix

if __name__ == '__main__':
	matrix = get_coactor_matrix()
	print matrix
	# save actor id info in a list called seeds
	seeds = raw_input("Please input your seeds:\n").split(" ")
	get_sim_actors(matrix, seeds)

