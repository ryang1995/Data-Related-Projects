import csv
import math
from datetime import datetime
import numpy as np

with open('./Phase2_data/imdb-actor-info.csv', 'rb') as f:
	reader = csv.reader(f)
	actor_info_data = [row for row in reader]
	actor_id_data = [row[0] for row in actor_info_data[1:]]

with open('./phase2_data/movie-actor.csv','rb') as f:
	reader = csv.DictReader(f)
	movie_actor = [row for row in reader]
	ranks = [row['actor_movie_rank'] for row in movie_actor]

with open('./phase2_data/mltags.csv', 'rb') as f:
	reader = csv.DictReader(f)
	alltags = [row for row in reader]
	ts = [row['timestamp'] for row in alltags]

with open('./phase2_data/genome-tags.csv', 'rb') as f:
	reader = csv.DictReader(f)
	tag_names = [row for row in reader]


def find_tag_name(tags):
	actor_tag_v = [0 for i in range(len(tag_names))]
	for i, row in enumerate(tag_names):
		for tag in tags:
			if tag['tagid'] == row['tagId']:
				actor_tag_v[i] = tag['weight']
	return actor_tag_v

def combine_tags(tags):
	tempid = []
	tags_final = []
	for i in range(len(tags)):
		if not tags[i]['tagid'] in tempid:
			tempid.append(tags[i]['tagid'])
			for j in range(i+1,len(tags)):
				if tags[j]['tagid'] == tags[i]['tagid']:
					tags[i]['weight'] += tags[j]['weight']
			tags_final.append(tags[i])
	for item in tags_final:
		item['weight'] = item['weight']
	return tags_final

def tfidf_model(tags):
	for tag in tags:
		movieid = []
		for item in alltags:
			if item['tagid'] == tag['tagid']:
				if not item['movieid'] in movieid:
					movieid.append(item['movieid'])
		actor_tag = {}
		for item in movie_actor:
			if item['movieid'] in movieid:
				actor_tag[item['actorid']] = item['movieid']
		n = len(actor_tag)
		idf = math.log10(len(actor_id_data)/(float)(n))
		tag['weight'] = tag['weight'] * idf
	return tags

def tf_model(tags):
	for tag in tags:
		# Use raw count as the tf weight.
		tag_weight = 1
 		# Calculate the rank_weight
		# Set the weight of the highest rank equals 0
		# Set the weight of the lowest rank equals 1
		rank_weight = ((float)(max(ranks))-(float)(tag['rank'])) / ((float)(max(ranks))-(float)(min(ranks)))
		# Calculate the timestamp_weight
		# Set the newest time equals 1
		# Set the oldest time equals 0
		new_time = datetime.strptime(max(ts), '%Y-%m-%d %H:%M:%S')
		old_time = datetime.strptime(min(ts), '%Y-%m-%d %H:%M:%S')
		tag_ts = datetime.strptime(tag['timestamp'], '%Y-%m-%d %H:%M:%S')
		time_section = (new_time-old_time).days*24*3600 + (new_time-old_time).seconds
		tag_section = (tag_ts-old_time).days*24*3600 + (tag_ts-old_time).seconds
		ts_weight = (float)(tag_section) / (float)(time_section)
		# Get the final weight of a tag by add all these three weights i get
		weight = (tag_weight + rank_weight + ts_weight)
		tag.update({'weight': weight})
		tag.pop('timestamp')
		tag.pop('rank')

	tags = combine_tags(tags)
	return tags

# Get all the tags with timestamp
def get_tags(movieid):
	tags_list = []
	for item in alltags:
		if item['movieid'] == movieid['movieid']:
			tags_list.append({'tagid': item['tagid'], 'timestamp': item['timestamp'], 'rank': movieid['rank']})
	return tags_list

# Get all the movies the actor acted
def get_movies_acted(actorid):
	movies_played = []
	for item in movie_actor:
		if item['actorid'] == actorid:
			movies_played.append({'movieid': item['movieid'], 'rank': item['actor_movie_rank']})
	return movies_played

def get_sim_actors(simi_matrix, seeds):
	# Get transition matrix
	tran_matrix = simi_matrix
	for i, row in enumerate(tran_matrix):
		count = 0
		for j, col in enumerate(row):
			if col != 0: count += 1
		for j, col in enumerate(row):
			if col != 0: tran_matrix[i][j] = 1/float(count)

	tran_matrix = tran_matrix.T

	c = raw_input("Please input the restart probability(input a number < 1 and > 0):\n")
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

def get_simi_matrix():
	a_t_matrix = []
	for actor in actor_id_data:
		moviesid = get_movies_acted(actor)
		for item in moviesid:
			tags = get_tags(item)
		tags = tf_model(tags)
		tags = tfidf_model(tags)
		actor_tag_vector = find_tag_name(tags)
		a_t_matrix.append(actor_tag_vector)
	a_t_matrix = np.array(a_t_matrix)
	simi_matrix = np.dot(a_t_matrix, a_t_matrix.T)
	return simi_matrix

if __name__ == '__main__':
	matrix = get_simi_matrix()
	print matrix

	seeds = raw_input("Please input your seeds:\n").split(" ")
	get_sim_actors(matrix, seeds)



