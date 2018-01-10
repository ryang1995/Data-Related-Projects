import csv
import sys
import os
import math
from datetime import datetime
from decimal import Decimal

# read the data and save as list
with open('./phase1_dataset/movie-actor.csv','rb') as f:
	reader = csv.DictReader(f)
	movie_actor = [row for row in reader]
	ranks = [row['actor_movie_rank'] for row in movie_actor]

with open('./phase1_dataset/mltags.csv', 'rb') as f:
	reader = csv.DictReader(f)
	alltags = [row for row in reader]
	ts = [row['timestamp'] for row in alltags]

with open('./phase1_dataset/genome-tags.csv', 'rb') as f:
	reader = csv.DictReader(f)
	tag_names = [row for row in reader]

def get_output(actorid, model, tags):
	# make outputs directory
	dirPath = os.path.join('../', 'outputs')
	if not os.path.exists(dirPath):
		os.mkdir(dirPath)
	# make task1 directory
	dirPath2 = os.path.join(dirPath, 'task1')
	if not os.path.exists(dirPath2):
		os.mkdir(dirPath2)
	filename = 'actor' + actorid + '_' + model + '.csv'
	file = os.path.join(dirPath2, filename)
	# write outputs into csv file
	with open(file, 'wb') as f:
		writer = csv.writer(f)
		writer.writerow(['tag', 'weight'])
		for tag in tags:
			# the type of tag is tuple.
			writer.writerow(tag)
	
def find_tag_name(tags):
	tag_dict = {}
	for tag in tags:
		for item in tag_names:
			if tag['tagid'] == item['tagId']:
				key = item['tag']
				value = tag['weight']
				tag_dict[key] = value
	tag_dict = sorted(tag_dict.iteritems(), key=lambda (k,v):(v,k), reverse=True)
	return tag_dict

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
	with open('./phase1_dataset/imdb-actor-info.csv', 'rb') as f:
		reader = csv.reader(f)
		actor = [row[0] for row in reader]
		N = len(actor)-1
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
		idf = math.log10(N/(float)(n))
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

def get_tags(movieid, tags):
	for item in alltags:
		if item['movieid'] == movieid['movieid']:
			tags.append({'tagid': item['tagid'], 'timestamp': item['timestamp'], 'rank': movieid['rank']})
	return tags

# Get all the movies the actor acted
def get_movies_acted(actorid):
	movies_played = []
	for item in movie_actor:
		if item['actorid'] == actorid:
			movies_played.append({'movieid': item['movieid'], 'rank': item['actor_movie_rank']})
	return movies_played

def main(actorid, model):
	moviesid = get_movies_acted(actorid)
	tags = []
	for item in moviesid:
		tags = get_tags(item, tags)
	if model == 'TF':
		tags = tf_model(tags)
	if model == 'TF-IDF':
		tags = tf_model(tags)
		tags = tfidf_model(tags)
	tags = find_tag_name(tags)
	get_output(actorid, model, tags)

if __name__ == '__main__':
	actorid = sys.argv[1]
	model = sys.argv[2]
	main(actorid, model)
