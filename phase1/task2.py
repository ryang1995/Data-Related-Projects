import sys
import csv
import os
import math
from datetime import datetime

with open('./phase1_dataset/mlmovies.csv') as f:
	reader = csv.DictReader(f)
	movie_genre = [row for row in reader]
	genre_column = [column['genres'] for column in movie_genre]

with open('./phase1_dataset/mltags.csv', 'rb') as f:
	reader = csv.DictReader(f)
	alltags = [row for row in reader]
	ts = [row['timestamp'] for row in alltags]

with open('./phase1_dataset/genome-tags.csv', 'rb') as f:
	reader = csv.DictReader(f)
	tag_names = [row for row in reader]

def get_output(genre, model, tags):
	# make outputs directory
	dirPath = os.path.join('../', 'outputs')
	if not os.path.exists(dirPath):
		os.mkdir(dirPath)
	# make task1 directory
	dirPath2 = os.path.join(dirPath, 'task2')
	if not os.path.exists(dirPath2):
		os.mkdir(dirPath2)
	filename = 'genre' + genre + '_' + model + '.csv'
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
	# Get all the genres
	genres = []
	for item in genre_column:
		if not '|' in item:
			if not item in genres:
				genres.append(item)
		else:
			for genre in item.split('|'):
				if not genre in genres:
					genres.append(genre)
	N = len(genres)
	movie_genre_dict = {item['movieid']:item['genres'] for item in movie_genre}
	for tag in tags:
		movieid = []
		for item in alltags:
			if item['tagid'] == tag['tagid']:
				if not item['movieid'] in movieid:
					movieid.append(item['movieid'])

		tag_genres = []
		for movie in movieid:
			genre = movie_genre_dict[movie]
			if '|' in genre:
				for item in genre.split('|'):
					if not item in tag_genres:
						tag_genres.append(item)
			else:
				if not genre in tag_genres:
					tag_genres.append(genre)
		n = len(tag_genres)
		idf = math.log10(N/(float)(n))
		tag['weight'] = tag['weight'] * idf
					
	return tags

def tf_model(tags):
	for tag in tags:
		# Use raw count as the tf weight.
		tag_weight = 1
		# Calculate the timestamp_weight
		# Set the newest time equals 1
		# Set the oldest time equals 0
		new_time = datetime.strptime(max(ts), '%Y-%m-%d %H:%M:%S')
		old_time = datetime.strptime(min(ts), '%Y-%m-%d %H:%M:%S')
		tag_ts = datetime.strptime(tag['timestamp'], '%Y-%m-%d %H:%M:%S')
		time_section = (new_time-old_time).days*24*3600 + (new_time-old_time).seconds
		tag_section = (tag_ts-old_time).days*24*3600 + (tag_ts-old_time).seconds
		ts_weight = tag_section / (float)(time_section)
		# Get the final weight of a tag by add all these three weights i get
		weight = tag_weight + ts_weight
		tag.update({'weight': weight})
		tag.pop('timestamp')
	tags = combine_tags(tags)
	return tags

def get_tags(movieid):
	tags = []
	for item in alltags:
		if item['movieid'] in movieid:
			tags.append({'tagid': item['tagid'], 'timestamp': item['timestamp']})
	return tags

def get_movieid(genre):
	movieid = []
	for item in movie_genre:
		if genre in item['genres']:
			movieid.append(item['movieid'])
	return movieid

def main(genre, model):
	movieid = get_movieid(genre)
	tags = get_tags(movieid)
	if model == 'TF':
		tags = tf_model(tags)
	if model == 'TF-IDF':
		tags = tf_model(tags)
		tags = tfidf_model(tags)
	tags = find_tag_name(tags)
	get_output(genre, model, tags)

if __name__ == '__main__':
	genre = sys.argv[1]
	model = sys.argv[2]
	main(genre, model)