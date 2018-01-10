import csv
import sys
import math
import os
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

def get_output(g1, g2, model, g1_tags):
	# make outputs directory
	dirPath = os.path.join('../', 'outputs')
	if not os.path.exists(dirPath):
		os.mkdir(dirPath)
	# make task1 directory
	dirPath2 = os.path.join(dirPath, 'task4')
	if not os.path.exists(dirPath2):
		os.mkdir(dirPath2)
	filename = 'genre' + g1 + '_genre' + g2 + '_' + model + '.csv'
	file = os.path.join(dirPath2, filename)
	# write outputs into csv file
	with open(file, 'wb') as f:
		writer = csv.writer(f)
		writer.writerow([g1])
		writer.writerow(['tag', 'weight'])
		for tag in g1_tags:
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

def p_diff2(tags, movieset, movieid_set):
	R = len(movieset)
	M = len(movieid_set)
	tags = [item['tagid'] for item in tags]
	tags = list(set(tags))
	tag_list = []
	for tag in tags:
		rlist = []
		mlist = []
		for item in alltags:
			if item['tagid'] == tag:
				if item['movieid'] in movieset:
					rlist.append(item['movieid'])
				if item['movieid'] in movieid_set:
					mlist.append(item['movieid'])
		r = R - len(rlist)
		m = M - len(mlist)
		temp1 = R - r + 1
		temp2 = m - r
		temp3 = M - m - R + r + 1
		temp4 = M - R + 1
		weight1 = math.log10(((r/(float)(temp1)) / ((temp2/(float)(temp3))+1)+1)
		weight2 = abs((r/(float)(R))- (temp2/(float)(temp4)))
		weight = weight1 * weight2
		tag_list.append({'tagid':tag,'weight':weight})
	return tag_list

def p_diff1(tags, movieset, movieid_set):
	R = len(movieset)
	M = len(movieid_set)
	tags = [item['tagid'] for item in tags]
	# Get all the tags related to genre
	tags = list(set(tags))
	tag_list = []
	for tag in tags:
		# the number of movies in genre g1, containing the tag
		rlist = []
		mlist = []
		for item in alltags:
			if item['tagid'] == tag:
				if item['movieid'] in movieset:
					rlist.append(item['movieid'])
				if item['movieid'] in movieid_set:
					mlist.append(item['movieid'])
		r = len(rlist)
		m = len(mlist)
		temp1 = R - r + 1
		temp2 = m - r
		temp3 = M - m - R + r + 1
		temp4 = M - R + 1
		weight1 = math.log10((r/(float)(temp1)) / ((temp2/(float)(temp3))+1))
		weight2 = abs((r/(float)(R))- (temp2/(float)(temp4)))
		weight = weight1 * weight2
		tag_list.append({'tagid':tag,'weight':weight})
	return tag_list

def idf(tags, movieid_set):
	# Get all the genres related to movies(g1)
	genres = []
	for item in movie_genre:
		if item['movieid'] in movieid_set:
			if not '|' in item['genres']:
				genres.append(item['genres'])
			else:
				for genre in item['genres'].split('|'):
					genres.append(genre)
	genres = list(set(genres))
	N = len(genres)

	movie_genre_dict = {item['movieid']:item['genres'] for item in movie_genre}
	for tag in tags:
		movieid = []
		for item in alltags:
			if item['tagid'] == tag['tagid']:
				if not item['movieid'] in movieid and item['movieid'] in movieid_set:
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

def tf(tags):
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

def get_tags(g1, g2):
	g1_tags = []
	g2_tags = []
	for item in alltags:
		if item['movieid'] in g1:
			g1_tags.append({'tagid': item['tagid'], 'timestamp': item['timestamp']})
		if item['movieid'] in g2:
			g2_tags.append({'tagid': item['tagid'], 'timestamp': item['timestamp']})
	return g1_tags, g2_tags

def get_movieid(g1, g2):
	g1_movieid = []
	g2_movieid = []
	for item in movie_genre:
		if g1 in item['genres']:
			g1_movieid.append(item['movieid'])
		if g2 in item['genres']:
			g2_movieid.append(item['movieid'])
	return g1_movieid, g2_movieid

def main(g1, g2, model):
	g1_movieid, g2_movieid = get_movieid(g1, g2)
	movieid_set = list(set(g1_movieid + g2_movieid))
	g1_tags, g2_tags = get_tags(g1_movieid, g2_movieid)
	if model == 'TF-IDF-DIFF':
		g1_tags = tf(g1_tags)
		g1_tags = idf(g1_tags, movieid_set)
		g1_tags = find_tag_name(g1_tags)
		get_output(g1, g2, model, g1_tags)
	elif model == 'P-DIFF1':
		g1_tags = p_diff1(g1_tags, g1_movieid, movieid_set)
		g1_tags = find_tag_name(g1_tags)
		get_output(g1, g2, model, g1_tags)
	elif model == 'P-DIFF2':
		g1_tags = p_diff2(g1_tags, g2_movieid, movieid_set)
		g1_tags = find_tag_name(g1_tags)
		get_output(g1, g2, model, g1_tags)

if __name__ == '__main__':
	g1 = sys.argv[1]
	g2 = sys.argv[2]
	model = sys.argv[3]
	main(g1, g2, model)
