import csv
import sys
import os
import math
from datetime import datetime

with open('./phase1_dataset/mlratings.csv') as f:
	reader = csv.DictReader(f)
	user_movie_rating = [{'movieid':row['movieid'], 'userid':row['userid']} for row in reader]

with open('./phase1_dataset/mltags.csv') as f:
	reader = csv.DictReader(f)
	mltags = [row for row in reader]
	ts = [row['timestamp'] for row in mltags]

with open('./phase1_dataset/genome-tags.csv', 'rb') as f:
	reader = csv.DictReader(f)
	tag_names = [row for row in reader]

def get_output(userid, model, tags):
	# make outputs directory
	dirPath = os.path.join('../', 'outputs')
	if not os.path.exists(dirPath):
		os.mkdir(dirPath)
	# make task1 directory
	dirPath2 = os.path.join(dirPath, 'task3')
	if not os.path.exists(dirPath2):
		os.mkdir(dirPath2)
	filename = 'userid' + userid + '_' + model + '.csv'
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
	with open('./phase1_dataset/mlusers.csv') as f:
		reader = csv.reader(f)
		users = [row for row in reader]
		N = len(users)-1
	for tag in tags:
		userid = []
		for item in mltags:
			if item['tagid'] == tag['tagid']:
				if not item['userid'] in userid:
					userid.append(item['userid'])
		n = len(userid)
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

def get_tags(moviesid):
	tags = []
	for item in mltags:
		if item['movieid'] in moviesid:
			tags.append({'tagid': item['tagid'], 'timestamp': item['timestamp']})
	return tags

def get_movieid_mltags(userid, moviesid):
	for i in range(len(mltags)-1):
		if mltags[i]['userid'] == userid:
			movieid = mltags[i]['movieid']
			if not movieid in moviesid:
				moviesid.append(movieid)
			for j in range(i+1, len(mltags)-1):
				if mltags[j]['userid'] != userid:
					return moviesid
				movieid = mltags[j]['movieid']
				if not movieid in moviesid:
					moviesid.append(movieid)
	return moviesid

def get_movieid_mlratings(userid):
	moviesid = []
	for i in range(len(user_movie_rating)-1):
		if user_movie_rating[i]['userid'] == userid:
			moviesid.append(user_movie_rating[i]['movieid'])
			for j in range(i+1, len(user_movie_rating)-1):
				if user_movie_rating[j]['userid'] != userid:
					return moviesid
				moviesid.append(user_movie_rating[j]['movieid'])
	return moviesid

def main(userid, model):
	moviesid = get_movieid_mlratings(userid)
	moviesid = get_movieid_mltags(userid, moviesid)
	tags = get_tags(moviesid)
	if model == 'TF':
		tags = tf_model(tags)
	if model == 'TF-IDF':
		tags = tf_model(tags)
		tags = tfidf_model(tags)
	tags = find_tag_name(tags)
	get_output(userid, model, tags)

if __name__ == '__main__':
	userid = sys.argv[1]
	model = sys.argv[2]
	main(userid, model)