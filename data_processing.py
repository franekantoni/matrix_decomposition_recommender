#USE PYTHON2


import numpy as np
import pickle
from sklearn.preprocessing import normalize
import pandas as pd
import re

def strip_quote(astring):
	if astring[0] == '"':
		return astring[1:]
	else:
		return astring


####.   DATA  #####  PROCESSING.  ###############################################


#load the data
data = pd.read_csv("ratings.csv")


#delete timestamp column
del data['timestamp']

#make data matrix sparse
data = data.pivot(index='userId', columns='movieId')

#change NaN to zeros
data = data.fillna(0)

#creatre a dict with the index as key and id as value
index_to_id = {}
for i, id_ in enumerate(data):
	index_to_id[i] = id_[1]

#create a dict with ids as keys and titles as values:
id_to_title = {}
with open("movies.csv", "r") as movies:
	for line in movies:
		title = re.findall(r"\,.+\)", line)
		line = line.split(",")
		if not title:
			title = line[1]
		else:
			title = title[0][1:]
		id_to_title[line[0]] = title

#create a dict with indexes as keys and titles as values:
index_to_title = {}
for index in index_to_id:

	id_ = index_to_id[int(index)]
	title = id_to_title.get(str(id_))
	index_to_title[index] = title

#create a dict with titles as keys and indexes as values:
title_to_index = {v: k for k, v in index_to_title.iteritems()}


with open('index_to_title_dict.pickle', 'wb') as handle:
    pickle.dump(index_to_title, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('title_to_index_dict.pickle', 'wb') as handle2:
    pickle.dump(title_to_index, handle2, protocol=pickle.HIGHEST_PROTOCOL)

#normalize the ratings for each user
data = normalize(data, norm='l2', axis=1)
with open('pickled_normalized_matrix.sav', 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

