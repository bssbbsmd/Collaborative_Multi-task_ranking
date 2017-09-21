'''Generate training set and test set files (user, item, rating) triples'''
from __future__ import print_function

import os
import sys
import random
import argparse
import itertools

def pair_comp_user(x, y):
	if x[0] == y[0]:
		return x[1] - y[1]
	else:
		return x[0] - y[0]

def pair_comp_item(x, y):
	if x[1] == y[1]:
		return x[0] - y[0]
	else:
		return x[1] - y[1]


def write_comps_ratings(f, user_id, ratings_list):
	for (rating1, rating2) in itertools.combinations(ratings_list, 2):
	if rating1[1] > rating2[1]:      
    	print("%d %d %.1f %d %.1f" %(user_id, rating1[0], rating1[1], rating2[0], rating2[1]), file=f)
	if rating1[1] < rating2[1]:
		print("%d %d %.1f %d %.1f" %(user_id, rating2[0], rating2[1], rating1[0], rating1[1]), file=f)

def write_comps_user(f, user_id, user_rating_list):
	line = str(user_id)+" "
	for (r1, r2) in itertools.combinations(user_rating_list, 2):
		if r1[1] > r2[1]:
			line += "{0}:{1} ".format(r1[0], r2[0])
		if r1[1] < r2[1]:
			line += "{0}:{1} ".format(r2[0], r1[0])	
	print(line, file=f)		

def write_comps_item(f, item_id, item_rating_list):
	line = str(item_id)+ " "
	for (r1, r2) in itertools.combinations(item_rating_list, 2):
		if r1[1] > r2[1]:
			line += "{0}:{1} ".format(r1[0], r2[0])
		if r1[1] < r2[1]:
			line += "{0}:{1} ".format(r2[0], r1[0])	
	print(line, file=f)	

def write_lsvm_user(f, user_id, user_rating_list):
	line = str(user_id)+" "
  	for (item_id, rating) in user_rating_list:
  		line = line + "{0}:{1} ".format(item_id, rating)
  	print(line, file=f)

def write_lsvm_item(f, item_id, item_rating_list):
	line = str(item_id)+" "
	for (user_id, rating) in item_rating_list:
		line = line + "{0}:{1} ".format(user_id, rating)
	print(line, file=f)


def num2comp(filename, output, n_train_ratio, n_test_least):
	n_users = 0
	n_items = 0
	n_ratings = 0
	train_triples_list = []
	test_triples_list = []
	f = open(filename, 'r')

	##########################################################
	##### obtain train and test triples by the ratio value 

	for line in f:
		(user_id, item_id, rating) = line.strip().split()
		rdm_num = random.random()
		if rdm_num <= n_train_ratio:
			train_triples_list.append((int(user_id), int(item_id), float(rating)))
		else:
			test_triples_list.append((int(user_id), int(item_id), float(rating)))
		n_users = max(n_users, int(user_id))
		n_items = max(n_items, int(item_id))
	f.close()

	print("Dataset for {0} users, {1} items loaded.".format(n_users, n_items)) 
  
  	########################################################################
  
  
  	#######################################################################
  	## obtain the training pairs for each user
  
  	# order the data by user id and obtain the ordered pairs for each user 
	train_triples_list.sort(cmp = pair_comp_user)
	print("Dataset sorted by user_ids.") 
 
	g1 = open(output + '_train_user_'+str(n_train_ratio*100)+'.lsvm', 'w')
	g2 = open(output + '_train_user_pairs_'+str(n_train_ratio*100)+'.pair', 'w') 

	prev_uid = train_triples_list[0][0]
	curr_uid = train_triples_list[0][0]

	rating_list = []

	for triple in train_triples_list:
  		curr_uid = triple[0]
  		if curr_uid != prev_uid:
  			if not rating_list:  # check if list is empty
  				continue
  			write_lsvm_user(g1, prev_uid, rating_list)
  			write_comps_ratings(g2, prev_uid, rating_list)
  			rating_list = []
  			rating_list.append((triple[1], triple[2]))
  			prev_uid = curr_uid
  		else:
  			rating_list.append((triple[1], triple[2]))

  	#write the last triple		
	write_lsvm_user(g1, prev_uid, rating_list)
	write_comps_ratings(g2, prev_uid, rating_list)



'''

    ratings_list = []
  
    while triples_list[idx][0] == u:
      ratings_list.append((triples_list[idx][1], triples_list[idx][2]))
      idx = idx + 1
      if idx == len(triples_list):
        break

    if len(ratings_list) >= n_train + n_test:
      user_id = user_id + 1
      random.shuffle(ratings_list)
      train = ratings_list[:n_train]
      train.sort(cmp=pair_comp)
      test  = ratings_list[n_train:]
      test.sort(cmp=pair_comp)
      write_comps_ratings(g1, user_id, train)
      write_lsvm(g2, user_id, test)
  g1.close()
'''  
	print("Comparisons generated for {0} users".format(user_id))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('input_file',
                      help="Dataset with user-item-rating triples")
  parser.add_argument('-o', '--output_file', action='store', dest='output',
                      default="", help="Prefix for the output files")
  parser.add_argument('-n', '--train_items', action='store', dest='n_train_ratio', type=float,
                      default=50, help="the ratio for training, e.g., if set 0.2, then 20 percentage for training and rest for test") 
  parser.add_argument('-t', '--test_item',   action='store', dest='n_test_least', type=int,
                      default=10, help="Minimum number of test items per user (Default 10)")
  parser.add_argument('-s', '--subsample',   action='store_true',
                      help="At most (N_TRAIN) comparions from (N_TRAIN) ratings are sampled for each user")
  args = parser.parse_args()

  if args.output == "":
    args.output = os.path.splitext(os.path.basename(args.input_file))[0]

  num2comp(args.input_file, args.output, args.n_train_ratio, args.n_test_least)
