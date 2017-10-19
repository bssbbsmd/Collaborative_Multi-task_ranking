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


def write_train_ratings(f, max_uid, max_iid, rating_list):
	line = str(max_uid)+ " "+ str(max_iid)
	print(line, file=f)
	for rating in rating_list:
		print(str(rating[0])+" "+str(rating[1])+" "+str(rating[2]), file=f)

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
	### wrting training rating file, it is used for update pointwise

	g0 = open(output + '_train_rating_'+str(n_train_ratio)+'.rating', 'w')
	write_train_ratings(g0, n_users, n_items, train_triples_list)
	g0.close()

	#######################################################################
	## obtain the training pairs for each user

	# order the data by user id and obtain the ordered pairs for each user 
	train_triples_list.sort(cmp = pair_comp_user)
	print("Training dataset sorted by user_ids.") 
 
	g_train_user_1 = open(output + '_train_user_'+str(n_train_ratio)+'.lsvm', 'w')
	g_train_user_2 = open(output + '_train_user_pairs_'+str(n_train_ratio)+'.pair', 'w') 

	prev_uid = train_triples_list[0][0]
	curr_uid = train_triples_list[0][0]

	rating_list = []

	for triple in train_triples_list:
  		curr_uid = triple[0]
  		if curr_uid != prev_uid:
  			if not rating_list:  # check if list is empty
  				continue
  			write_lsvm_user(g_train_user_1, prev_uid, rating_list)
  			write_comps_user(g_train_user_2, prev_uid, rating_list)
  			rating_list = []
  			rating_list.append((triple[1], triple[2]))
  			prev_uid = curr_uid
  		else:
  			rating_list.append((triple[1], triple[2]))

  	#write the last triple		
	write_lsvm_user(g_train_user_1, prev_uid, rating_list)
	write_comps_user(g_train_user_2, prev_uid, rating_list)
	
	g_train_user_1.close()
	g_train_user_2.close()
	#######################################################################

	#######################################################################
	## obtain the test pairs for each user: for test, the selected users
	## should have at least 10 ratings (thus, we can evaluate using metrics
	## like NDCG@10 )
	test_triples_list.sort(cmp = pair_comp_user)
	print("Traing dataset sorted by user_ids")

	g_test_user_1 = open(output+'_test_user_'+str(n_train_ratio)+'.lsvm','w')

	prev_uid = test_triples_list[0][0]
	curr_uid = test_triples_list[0][0]

	rating_list = []
	for triple in test_triples_list:
		curr_uid = triple[0]
		if curr_uid != prev_uid:
			if not rating_list:
				continue
			if len(rating_list) >= n_test_least:			
				write_lsvm_user(g_test_user_1, prev_uid, rating_list)
			rating_list = []
			rating_list.append((triple[1], triple[2]))
			prev_uid = curr_uid
		else:
			rating_list.append((triple[1], triple[2]))

	# write the list user
	if len(rating_list) >= n_test_least:
		write_lsvm_user(g_test_user_1,prev_uid,rating_list)

	g_test_user_1.close()
	#######################################################################

	#######################################################################
	## obtain the train pairs for each item:
	train_triples_list.sort(cmp = pair_comp_item)
	print("Dataset sorted by item_ids.") 
 
	g_train_item_1 = open(output + '_train_item_'+str(n_train_ratio)+'.lsvm', 'w')
	g_train_item_2 = open(output + '_train_item_pairs_'+str(n_train_ratio)+'.pair', 'w') 

	prev_iid = train_triples_list[0][1]
	curr_iid = train_triples_list[0][1]

	rating_list = []

	for triple in train_triples_list:
  		curr_iid = triple[1]
  		if curr_iid != prev_iid:
  			if not rating_list:  # check if list is empty
  				continue
  			write_lsvm_item(g_train_item_1, prev_iid, rating_list)
  			write_comps_item(g_train_item_2, prev_iid, rating_list)
  			rating_list = []
  			rating_list.append((triple[0], triple[2]))
  			prev_iid = curr_iid
  		else:
  			rating_list.append((triple[0], triple[2]))

  	#write the last triple		
	write_lsvm_item(g_train_item_1, prev_iid, rating_list)
	write_comps_item(g_train_item_2, prev_iid, rating_list)
	
	g_train_item_1.close()
	g_train_item_2.close()

	#######################################################################

	#######################################################################
	## obtain the test pairs for each item: for test, the selected items
	## should have at least 10 ratings (thus, we can evaluate using metrics
	## like NDCG@10 )
	test_triples_list.sort(cmp = pair_comp_item)
	print("Traing dataset sorted by user_ids")

	g_test_item_1 = open(output+'_test_item_'+str(n_train_ratio)+'.lsvm','w')

	prev_iid = test_triples_list[0][1]
	curr_iid = test_triples_list[0][1]

	rating_list = []
	for triple in test_triples_list:
		curr_iid = triple[1]
		if curr_iid != prev_iid:
			if not rating_list:
				continue
			if len(rating_list) >= n_test_least:			
				write_lsvm_item(g_test_item_1, prev_iid, rating_list)
			rating_list = []
			rating_list.append((triple[0], triple[2]))
			prev_iid = curr_iid
		else:
			rating_list.append((triple[0], triple[2]))

	# write the list user
	if len(rating_list) >= n_test_least:
		write_lsvm_item(g_test_item_1,prev_iid,rating_list)

	g_test_item_1.close()
	#######################################################################





if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('input_file',
                      help="Dataset with user-item-rating triples")
  parser.add_argument('-o', '--output_file', action='store', dest='output',
                      default="", help="Prefix for the output files")
  parser.add_argument('-r', '--train_items', action='store', dest='n_train_ratio', type=float,
                      default=50, help="the ratio for training, e.g., if set 0.2, then 20 percentage for training and rest for test") 
  parser.add_argument('-t', '--test_item',   action='store', dest='n_test_least', type=int,
                      default=10, help="Minimum number of test items per user (Default 10)")
  parser.add_argument('-s', '--subsample',   action='store_true',
                      help="At most (N_TRAIN) comparions from (N_TRAIN) ratings are sampled for each user")
  args = parser.parse_args()

  if args.output == "":
    args.output = os.path.splitext(os.path.basename(args.input_file))[0]

  num2comp(args.input_file, args.output, args.n_train_ratio, args.n_test_least)
