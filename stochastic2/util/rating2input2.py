'''Generate training set and test set files (user, item, rating) triples'''
from __future__ import print_function
import os
import sys
import random
import argparse
import itertools

def pair_comp(x, y):
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

def write_comps_ratings_2(f, user_id, ratings_list, n_train):
  random.shuffle(ratings_list)
  if len(ratings_list) > n_train:  
    for (rating1, rating2) in itertools.combinations(ratings_list[:n_train], 2):
      if rating1[1] > rating2[1]:      
        print("%d %d %.1f %d %.1f" %(user_id, rating1[0], rating1[1], rating2[0], rating2[1]), file=f)
      if rating1[1] < rating2[1]:
        print("%d %d %.1f %d %.1f" %(user_id, rating2[0], rating2[1], rating1[0], rating1[1]), file=f)      
  else:
    for (rating1, rating2) in itertools.combinations(ratings_list, 2):
      if rating1[1] > rating2[1]:      
        print("%d %d %.1f %d %.1f" %(user_id, rating1[0], rating1[1], rating2[0], rating2[1]), file=f)
      if rating1[1] < rating2[1]:
        print("%d %d %.1f %d %.1f" %(user_id, rating2[0], rating2[1], rating1[0], rating1[1]), file=f) 


def write_lsvm_one(f, user_id, ratings_list):
  line = str(user_id)+" " 
  for (item_id, rating) in ratings_list:
    line = line + "{0}:{1} ".format(item_id, rating)
  print(line, file=f)

def write_rating(f, n_users, n_items, data):
  line = str(n_users)+" "+str(n_items)
  print(line, file=f)
  for i in xrange(1, len(data)+1):
    cur_list = data[i-1]
    for rating in cur_list:
      print(str(i)+" "+str(rating[0])+" "+str(rating[1]), file=f)  

def write_comp(f, n_users, n_items, data):
  line = str(n_users)+" "+str(n_items)
  print(line, file=f)
  for i in xrange(1, len(data)+1):
    write_comps_ratings(f, i, data[i-1]) 

def write_lsvm(f, n_users, n_items, data):
  line = str(n_users)+" "+str(n_items)
  print(line, file=f)
  for i in xrange(1, len(data)+1):
    write_lsvm_one(f, i, data[i-1])      

def num2comp(filename, output, n_train, n_test):
  n_users = 0
  n_items = 0
  n_ratings = 0
  triples_list = []
  f = open(filename, 'r')
  for line in f:
    (user_id, item_id, rating) = line.strip().split()
    triples_list.append((int(user_id), int(item_id), float(rating)))
    n_users = max(n_users, int(user_id))
    n_items = max(n_items, int(item_id))
  f.close()

  print("Dataset for {0} users, {1} items loaded.".format(n_users, n_items)) 

  triples_list.sort(cmp=pair_comp)

  print("Dataset sorted.")

  idx = 0
  num_users = 0

  g1 = open(output + '_train_'+str(n_train)+'.pair', 'w')
  g2 = open(output + '_test_'+ str(n_train)+'.lsvm', 'w') 
  g3 = open(output + '_train_'+str(n_train)+'.rating', 'w') 
  
  train_list = []
  test_list = []
  train_triples = []

  for u in xrange(1, n_users+1):
    ratings_list = []

    while triples_list[idx][0] == u:
      ratings_list.append((triples_list[idx][1], triples_list[idx][2]))
      idx = idx + 1
      if idx == len(triples_list):
        break

    if len(ratings_list) >= n_train + n_test:
      num_users = num_users + 1
      random.shuffle(ratings_list)
      train = ratings_list[:n_train]
      train.sort(cmp=pair_comp)
      train_list.append(train)
      for _rating in train:
        train_triples.append((num_users, _rating[0], _rating[1]) )
      test  = ratings_list[n_train:]
      test.sort(cmp=pair_comp)
      test_list.append(test)
  
  write_comp(g1, num_users, n_items, train_list)
  write_lsvm(g2, num_users, n_items, test_list)
  write_rating(g3, num_users, n_items, train_list)
  
  g1.close()
  g2.close()
  g3.close()

  #######################################################################
  ## obtain the train pairs for each item:
  train_triples.sort(cmp=pair_comp_item)
  g4 = open(output + '_train_'+str(n_train)+'_add.pair', 'w')
  
  first_line = str(num_users)+" "+str(n_items)
  print(first_line, file=g4)

  prev_iid = train_triples[0][1]
  curr_iid = prev_iid

  rating_list = []

  for triple in train_triples:
    curr_iid = triple[1]
    if curr_iid != prev_iid:
      if not rating_list:  # check if list is empty
        continue
      write_comps_ratings_2(g4, prev_iid, rating_list, n_train)
      rating_list = []
      rating_list.append((triple[0], triple[2]))
      prev_iid = curr_iid
    else:
      rating_list.append((triple[0], triple[2]))
  
  write_comps_ratings_2(g4, prev_iid, rating_list, n_train)
  g4.close()

  print("Comparisons generated for {0} users".format(num_users))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('input_file',
                      help="Dataset with user-item-rating triples")
  parser.add_argument('-o', '--output_file', action='store', dest='output',
                      default="", help="Prefix for the output files")
  parser.add_argument('-n', '--train_items', action='store', dest='n_train', type=int,
                      default=50, help="Number of training items per user (Default 50)") 
  parser.add_argument('-t', '--test_item', action='store', dest='n_test', type=int,
                      default=10, help="Minimum number of test items per user (Default 10)")
  parser.add_argument('-s', '--subsample', action='store_true',
                      help="At most (N_TRAIN) comparions from (N_TRAIN) ratings are sampled for each user")
  args = parser.parse_args()

  if args.output == "":
    args.output = os.path.splitext(os.path.basename(args.input_file))[0]

num2comp(args.input_file, args.output, args.n_train, args.n_test)
