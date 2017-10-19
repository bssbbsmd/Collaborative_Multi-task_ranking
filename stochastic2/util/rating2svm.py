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


def write_lsvm(f, ratings_list): 
  line=""
  for (item_id, rating) in ratings_list:
    line = line + "{0}:{1} ".format(item_id, rating)
  print(line, file=f)

def rating2svm(input, output, dele):
  triples_list = []
  n_users = 0
  n_items = 0

  f = open(input, 'r')

  if dele==1: #IGNORE THE FIRST line
    first_line = f.readline()

  for line in f:
    (user_id, item_id, rating) = line.strip().split()
    triples_list.append((int(user_id), int(item_id), float(rating)))
    n_users = max(n_users, int(user_id))
    n_items = max(n_items, int(item_id))
  f.close()

  print("Dataset for {0} users, {1} items loaded.".format(n_users, n_items)) 

  triples_list.sort(cmp=pair_comp_user)

  print("Dataset sorted by item id.")

  g1 = open(output+'.svm', 'w')

  idx=0

  for u in xrange(1, n_users+1):
    ratings_list = []

    while triples_list[idx][0] == u:
      ratings_list.append((triples_list[idx][1], triples_list[idx][2]))
      idx = idx + 1
      if idx == len(triples_list):
        break
    if len(ratings_list)>0:    
      write_lsvm(g1, ratings_list)    
  
  g1.close()

  #######################################################################
  ## obtain the train pairs for each item:


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--input_file', action='store', dest='input',
                      default="", help="Prefix for the input files")
  parser.add_argument('-o', '--out_file', action='store', dest='output',
                      default="", help="Prefix for the output files")
  parser.add_argument('-d1', '--dele_first_line', action='store', dest='dele', type=int,
                      default=1, help="ignore the first line: 1, keep the first line: 0")
  args = parser.parse_args()

  if args.output == "":
    args.output = os.path.splitext(os.path.basename(args.input_file))[0]

rating2svm(args.input, args.output, args.dele)
