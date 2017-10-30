'''Generate training set and test set files (user, item, rating) triples from '''
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

#rating list formated as ["1:3.0", "4:3.0"]
def write_rating(f, uid, rating_list): 
  for entry in rating_list:
    tmp = entry.split(':')
    iid = tmp[0]
    rating = tmp[1]
    print(str(uid)+" "+iid+" "+rating, file=f)

def rating2svm(input, output, dele):
  triples_list = []
  n_users = 0
  n_items = 0

  f = open(input, 'r')
  g1 = open(output+'.rating', 'w')

  if dele==1: #IGNORE THE FIRST line
    first_line = f.readline()

  for line in f:
    rating_list = line.strip().split()
    n_users = n_users + 1
    write_rating(g1, n_users, rating_list)
  f.close()
  g1.close()

  print("Dataset for {0} users.".format(n_users)) 

  #######################################################################
  ## obtain the train pairs for each item:


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--input_file', action='store', dest='input',
                      default="", help="Prefix for the input files")
  parser.add_argument('-o', '--out_file', action='store', dest='output',
                      default="", help="Prefix for the output files")
  parser.add_argument('-d1', '--delete_first_line', action='store', dest='dele', type=int,
                      default=0, help="ignore the first line: 1, keep the first line: 0")
  args = parser.parse_args()

  if args.output == "":
    args.output = os.path.splitext(os.path.basename(args.input_file))[0]

rating2svm(args.input, args.output, args.dele)
