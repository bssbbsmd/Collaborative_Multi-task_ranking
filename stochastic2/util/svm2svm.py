'''Generate training set and test set files (user, item, rating) triples'''
from __future__ import print_function
import os
import sys
import random
import argparse
import itertools

def write_lsvm(f, ratings_list): 
  line=""
  for item in ratings_list:
    line = line + item + " "
  print(line, file=f)

def svm2svm(input, output, dele):
  triples_list = []
  n_users = 0
  n_items = 0

  f = open(input, 'r')
  g1 = open(output+'.svm', 'w')

  if dele==1: #IGNORE THE FIRST line
    first_line = f.readline()

  for line in f:
    split_line = line.strip().split()
    write_lsvm(g1, split_line[1:])
  f.close()
  g1.close()

  print("Dataset for {0} users, {1} items loaded.".format(n_users, n_items)) 


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

svm2svm(args.input, args.output, args.dele)
