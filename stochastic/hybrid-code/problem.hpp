#ifndef __PROBLEM_HPP__
#define __PROBLEM_HPP__

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include <fstream>

#include "elements.hpp"
#include "loss2.hpp"

using namespace std;

/**Reminds:  the input ids (new_user_id = original_user_id - 1 and item_id) starts from 0 **/

class Problem {
  public: 
    int n_users, n_items, n_itemwise_train_comps, n_userwise_train_comps; // number of users/items in training sample, number of samples in traing and testing data set
    double lambda;

    loss_option_t loss_option = L2_HINGE;

    vector<comparison>   userwise_train;  // user pairs
    vector<comparison>   itemwise_train;  // item pairs
    vector<int>          n_pairs_by_user;  // contain the number of pairs for each item
    vector<int>          n_pairs_by_item;  // contain the number of pairs for each user

    Problem();
    Problem(loss_option_t, double);				// default constructor
    ~Problem();					// default destructor
    void read_data_userwise(const std::string&);	// read function
    void read_data_itemwise(const std::string&);
  
    int get_nusers() { return n_users; }
    int get_nitems() { return n_items; }
    
    double evaluate(Model& model);
};

// may be more parameters can be specified here
Problem::Problem() {
}

Problem::Problem (loss_option_t option, double l) : lambda(l), loss_option(option) { 
}

Problem::~Problem () {
}

void Problem::read_data_itemwise(const std::string &train_file) {
  // Prepare to read files
  n_users = n_items = 0;
  ifstream f;

  // Read training comparisons
  f.open(train_file);

  if (f.is_open()) {
    int uid, i1id, i2id;
    double i1_r, i2_r;
    //the first line contains two numers: n_users and n_items
    f >> n_users >> n_items;
    string t_str;
    getline(f, t_str);

    n_pairs_by_user.resize(n_users);

    for(int i=0; i<n_users; i++)  n_pairs_by_user[i] = 0;

    while (f >> uid >> i1id >> i1_r >> i2id >> i2_r) { 
      // now user_id and item_id starts from 0
      --uid; --i1id; --i2id; 
      n_pairs_by_user[uid] = n_pairs_by_user[uid]+1;      
      itemwise_train.push_back(comparison(uid, i1id, i1_r, i2id, i2_r, 1));
    }
    n_itemwise_train_comps = itemwise_train.size();
  } else {
    printf("Error in opening the training file!\n");
    exit(EXIT_FAILURE);
  }
  f.close();
  printf("%d users, %d items, %d itemwise comparisons\n", n_users, n_items, n_itemwise_train_comps);
}	

void Problem::read_data_userwise(const std::string &train_file) { 
  n_users = n_items = 0;  // Prepare to read files
  ifstream f;

  f.open(train_file);     // Read training comparisons

  if (f.is_open()) {
    int iid, u1id, u2id, uid_current = 0;
    double u1_r, u2_r;
    //the first line contains two numers: n_users and n_items
    
    f >> n_users >> n_items;
    
    string t_str;
    getline(f, t_str);

    n_pairs_by_item.resize(n_items);

    for(int i=0; i<n_items; i++)  n_pairs_by_item[i] = 0;

    while (f >> iid >> u1id >> u1_r >> u2id >> u2_r) {
      // now user_id and item_id starts from 0
      --iid; --u1id; --u2id; 
      n_pairs_by_item[iid] = n_pairs_by_item[iid]+1;      
      userwise_train.push_back(comparison(iid, u1id, u1_r, u2id, u2_r, 1));
    }
    n_userwise_train_comps = userwise_train.size();
  } else {
    printf("Error in opening the training file!\n");
    exit(EXIT_FAILURE);
  }
  f.close();
  printf("%d users, %d items, %d userwise comparisons\n", n_users, n_items, n_userwise_train_comps);
} 

/*
double Problem::evaluate(Model& model) {
  double l = compute_loss(model, train, loss_option);
  double u = model.Unormsq();
  double v = model.Vnormsq();
  double f = l + lambda*(u+v);
  return f;
}*/


#endif
