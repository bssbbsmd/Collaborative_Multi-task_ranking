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
#include "loss.hpp"

using namespace std;

/**Reminds:  the input ids (new_user_id = original_user_id - 1 and item_id) starts from 0 **/

class Problem {
  public: 
    int n_users, n_items, n_itemwise_train_comps, n_userwise_train_comps, n_train_rating; // number of users/items in training sample, number of samples in traing and testing data set
    double lambda;

    loss_option_t loss_option = L2_HINGE;

    vector<rating>       rating_train;    // the rating vector 
    vector<comparison>   userwise_train;  // user pairs
    vector<comparison>   itemwise_train;  // item pairs
    vector<int>          n_pairs_by_user_u;  // contain the number of pairs for each user
    vector<int>          n_pairs_by_user_i;  // contain the number of pairs for each item
    vector<int>          n_pairs_by_item_u;  // contain the number of pairs for each user
    vector<int>          n_pairs_by_item_i;  // contain the number of pairs for each item

    Problem();
    Problem(loss_option_t, double);				// default constructor
    ~Problem();					// default destructor

    void read_data_rating(const std::string&);
    void read_data_userwise(const std::string&);	// read function
    void read_data_itemwise(const std::string&);

    int get_nusers() { return n_users; }
    int get_nitems() { return n_items; }

    void print_training_data_info();
    
    double total_loss(Model& model);
};

// may be more parameters can be specified here
Problem::Problem() {
}

Problem::Problem (loss_option_t option, double l) : lambda(l), loss_option(option) { 
}

Problem::~Problem () {
}

void Problem::print_training_data_info(){
    int total_training_pairs_user = 0;
    for(int i=0; i< n_users; i++){
        int uid = i+1;
        total_training_pairs_user += n_pairs_by_user_u[i];
        //cout << "User ["<<uid<<"] has "<< n_pairs_by_user[i]<< " training itemwise pairs"<< endl; 
    
    }
    cout << "User has totally "<< total_training_pairs_user<< " training itemwise pairs"<< endl;     
    int total_training_pairs_item = 0;
    for(int i=0; i<n_items;i++){
        int iid=i+1;
        total_training_pairs_item += n_pairs_by_item_i[i];
     //   cout << "Item ["<<iid<<"] has "<< n_pairs_by_item[i]<< " training userwise pairs"<< endl;   
    }
    cout << "Item has totally "<< total_training_pairs_item << " training userwise pairs"<< endl << endl;
}


void Problem::read_data_rating(const std::string &rating_file){
  n_users = n_items = 0;
  ifstream f;

  // Read training ratings
  f.open(rating_file);

  if (f.is_open()) {
    int uid, iid;
    double score;
    //the first line contains two numers: n_users and n_items
    f >> n_users >> n_items;
    string t_str;
    getline(f, t_str);

   // for(int i=0; i<n_users; i++)  n_pairs_by_user[i] = 0;

    while (f >> uid >> iid >> score) { 
      // now user_id and item_id starts from 0
      --uid; 
      --iid;       
      rating_train.push_back(rating(uid, iid, score));
    }
    n_train_rating = rating_train.size();
  } else {
    printf("Error in opening the training file!\n");
    exit(EXIT_FAILURE);
  }
  f.close();
  printf("%d users, %d items, %d training ratings\n", n_users, n_items, n_train_rating);
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

    n_pairs_by_user_u.resize(n_users,0);
    n_pairs_by_user_i.resize(n_items,0);

   // for(int i=0; i<n_users; i++)  n_pairs_by_user[i] = 0;

    while (f >> uid >> i1id >> i1_r >> i2id >> i2_r) { 
      // now user_id and item_id starts from 0
      --uid; --i1id; --i2id; 
      n_pairs_by_user_u[uid] = n_pairs_by_user_u[uid]+1;
      n_pairs_by_user_i[i1id] = n_pairs_by_user_i[i1id]+1;
      n_pairs_by_user_i[i2id] = n_pairs_by_user_i[i2id]+1;       
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

    n_pairs_by_item_i.resize(n_items,0);
    n_pairs_by_item_u.resize(n_users,0);

  //  for(int i=0; i<n_items; i++)  n_pairs_by_item[i] = 0;

    while (f >> iid >> u1id >> u1_r >> u2id >> u2_r) {
      // now user_id and item_id starts from 0
      --iid; --u1id; --u2id; 
      n_pairs_by_item_i[iid] = n_pairs_by_item_i[iid]+1;
      n_pairs_by_item_u[u1id] = n_pairs_by_item_u[u1id]+1;
      n_pairs_by_item_u[u2id] = n_pairs_by_item_u[u2id]+1;      
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


double Problem::total_loss(Model& model) {
 // double l = compute_loss(model, train, loss_option);
  double l = 0.;
  double u = model.Unormsq();
  double v = model.Vnormsq();
  double f = l + lambda*(u+v);
  return f;
}


#endif
