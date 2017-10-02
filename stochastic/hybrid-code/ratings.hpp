#ifndef __RATINGS_H__
#define __RATINGS_H__

#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include <utility>
#include <map>
#include <iostream>
#include <functional>   // std::greater
#include <fstream>
#include <sstream>

#include "elements.hpp"
#include "model.hpp"

using namespace std;

class TestMatrix {
  public:
    int                   n_users, n_items;
    int                   ndcg_k = 0;
    int                   n_userwise_test_pairs=0;
    int                   n_itemwise_test_pairs=0;

    vector<vector<pair<int, double>>>   userwise_test_pairs;
    vector<vector<pair<int, double>>>   itemwise_test_pairs;
    vector<double> user_dcg_max;
    vector<double> item_dcg_max;

    TestMatrix() : n_users(0), n_items(0) {}
    TestMatrix(int nu, int ni): n_users(nu), n_items(ni) {}

    
    void   compute_user_dcgmax(int);     //for each user i.e. itemwise
    double compute_user_ndcg(int, const std::vector<double>&) const; //for each user i.e. itemwise
    void   compute_item_dcgmax(int);     //for each item i.e. userwise
    double compute_item_ndcg(int, const std::vector<double>&) const; //for each item i.e. userwise

    void read_lsvm_itemwise(const std::string&);
    void read_lsvm_userwise(const std::string&);
};

std::ifstream::pos_type filesize(const char* filename)
{
    std::ifstream in(filename, std::ios::binary | std::ios::ate);
    return in.tellg(); 
}

void TestMatrix::read_lsvm_itemwise(const std::string& filename) {
  string user_str, attribute_str;
  stringstream attribute_sstr;

  ifstream f;
  f.open(filename);

  if(f.is_open()){
    f >> n_users >> n_items;
    itemwise_test_pairs.resize(n_users);
    getline(f, user_str);
    while(!f.eof()) {
      getline(f, user_str);

      size_t pos1 = 0, pos2;
      int uid;

      pos2 = user_str.find(' ', pos1); 
      if(pos2 == std::string::npos) break;
      attribute_str = user_str.substr(pos1, pos2-pos1);
      attribute_sstr.clear(); 
      attribute_sstr.str(attribute_str);
      attribute_sstr >> uid;
      uid--;  // let uid start from 0

      int iid;
      double sc;
      pos1 = pos2+1;

      while(1){
            pos2 = user_str.find(':', pos1);        
            if(pos2 == std::string::npos) break;
            attribute_str = user_str.substr(pos1, pos2-pos1);
            attribute_sstr.clear();
            attribute_sstr.str(attribute_str);
            attribute_sstr >> iid;  
            iid--;  // then iid starts from 0

            pos1 = pos2+1;
            pos2 = user_str.find(' ', pos1);
            attribute_str = user_str.substr(pos1, pos2-pos1);
            attribute_sstr.clear();
            attribute_sstr.str(attribute_str);
            attribute_sstr >> sc;
            pos1 = pos2+1;  

            itemwise_test_pairs[uid].push_back(make_pair(iid, sc));
            n_itemwise_test_pairs++;
      }
    }
  }
  f.close();
}

void TestMatrix::read_lsvm_userwise(const std::string& filename) {
  string item_str, attribute_str;
  stringstream attribute_sstr;

  ifstream f;
  f.open(filename);

  if(f.is_open()){
    f >> n_users >> n_items;
    userwise_test_pairs.resize(n_items);
    getline(f, item_str);
    while(!f.eof()) {
      getline(f, item_str);
      size_t pos1 = 0, pos2;
      
      int iid;
      pos2 = item_str.find(' ', pos1); 
      if(pos2 == std::string::npos) break;
      attribute_str = item_str.substr(pos1, pos2-pos1);
      attribute_sstr.clear(); 
      attribute_sstr.str(attribute_str);
      attribute_sstr >> iid;
      iid--;  // let iid start from 0

      int uid;
      double sc;
      pos1 = pos2+1;

      while(1){
            pos2 = item_str.find(':', pos1);        
            if(pos2 == std::string::npos) break;
            attribute_str = item_str.substr(pos1, pos2-pos1);
            attribute_sstr.clear();
            attribute_sstr.str(attribute_str);
            attribute_sstr >> uid;  
            uid--;  // then uid starts from 0

            pos1 = pos2+1;
            pos2 = item_str.find(' ', pos1);
            attribute_str = item_str.substr(pos1, pos2-pos1);
            attribute_sstr.clear();
            attribute_sstr.str(attribute_str);
            attribute_sstr >> sc;
            pos1 = pos2+1;  

            userwise_test_pairs[iid].push_back(make_pair(uid, sc));
            n_userwise_test_pairs++;
      }
    }
  }
  f.close();
}


void TestMatrix::compute_user_dcgmax(int ndcgK) {
  ndcg_k = ndcgK;
 // vector<double> user_dcg_max(0)
  user_dcg_max.resize(n_users, 0.);  

  vector<double> ratings_current_user(0);

  for(int uid=0; uid < n_users; ++uid) {
    if(!itemwise_test_pairs[uid].empty()){
      for(int i=0; i < itemwise_test_pairs[uid].size(); ++i) {
        ratings_current_user.push_back(itemwise_test_pairs[uid][i].second);
      } 

      std::sort(ratings_current_user.begin(), ratings_current_user.end(), std::greater<double>()); // sort in decreasing order
      
      for(int k=1; k<=ndcg_k; ++k) {
        user_dcg_max[uid] += (double)(pow(2, ratings_current_user[k-1]) - 1.) / log2(k+1); 
      }      
      ratings_current_user.clear();
    }
  }
}

void TestMatrix::compute_item_dcgmax(int ndcgK) {
  ndcg_k = ndcgK;
 // vector<double> item_dcg_max(0)
  item_dcg_max.resize(n_items, 0.);  
  vector<double> ratings_current_item(0);

  for(int iid=0; iid < n_items; ++iid) {
    if(!userwise_test_pairs[iid].empty()){
      for(int i=0; i < userwise_test_pairs[iid].size(); ++i) {
        ratings_current_item.push_back(userwise_test_pairs[iid][i].second);
      }
      std::sort(ratings_current_item.begin(), ratings_current_item.end(), std::greater<double>());

      for(int k=1; k <= ndcg_k; ++k) {
        item_dcg_max[iid] += (double)(pow(2, ratings_current_item[k-1]) - 1.) / log2(k+1); 
      }
      ratings_current_item.clear();
    }
  }
}

double TestMatrix::compute_user_ndcg(int uid, const std::vector<double>& score) const {
  std::vector<std::pair<double,int> > ranking;
  
  for(int j=0; j<score.size(); ++j) 
    ranking.push_back(std::pair<double,int>(score[j], 0));

  double min_score = ranking[0].first;
  for(int j=0; j<ranking.size(); ++j) 
    min_score = std::min(min_score, ranking[j].first);

  double dcg = 0.;
  for(int k=1; k<=ndcg_k; ++k) {
    int topk_idx = -1;
    double max_score = min_score - 1.;

    //bubble sort
    for(int j=0; j<ranking.size(); ++j) {
      if ((ranking[j].second == 0) && (ranking[j].first > max_score)) {
        max_score = ranking[j].first;
        topk_idx = j;
      }
    }
    ranking[topk_idx].second = k;
    
    dcg += (double)(pow(2, itemwise_test_pairs[uid][topk_idx].second) -1) / std::log2((double)(k+1));
  //  dcg += (double)(pow(2,ratings[idx[uid]+topk_idx].score) - 1) / log2((double)(k+1));
  }

  return dcg / user_dcg_max[uid];
} 

double TestMatrix::compute_item_ndcg(int iid, const std::vector<double>& score) const {
  std::vector<std::pair<double,int> > ranking; 

  std::cout << score.size() <<std::endl;
  if(userwise_test_pairs[iid].size() != score.size()){
    std::cout<< "pairs does not match" << std::endl;
  }

  for(int j=0; j < score.size(); ++j) 
    ranking.push_back(std::pair<double,int>(score[j], 0));

  double min_score = ranking[0].first;

  for(int j=0; j<ranking.size(); ++j) 
    min_score = std::min(min_score, ranking[j].first);

  double dcg = 0.;
  for(int k=1; k<=ndcg_k; ++k) {
    int topk_idx = -1;
    double max_score = min_score - 1.;
    //bubble sort
    for(int j=0; j<ranking.size(); ++j) {
      if ((ranking[j].second == 0) && (ranking[j].first > max_score)) {
        max_score = ranking[j].first;
        topk_idx = j;
      }
    }
    ranking[topk_idx].second = k;    
    dcg += (double)(pow(2, userwise_test_pairs[iid][topk_idx].second) -1) / std::log2((double)(k+1));
  //  dcg += (double)(pow(2,ratings[idx[uid]+topk_idx].score) - 1) / log2((double)(k+1));
  }

  return dcg / item_dcg_max[iid];
} 

#endif
