#ifndef __EVALUATOR_HPP__
#define __EVALUATOR_HPP__

#include <utility>
#include <vector>
#include <algorithm>
#include <queue>
#include <iostream>
#include <fstream>
#include <unordered_set>

#include "model.hpp"
#include "ratings.hpp"
#include "loss.hpp"
	
class Evaluator {
  public: 
    virtual void evaluate(const Model&) {} 
    virtual void load_files(const std::string&, const std::string&, std::vector<int>&) = 0;
    virtual int get_nusers() {}
    virtual int get_nitems() {} 	     
};


class EvaluatorRating : public Evaluator {
  TestMatrix test;
  std::vector<int> k;
  
  public:
    void load_files(const std::string&, const std::string&, std::vector<int>&);
    void evaluate(const Model&);
    int get_nusers() {return test.n_users;}
    int get_nitems() {return test.n_items;}	
};

void EvaluatorRating::load_files (const std::string& itemwise_test_file, const std::string& userwise_test_file, std::vector<int>& ik) {
  	test.read_lsvm_itemwise(itemwise_test_file);
  	test.read_lsvm_userwise(userwise_test_file);
  	std::cout << "N_users:" << test.n_users << " N_items:"<<test.n_items<<" N_itemwise_test_ratings:"
  			<<test.n_itemwise_test_pairs << " N_userwise_test_ratings:"<< test.n_userwise_test_pairs<<std::endl;

  	test.compute_user_dcgmax(ik[0]);
  	test.compute_item_dcgmax(ik[0]);

  	std::cout << ">> Calculate DCG-max for each user and item, done!!!" << std::endl;

  	k = ik;
  	std::sort(k.begin(), k.end());
}

void EvaluatorRating::evaluate(const Model& model) {
	//double err  = compute_pairwiseError(test, model);
	//std::cout << ">>>>>>>> start evaluating ... ";
  	//double ndcg = compute_ndcg(test, model, 0);
  	double ndcg1 = compute_ndcg(test, model, 1);
  	//std::cout<<"PR NDCG@"<<k[0]<<"="<<ndcg;
  	std::cout<<"\t UT M=NDCG@"<<k[0]<<"="<<ndcg1<<std::endl; 
}

#endif
