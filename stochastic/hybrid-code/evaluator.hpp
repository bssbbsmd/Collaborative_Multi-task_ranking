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
    virtual void evaluateAUC(const Model&) {}
    virtual void load_files(const std::string&, const std::string&, std::vector<int>&) = 0;
    virtual int get_nusers() {}
    virtual int get_nitems() {} 	     
	    
    std::vector<int> k;
    int k_max;
};


class EvaluatorRating : public Evaluator {
  TestMatrix test;
  
  public:
    void load_files(const std::string&, const std::string&, std::vector<int>&);
    void evaluate(const Model&);
    int get_nusers() {return test.n_users;}
    int get_nitems() {return test.n_items;}	
};

void EvaluatorRating::load_files (const std::string& itemwise_test_file, const std::string& userwise_test_file, std::vector<int>& ik) {
  	test.read_lsvm_itemwise(itemwise_test_file);
  	test.read_lsvm_userwise(userwise_test_file);
  	test.compute_user_dcgmax(10);
  	test.compute_item_dcgmax(10);
  	k = ik;
  	std::sort(k.begin(), k.end());
  	k_max = k[k.size()-1];
}

void EvaluatorRating::evaluate(const Model& model) {
	//double err  = compute_pairwiseError(test, model);
  	double ndcg = compute_ndcg(test, model);
  	printf("NDCG@10=%f", err, ndcg);
}

#endif
