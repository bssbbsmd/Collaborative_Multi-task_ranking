/***************************************
*evaluate.h contains the function to 
*calculate the measure for test data
***************************************/

#ifndef __EVALUATE_H__
#define __EVALUATE_H__

#include <utility>
#include <vector>
#include <algorithm>
#include <stdlib.h>
#include <math.h>


bool compare_score(double i, double j) {
    return (i>j);
}

std::vector<double> compute_dcg_max(const std::vector<std::vector<std::pair<int, double>>> &test_ratings, int ndcg_k){
    std::vector<double> dcg_max(0);
    int n_users = test_ratings.size();
    dcg_max.resize(n_users, 0.);
	        
    std::vector<double> user_scores(0);

    for(int uid=0; uid < n_users; uid++){
	int n_items = test_ratings[uid].size();
	for( int i=0; i < n_items; i++)
	    user_scores.push_back(test_ratings[uid][i].second); 
	std::sort(user_scores.begin(), user_scores.end(), compare_score);
    	for(int k=1; k<=ndcg_k; k++) {
	    dcg_max[uid] += (double)(std::pow(2,user_scores[k-1]) - 1.) / std::log2(k+1); 	
	}
        user_scores.clear();
    }
    return dcg_max;	
}


//calculate the ndcg for each user using the idea of bubble sort
double compute_user_ndcg(int uid, const std::vector<double>& score, const std::vector<std::vector<std::pair<int, double>>> &test_ratings, const std::vector<double> &dcg_max, int ndcg_k) { 
    std::vector<std::pair<double,int> > ranking;

    for(unsigned int j=0; j < score.size(); ++j) 
	ranking.push_back(std::pair<double,int>(score[j],0));

    double min_score = ranking[0].first;

    for(unsigned int j=0; j < ranking.size(); ++j) {
	min_score = std::min(min_score, ranking[j].first);
    }

    double dcg = 0.;
    for(int k=1; k<=ndcg_k; ++k) {
    	int topk_idx = -1;
    	double max_score = min_score - 1.;

    	//bubble sort 
   	for(unsigned int j = 0; j<ranking.size(); ++j) {
      	    if ((ranking[j].second == 0) && (ranking[j].first > max_score)) {
            	max_score = ranking[j].first;
        	topk_idx = j;
            }
    	}
    	ranking[topk_idx].second = k;
     
    //	dcg += (double)(pow(2,ratings[idx[uid]+topk_idx].score) - 1) / log2((double)(k+1));
	dcg += (double)(std::pow(2, test_ratings[uid][topk_idx].second) -1) / std::log2((double)(k+1));
    }
    return dcg / dcg_max[uid];
} 



double compute_ndcg(const std::vector<std::vector<std::pair<int, double>>> &test_ratings, const std::vector<std::vector<double>> &users_features, const vector<vector<double>> &items_features, int ndcg_k) {
       
    int n_users = test_ratings.size();
    int rank = users_features[0].size();
    if(n_users != (int) users_features.size()) {
	std::cout << "Error, the number of test users does not match" << std::endl; 
	std::exit(0);
    }
    // int n_items = items_features.size();
    // std::cout << "[test file] number of users: "<<n_users << " items: "<< n_items << std::endl;    
    	
    //compute the DCG max for all users
    std::vector<double> dcg_max = compute_dcg_max(test_ratings, ndcg_k);
    //std::cout << "compute dcg max ...... done" << std::endl;
    
    double ndcg_sum = 0.;
    std::vector<double> score;    
    
    for(int uid= 0; uid < n_users; uid++){
//	double dcg = 0.;
 
	score.clear();

	//calculate the dcg for each user
	for(unsigned int i=0; i < test_ratings[uid].size(); i++){
            
	    int iid = test_ratings[uid][i].first-1;
	    if( iid < static_cast<int>(items_features.size())){
		double prod = 0.;
		for(int k=0; k < rank; k++){
		    prod += users_features[uid][k] * items_features[iid][k];
		}
		score.push_back(prod);
	    }else{
		score.push_back(-1e10);
	    }
	}
	//computer the ndcg value for each user
       // std::cout << "Starting calculate ndcg for user: " << uid << ".........";
	ndcg_sum += compute_user_ndcg(uid, score, test_ratings, dcg_max, ndcg_k);
	//std::cout << "done" << std::endl;	
    }

    return ndcg_sum / (double)n_users;	
}

#endif

