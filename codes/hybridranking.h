#ifndef __HYBRIDRANKING_H_
#define __HYBRIDRANKING_H_

#include <vector>
#include <utility>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <utility>
#include <vector>
#include <iostream>
#include <stdlib.h>

#include "./evaluate.h"
#include "./InputData.h"

using namespace std;

class HybridRanking{
    public:
    string f_lsvm, l_pair, test;

    int ndcg_k;
    int rank;
    int max_iterations;
    double alpha, beta;
    double learning_rate, lambda, relative_learning_rate;
    double eps;    
    bool initialized;

    int n_users, n_items;
    InputData input;

    int pairwise_method;

    vector<vector<double>> items_features;
    vector<vector<double>> users_features; 
    
    bool show_seperate_loss;
    bool show_each_iteration;
 // HybridRanking(): n_users(0), n_items(0){}
    HybridRanking(){
        n_users = 0;
        n_items = 0;
  	ndcg_k = 10;
  	rank = 10;
  	max_iterations = 100;
        alpha = 1.;
	beta = 0.;
	learning_rate = 0.1;
	relative_learning_rate = 1.;
        lambda = 1.; 
	eps = 1e-2;
 	initialized = true;
	show_seperate_loss = false;
	show_each_iteration= true;

	pairwise_method = 1;
    };
    
    void read_input();
        
    void randomly_initialize(vector<vector<double>> &features);
    
    double predict(const int uid, const int iid); //

    double predict(const vector<double> &user, const vector<double> &item);

    double compute_loss(const vector<vector<pair<int, double>>> &ratings_matrix, const vector<vector<pair<int, int>>> &pairs_matrix, const vector<vector<double>> &users_features1, const vector<vector<double>> &items_features1);
    
    void compute_gradient_ui_point(int uid, const vector<vector<pair<int, double>>> &ratings_matrix,  vector<double> &ui_prime);
    
    void compute_gradient_vj_point(int iid, const vector<vector<pair<int, double>>> &ratings_matrix_t, vector<double> &vj_prime);    
    
    void compute_gradient_ui_pair(int uid, const vector<vector<pair<int, int>>>  &ratings_matrix_comp, vector<double> &ui_prime);
    
    void compute_gradient_vj_pair(int iid, const vector<vector<pair<int, double>>> &ratings_matrix, const vector<vector<pair<int, double>>> &ratings_matrix_t, vector<double> &vj_prime); 
  
    void compute_gradient_ui_list(int uid, const vector<vector<pair<int, double>>> &ratings_matrix, vector<double> &ui_prime);
    
    void compute_gradient_vj_list(int iid, const vector<vector<pair<int, double>>> &ratings_matrix, const vector<vector<pair<int, double>>> &ratings_matrix_t, vector<double> &vj_prime);
  
    void update(const vector<vector<pair<int, double>>> &ratings_matrix, const vector<vector<pair<int, int>>> &ratings_matrix_comp);
    
    void train();
    
    void copy_matrix(vector<vector<double>> &a, const vector<vector<double>> &b);   //copy b to a;
};

void HybridRanking::read_input(){
    input.readInputTrain(f_lsvm, l_pair);
    input.readInputTest(test); 
    n_users = input.n_users;
    n_items = input.n_items;
}

void HybridRanking::randomly_initialize(vector<vector<double>> &features) {
    srand (time(NULL));
    // ``Good range'' defined by experiments
    for (auto &v : features) {
        for (auto &f : v) {
            f = (double)rand()/ (double)RAND_MAX / sqrt((double)rank);
        }
    }
}

void HybridRanking::copy_matrix(vector<vector<double>> &a, const vector<vector<double>> &b){
    for(unsigned i=0; i < b.size(); i++)
        for(int k=0; k < rank; k++){
            a[i][k] = b[i][k];
        }    
}

double HybridRanking::predict(const int uid, const int iid){
    double score = 0.;
    for(int i=0; i < rank; i++)
	    score += users_features[uid][i] * items_features[iid][i];
    return score;
}

double HybridRanking::predict(const vector<double> &user, const vector<double> &item){
    double score = 0.;
    for(int i=0; i < rank; i++)
	    score += user[i]*item[i];
    return score;
}

double HybridRanking::compute_loss(const vector<vector<pair<int, double>>> &ratings_matrix, const vector<vector<pair<int, int>>> &pairs_matrix, const vector<vector<double>> &users_features1, const vector<vector<double>> &items_features1){
    double loss = 0;
    double point_loss = 0, pair_loss=0, list_loss=0;
    
    if(ratings_matrix.size() != pairs_matrix.size())
	cout << "Read Input Error" << endl;
    //calculate pointwise loss
    for(int uid=0; uid < n_users; uid++)
    	for(unsigned i=0; i < ratings_matrix[uid].size(); i++){
    	    int iid = ratings_matrix[uid][i].first - 1;
    	    double score = ratings_matrix[uid][i].second;
    	    double predict_score = predict(users_features1[uid], items_features1[iid]);
                point_loss += std::pow(score-predict_score, 2);
            }
    if(show_seperate_loss) cout<<"point_loss:"<<point_loss << " ";
    
    //calculate pairwise loss
    for(int uid=0; uid < n_users; uid++)
	    for(unsigned i=0; i < pairs_matrix[uid].size(); i++){
    	    int iid_higher = pairs_matrix[uid][i].first - 1;
    	    int iid_lower  = pairs_matrix[uid][i].second- 1;
    	    double predict_higher_score = predict(users_features1[uid], items_features1[iid_higher]);
    	    double predict_lower_score  = predict(users_features1[uid], items_features1[iid_lower]);
    	    pair_loss -= log(exp(predict_higher_score) / (exp(predict_higher_score) + exp(predict_lower_score)));           
	    }

    if(show_seperate_loss) cout<<"pair_loss:"<<pair_loss << " ";
    //calculate listwise loss
    for(int uid=0; uid < n_users; uid++){
    	double predict_sum = 0;
    	double observe_sum = 0;
    
    	for(unsigned i=0; i < ratings_matrix[uid].size(); i++){
    	    int iid = ratings_matrix[uid][i].first - 1;
    	    double score = ratings_matrix[uid][i].second;
    	    observe_sum += exp(score);
    	    predict_sum += exp(predict(users_features1[uid], items_features1[iid]));
    	}
    
    	for(unsigned i=0; i < ratings_matrix[uid].size(); i++){
    	    int iid = ratings_matrix[uid][i].first - 1;
    	    double score = ratings_matrix[uid][i].second;
    	    list_loss -= exp(score)/observe_sum * log( exp(predict(users_features1[uid], items_features1[iid])) / predict_sum );
    	}
    }
    
    if(show_seperate_loss) cout<<"list_loss:"<<list_loss << " ";
    // add each loss
    loss = alpha * point_loss + beta * pair_loss + (1-alpha-beta) * list_loss;
    
    //add the regularization term to the loss
    for(int uid = 0; uid < n_users; uid++)
	for(int k = 0; k < rank; k++)
	    loss += pow(users_features1[uid][k], 2) * lambda / 2;
    
    for(int iid = 0; iid < n_items; iid++)
	for(int k = 0; k < rank; k++)
	    loss += pow(items_features1[iid][k], 2) * lambda / 2; 
    if(show_seperate_loss) cout<<"Total_loss:"<< loss <<endl;
    return loss; 

}

void HybridRanking::compute_gradient_ui_point(int uid, const vector<vector<pair<int, double>>> &ratings_matrix, vector<double> &ui_prime){
    fill(ui_prime.begin(), ui_prime.end(), 0.0);
    
    for(unsigned i = 0; i < ratings_matrix[uid].size(); i++){
	int iid = ratings_matrix[uid][i].first - 1;
	double score = ratings_matrix[uid][i].second;
	for(int k=0; k < rank; k++){
	    ui_prime[k] -= 2 * (score - predict(uid, iid)) * items_features[iid][k];
	} 
    }
    for(int k=0; k < rank; k++){
   	ui_prime[k] += lambda * users_features[uid][k];
    }   
}

void HybridRanking::compute_gradient_vj_point(int iid, const vector<vector<pair<int, double>>> &ratings_matrix_t, vector<double> &vj_prime){
    fill(vj_prime.begin(), vj_prime.end(), 0.0);

    for(unsigned u=0; u < ratings_matrix_t[iid].size(); u++){
	int uid = ratings_matrix_t[iid][u].first;
	double score = ratings_matrix_t[iid][u].second;
	for(int k=0; k < rank; k++){
	    vj_prime[k] -= 2 * (score - predict(uid, iid)) * users_features[uid][k]; 
	}	
    }
    
 //   if(ratings_matrix_t[iid].size() != 0)
    for(int k=0; k < rank; k++){
  	    vj_prime[k] += lambda * items_features[iid][k]; 
    }
}

void HybridRanking::compute_gradient_ui_pair(int uid, const vector<vector<pair<int, int>>>  &ratings_matrix_comp, vector<double> &ui_prime){
    fill(ui_prime.begin(), ui_prime.end(), 0.0);	

    for(unsigned i=0; i < ratings_matrix_comp[uid].size(); i++){
	int iid_higher = ratings_matrix_comp[uid][i].first - 1;
	int iid_lower = ratings_matrix_comp[uid][i].second - 1;
        for(int k=0; k < rank; k++){
	    ui_prime[k] -= (1./(1.+ exp(predict(uid, iid_higher)-predict(uid, iid_lower))))*(items_features[iid_higher][k] - items_features[iid_lower][k]);
	}
    }

    for(int k=0; k < rank; k++){ 
	ui_prime[k] += lambda * users_features[uid][k];
    }
}
    
void HybridRanking::compute_gradient_vj_pair(int iid, const vector<vector<pair<int, double>>> &ratings_matrix, const vector<vector<pair<int, double>>> &ratings_matrix_t, vector<double> &vj_prime){
    fill(vj_prime.begin(), vj_prime.end(), 0.0);
    
    for(unsigned u =0; u < ratings_matrix_t[iid].size(); u++){
	int uid = ratings_matrix_t[iid][u].first;  //uid start from 0
	double i_score = ratings_matrix_t[iid][u].second;        
        
        double u_sum = 0.;
	for(unsigned i=0; i < ratings_matrix[uid].size(); i++){
	    int jid = ratings_matrix[uid][i].first-1;
	    double j_score = ratings_matrix[uid][i].second;
	    if(j_score > i_score){
		u_sum += 1./(1. + exp(predict(uid, jid)-predict(uid, iid)));
	    }
            if(j_score < i_score){
		u_sum -= 1./(1. + exp(predict(uid, iid)-predict(uid, jid)));
	    }
	}
	
	for(int k=0; k < rank; k++){
	    vj_prime[k] += u_sum * users_features[uid][k];
	}
    }
     
    for(int k=0; k < rank; k++){
 	vj_prime[k] += lambda * items_features[iid][k]; 
    }        
}  

void HybridRanking::compute_gradient_ui_list(int uid, const vector<vector<pair<int, double>>> &ratings_matrix, vector<double> &ui_prime){
    fill(ui_prime.begin(), ui_prime.end(), 0.0);
    
    double predict_sum = 0.;
    double observe_sum = 0.;
    
    for(unsigned i=0; i < ratings_matrix[uid].size(); i++){
        int iid = ratings_matrix[uid][i].first - 1;
	    double score = ratings_matrix[uid][i].second;
	    observe_sum += exp(score);
	    predict_sum += exp(predict(uid, iid));
    }
    
    for(unsigned i=0; i < ratings_matrix[uid].size(); i++){
        int iid = ratings_matrix[uid][i].first-1;
        double score = ratings_matrix[uid][i].second;
        
        double multiplier = exp(predict(uid, iid))/predict_sum - exp(score)/observe_sum; 
        
        for(int k=0; k < rank; k++){
            ui_prime[k] += multiplier * items_features[iid][k];
        }
    }
    
    for(int k=0; k < rank; k++){
        ui_prime[k] += users_features[uid][k] * lambda;
    }
}
    
void HybridRanking::compute_gradient_vj_list(int iid, const vector<vector<pair<int, double>>> &ratings_matrix, const vector<vector<pair<int, double>>> &ratings_matrix_t, vector<double> &vj_prime){
    fill(vj_prime.begin(), vj_prime.end(), 0.0);
    
    for(unsigned u=0; u < ratings_matrix_t[iid].size(); u++){
        int uid = ratings_matrix_t[iid][u].first;
        int score = ratings_matrix_t[iid][u].second;
        
        double predict_sum = 0.;
        double observe_sum = 0.;
        
        for(unsigned i=0; i < ratings_matrix[uid].size(); i++){
            observe_sum += exp(ratings_matrix[uid][i].second);
            predict_sum += exp(predict(uid, ratings_matrix[uid][i].first-1));
        }
        
        double multiplier = exp(predict(uid, iid))/predict_sum - exp(score)/observe_sum; 
        
        for(int k=0; k < rank; k++){
            vj_prime[k] += multiplier * users_features[uid][k];
        }
    }
    
    for(int k=0; k < rank; k++){
        vj_prime[k] += lambda * items_features[iid][k];
    }
}

void HybridRanking::update(const vector<vector<pair<int, double>>> &ratings_matrix, const vector<vector<pair<int, int>>> &ratings_matrix_comp){
    
    // get the transpose of ratings matrix; so all the users who rated item j can be obtained
    vector<vector<pair<int, double>>> ratings_matrix_t;    
    for(int uid=0; uid < n_users; uid++)
	    for ( const auto &rating : ratings_matrix[uid]){
    	    	if(ratings_matrix_t.size() <= static_cast<unsigned int>(rating.first)) {
    		    ratings_matrix_t.resize(rating.first);
	        }
	        ratings_matrix_t[rating.first-1].push_back( make_pair(uid, rating.second)); // 0 (iid in ratings_matrix_t) = 1 (iid in ratings_matrix) uid always starts from 0
	    }	
    ratings_matrix_t.shrink_to_fit();
    
    users_features.resize(n_users);
    items_features.resize(n_items);
    
    for (auto &f : users_features)
        f.resize(rank);
    for (auto &f : items_features)
        f.resize(rank);
    // randomly_initialize the users_features U and items_features V
    if(initialized){
        randomly_initialize(users_features);
        randomly_initialize(items_features);
    }
    
    vector<double> ui_prime(rank);
    vector<double> vj_prime(rank);
    
    
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    
    vector<vector<double>> users_features_next(n_users);
    vector<vector<double>> items_features_next(n_items);
    
    for (auto &f : users_features_next)
        f.resize(rank);
    for (auto &f : items_features_next)
        f.resize(rank);
    
    double loss1= compute_loss(ratings_matrix, ratings_matrix_comp, users_features, items_features); 
    double ndcg = compute_ndcg(input.ratings_test, users_features, items_features, ndcg_k);
    std::cout << "Iterations [0]\tloss: "<< loss1 << "\t NDCG: "<<ndcg << endl;

    // coordinate descent
    int it = 0;
    
 //   for(int it = 0; it < max_iterations; it++){
    while(it < max_iterations) {
       // learning_rate = learning_rate * 2.0;
        double z = distribution(generator);
        //****************************************
        //if 0 <= z < alpha update pointwise loss
        //****************************************
        if(0 <=z && z < alpha){
    	    for(unsigned uid=0; uid < ratings_matrix.size(); uid++){
                    compute_gradient_ui_point(uid, ratings_matrix, ui_prime);
                    for(int k=0; k < rank; k++)
                        users_features_next[uid][k] =  users_features[uid][k] - learning_rate * ui_prime[k];
            }
            for(unsigned iid=0; iid < ratings_matrix_t.size(); iid++){
                compute_gradient_vj_point(iid, ratings_matrix_t, vj_prime);
                for(int k=0; k < rank; k++)
                    items_features_next[iid][k] = items_features[iid][k]- learning_rate * vj_prime[k];
            }
            
    	    double loss2 = compute_loss(ratings_matrix, ratings_matrix_comp, users_features_next, items_features_next); 
    	    
    	    while(loss2 > loss1){
		cout<<"learning_rate too large ..."<<endl;
    	        learning_rate = learning_rate / 2.0;
        	for(unsigned uid=0; uid < ratings_matrix.size(); uid++){
                    compute_gradient_ui_point(uid, ratings_matrix, ui_prime);
                    for(int k=0; k < rank; k++)
                        users_features_next[uid][k] =  users_features[uid][k] - learning_rate * ui_prime[k];
                }

                for(unsigned iid=0; iid < ratings_matrix_t.size(); iid++){
                    compute_gradient_vj_point(iid, ratings_matrix_t, vj_prime);
                    for(int k=0; k < rank; k++)
                        items_features_next[iid][k] = items_features[iid][k]-learning_rate * vj_prime[k];
                }
		loss2 = compute_loss(ratings_matrix, ratings_matrix_comp, users_features_next, items_features_next);
    	    }
    	    
    	    copy_matrix(users_features, users_features_next);
    	    copy_matrix(items_features, items_features_next);
    	    
    	    double delta = (loss1-loss2)/loss1; 
    	    if(delta <= eps){
    	        break;
    	    }else{
    	        loss1 = loss2;
    	        it++;
    	        ndcg = compute_ndcg(input.ratings_test, users_features, items_features, ndcg_k);
    	        std::cout << "Iterations ["<< it <<"]\t update pointwise:\tloss: "<< loss1 << "\t NDCG: "<<ndcg << endl;
    	    }
        }
        
        //********************************************************************************************************************
        //if alpha <= z < alpha_beta, update pairwise loss with coordinate descent
        //********************************************************************************************************************
        if(alpha <= z && z < alpha+beta && pairwise_method == 1){
	        cout << "Pair[1] ";
    	    if(show_each_iteration){ 
    	    	cerr << "Iterations ["<< it <<"] U....";
    	    }
            //update Ui
            for(unsigned uid=0; uid < ratings_matrix.size(); uid++){
                compute_gradient_ui_pair(uid, ratings_matrix_comp, ui_prime);
                for(int k=0; k < rank; k++)
                    users_features[uid][k] -= learning_rate * ui_prime[k];
            }
    	    if(show_each_iteration){
                double loss_tmp = compute_loss(ratings_matrix, ratings_matrix_comp, users_features, items_features);
		        double train_ndcg= compute_ndcg(input.ratings_train, users_features, items_features, ndcg_k);
		        double test_ndcg= compute_ndcg(input.ratings_test, users_features, items_features, ndcg_k);
                cout <<"done\t loss="<<loss_tmp<<"\t train_ndcg="<< train_ndcg << "\t test_ndcg="<< test_ndcg << "\nPair[1] Iterations ["<< it <<"] V... ";
    	    }           
            
	    //update Vj
            for(unsigned iid=0; iid < ratings_matrix_t.size(); iid++){
                compute_gradient_vj_pair(iid, ratings_matrix, ratings_matrix_t, vj_prime);
                for(int k=0; k < rank; k++)
                    items_features[iid][k] -= learning_rate * vj_prime[k];
            }
    	    if(show_each_iteration){
    	    	double loss_tmp_2 = compute_loss(ratings_matrix, ratings_matrix_comp, users_features, items_features);
		        double train_ndcg= compute_ndcg(input.ratings_train, users_features, items_features, ndcg_k);
		        double test_ndcg=  compute_ndcg(input.ratings_test, users_features, items_features, ndcg_k);
    	    	cout <<"done\t loss="<<loss_tmp_2 <<"\t train_ndcg="<< train_ndcg <<"\t test_ndcg="<< test_ndcg << endl;
    	    }
	                	
    	    it++;
    	    double loss2 = compute_loss(ratings_matrix, ratings_matrix_comp, users_features, items_features);
    	    
    	    while(loss2>loss1){
    		cout<<"learning_rate too large ..."<<endl;
        	        learning_rate = learning_rate / 2.0;

	    }

//	    ndcg = compute_ndcg(input.ratings_test, users_features, items_features, ndcg_k);
 //   	    std::cout << "Iterations ["<< it <<"]\t update pairwise:\tloss: "<< loss1 << "\t NDCG: "<<ndcg << endl;
        }
       
	//********************************************************************************************************************
        //if alpha <= z < alpha_beta, update pairwise loss with coordinate descent plus update U using regression
        //********************************************************************************************************************
        if(alpha <= z && z < alpha+beta && pairwise_method == 2){
	    cout << "Pair[2]";
    	    if(show_each_iteration){ 
    	    	cerr << " Iterations ["<< it <<"] U....";
    	    }
            //update Ui using Regression Loss:
            //for(unsigned uid=0; uid < ratings_matrix.size(); uid++){
            //    compute_gradient_ui_pair(uid, ratings_matrix_comp, ui_prime);
            //    for(int k=0; k < rank; k++)
            //        users_features[uid][k] -= learning_rate * ui_prime[k];
            //}

            for(unsigned uid=0; uid < ratings_matrix.size(); uid++){
                compute_gradient_ui_point(uid, ratings_matrix, ui_prime);
                for(int k=0; k < rank; k++)
                    users_features[uid][k] =  users_features[uid][k] - relative_learning_rate*learning_rate * ui_prime[k];
            }

    	    if(show_each_iteration){
                double loss_tmp = compute_loss(ratings_matrix, ratings_matrix_comp, users_features, items_features);
		double train_ndcg= compute_ndcg(input.ratings_train, users_features, items_features, ndcg_k);
		double test_ndcg= compute_ndcg(input.ratings_test, users_features, items_features, ndcg_k);
                cout <<"done \t loss="<<loss_tmp<<"\t train_ndcg="<< train_ndcg << "\t test_ndcg="<< test_ndcg << "\nPair[2] Iterations ["<< it <<"] V... ";
    	    }           
            
	    //update Vj
            for(unsigned iid=0; iid < ratings_matrix_t.size(); iid++){
                compute_gradient_vj_pair(iid, ratings_matrix, ratings_matrix_t, vj_prime);
                for(int k=0; k < rank; k++)
                    items_features[iid][k] -= learning_rate * vj_prime[k];
            }
    	    if(show_each_iteration){
    	    	double loss_tmp_2 = compute_loss(ratings_matrix, ratings_matrix_comp, users_features, items_features);
		double train_ndcg= compute_ndcg(input.ratings_train, users_features, items_features, ndcg_k);
		double test_ndcg= compute_ndcg(input.ratings_test, users_features, items_features, ndcg_k);
    	    	cout <<"done \t loss="<<loss_tmp_2 <<"\t train_ndcg="<< train_ndcg <<"\t test_ndcg="<< test_ndcg << endl;
    	    }
	                	
	    it++;
	    double loss2 = compute_loss(ratings_matrix, ratings_matrix_comp, users_features, items_features);
	    
	    if(loss2>loss1){
		cout<<"learning_rate too large ..."<<endl;
    	        learning_rate = learning_rate / 2.0;

	    }

        }
        
	//********************************************************************************************************************
        //if alpha <= z < alpha_beta, update pairwise loss with coordinate descent. Only update V
        //********************************************************************************************************************
        if(alpha <= z && z < alpha+beta && pairwise_method == 3){    	    
            //update Ui using Regression Loss:
            //for(unsigned uid=0; uid < ratings_matrix.size(); uid++){
            //    compute_gradient_ui_pair(uid, ratings_matrix_comp, ui_prime);
            //    for(int k=0; k < rank; k++)
            //        users_features[uid][k] -= learning_rate * ui_prime[k];
            //}
/*
            for(unsigned uid=0; uid < ratings_matrix.size(); uid++){
                compute_gradient_ui_point(uid, ratings_matrix, ui_prime);
                for(int k=0; k < rank; k++)
                    users_features[uid][k] =  users_features[uid][k] - relative_learning_rate*learning_rate * ui_prime[k];
            }

    	    if(show_each_iteration){
		cout << "Pair[3] Iterations ["<< it <<"] U....";
                double loss_tmp = compute_loss(ratings_matrix, ratings_matrix_comp, users_features, items_features);
		double train_ndcg= compute_ndcg(input.ratings_train, users_features, items_features, ndcg_k);
		double test_ndcg= compute_ndcg(input.ratings_test, users_features, items_features, ndcg_k);
                cout <<"done \t loss="<<loss_tmp<<"\t train_ndcg="<< train_ndcg << "\t test_ndcg="<< test_ndcg<< endl;
    	    }           
*/            
	    //update Vj
            for(unsigned iid=0; iid < ratings_matrix_t.size(); iid++){
                compute_gradient_vj_pair(iid, ratings_matrix, ratings_matrix_t, vj_prime);
                for(int k=0; k < rank; k++)
                    items_features[iid][k] -= learning_rate * vj_prime[k];
            }
    	    if(show_each_iteration){
 		cout << "Pair[3] Iterations ["<< it <<"] V... ";
    	    	double loss_tmp_2 = compute_loss(ratings_matrix, ratings_matrix_comp, users_features, items_features);
		double train_ndcg= compute_ndcg(input.ratings_train, users_features, items_features, ndcg_k);
		double test_ndcg= compute_ndcg(input.ratings_test, users_features, items_features, ndcg_k);
    	    	cout <<"done \t loss="<<loss_tmp_2 <<"\t train_ndcg="<< train_ndcg <<"\t test_ndcg="<< test_ndcg << endl;
    	    }
	                	
	    it++;
	    double loss2 = compute_loss(ratings_matrix, ratings_matrix_comp, users_features, items_features);
	    
	    if(loss2>loss1){
		cout<<"learning_rate too large ..."<<endl;
    	        learning_rate = learning_rate / 2.0;

	    }

        }



        //***********************************************
        //if alpha+beta <=z && z <= 1.,update listwise loss
        //***********************************************
        if(alpha+beta <=z && z <= 1.){ 
    	    if(show_each_iteration){
                //update Ui
                cerr << "Starting listwise update U... ";
    	    }
            for(unsigned uid=0; uid < ratings_matrix.size(); uid++){
                compute_gradient_ui_list(uid, ratings_matrix, ui_prime);
                for(int k=0; k < rank; k++)
                    users_features[uid][k] -= learning_rate * ui_prime[k];
            }
    	    if(show_each_iteration){
    	    //	loss1 = compute_loss(ratings_matrix, ratings_matrix_comp);
            //    	cerr <<"done; loss="<<loss1<<"\nstarting listwise update V... ";
    	    }
                //update Vj
            for(unsigned iid=0; iid < ratings_matrix_t.size(); iid++){
                compute_gradient_vj_list(iid, ratings_matrix, ratings_matrix_t, vj_prime);
                for(int k=0; k < rank; k++)
                    items_features[iid][k] -= learning_rate * vj_prime[k];
            }
    	    if(show_each_iteration){
    	    //	loss1 = compute_loss(ratings_matrix, ratings_matrix_comp);
    	    //	cout <<"done; loss1="<<loss<<endl;
    	    }

	    it++;
            loss1 = compute_loss(ratings_matrix, ratings_matrix_comp, users_features, items_features);
	    ndcg = compute_ndcg(input.ratings_test, users_features, items_features, ndcg_k);
    	    std::cout << "Iterations ["<< it <<"]\t update listwise:\tloss: "<< loss1 << "\t NDCG: "<<ndcg << endl;

        }
    }
    
}

void HybridRanking::train(){
    read_input();
    cout << n_users << " users and "<<n_items << " items"<<endl;
    cout << "START TRAING.........."<<endl;
    update(input.ratings_train, input.ratings_train_comp);
}



#endif





