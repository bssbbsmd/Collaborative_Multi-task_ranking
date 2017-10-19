#ifndef __HYBRIDRANK_HPP__
#define __HYBRIDRANK_HPP__

#include <random>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <vector>

#include "../elements.hpp"
#include "../model.hpp"
#include "../ratings.hpp"
#include "../loss.hpp"
#include "../problem.hpp"
#include "../evaluator.hpp"
#include "solver.hpp"

using namespace std;

class HybridRank : public Solver{
	protected:
		double alpha, beta; // the parameters to set for step_size		
	    double lambda; // the weights for squared penalty 
	    double gamma; // relative learning rate: lr_reg =  lr_pair * gamma;
		int learn_choice;
		double step_size;

   		vector<int> n_comps_by_user_u, n_comps_by_user_i;
   		vector<int> n_comps_by_item_u, n_comps_by_item_i;

   		void sgd_reg_step_itemwise(Model&, const rating&, double, double);  // itempairs
   		void sgd_reg_step_userwise(Model&, const rating&, double, double);  // userpairs
   		void sgd_itemwise_reg_step(Model&, const comparison&, double, double);
   		void sgd_userwise_reg_step(Model&, const comparison&, double, double);
		void sgd_itemwise_step(Model&, const comparison&, double, double);
		void sgd_userwise_step(Model&, const comparison&, double, double);

		/*
   		bool sgd_pair_step_0(Model&, const comparison&, double, double);
		bool sgd_pair_step_1(Model&, const comparison&, double, double); // regular pairwise update
		bool sgd_pair_step_2(Model&, const comparison&, double, double); // update U by 
		bool sgd_pair_step_3(Model&, const comparison&, const comparison&, double, double);
		bool sgd_pair_step_by_choice(Model&, const comparison&, const comparison&, double, double, int);
		*/

	public:
		HybridRank(): Solver(){}
		HybridRank(double alp, double bet, double lam, init_option_t init, int n_th, int m_it=10, int update_choice=1, double stepsize=0.01, double gam=1.):
			 Solver(init, m_it, n_th), alpha(alp), beta(bet), lambda(lam), learn_choice(update_choice), step_size(stepsize), gamma(gam) {}
		void solve(Problem&, Model&, Evaluator* eval);
};

// Problem contains the training data;
// Model contains the model parameters U and V;
// Evaluate contains the evaluations, which contains the testing data;
void HybridRank::solve(Problem& prob, Model& model,  Evaluator* eval){
	n_users = prob.n_users;
	n_items = prob.n_items;
	//n_train_comps = prob.n_train_comps;

	n_comps_by_user_u = prob.n_pairs_by_user_u;
	n_comps_by_user_i = prob.n_pairs_by_user_i;
	n_comps_by_item_u = prob.n_pairs_by_item_u;
	n_comps_by_item_i = prob.n_pairs_by_item_i;

/*
	for(int i=0; i < n_train_comps; ++i){
		++n_comps_by_user[prob.train[i].user_id];
		++n_comps_by_item[prob.train[i].item1_id];
		++n_comps_by_item[prob.train[i].item2_id];
	}
*/
	double time = omp_get_wtime();	
	initialize(prob, model, init_option); 
	time = omp_get_wtime() - time; 
	cout << "Parameter Initialization time cost .... " << time << endl;	

	int n_total_updates = min(prob.n_itemwise_train_comps, prob.n_userwise_train_comps);

	int n_max_updates = n_total_updates/ n_threads;

	gamma = (double)prob.n_train_rating / n_total_updates * gamma;

	std::cout << gamma << std::endl;

	bool flag = false;
	cout << "Iteration " << "Time " << "PairErr " << "NDCG@10"<<endl;
	for(int iter = 0; iter < max_iter; ++iter){
		double time_single_iter = omp_get_wtime();
		#pragma omp parallel
		{
			std::mt19937 gen(n_threads*iter + omp_get_thread_num());   // seed1
			std::uniform_int_distribution<int> 	 rating_randidx(0, prob.n_train_rating-1);
			std::uniform_int_distribution<int> itemwise_randidx(0, prob.n_itemwise_train_comps-1);
			std::uniform_int_distribution<int> userwise_randidx(0, prob.n_userwise_train_comps-1);	

			std::uniform_real_distribution<double> randa(0.0, 1.0);	
			//std::uniform_real_distribution<double> randb(0.0, 1.0);
			//std::uniform_real_distribution<double> randc(0.0, 1.0);

			//double stepsize = step_size
		
			double stepsize = step_size * pow(0.5, iter);

			for(int n_updates = 1; n_updates < n_max_updates; ++ n_updates){	

/*
				//sample a value between 0 and 1
				double a = randa(gen);

				if(0<=a && a< alpha){ //optimizing personalized ranking
					int item_pair_idx = itemwise_randidx(gen);
					sgd_itemwise_step(model, prob.itemwise_train[item_pair_idx], lambda, stepsize);
				}
				else if(alpha <= a && a <= alpha + beta ){ //optimizing user targeting
					int user_pair_idx = userwise_randidx(gen);
					sgd_userwise_step(model, prob.userwise_train[user_pair_idx], lambda, stepsize);
				} else {
					//int item_pair_idx = itemwise_randidx(gen);
				    //sgd_itemwise_reg_step(model, prob.itemwise_train[item_pair_idx], lambda, stepsize);
					int train_rating_idx = rating_randidx(gen);
					sgd_reg_step(model, prob.rating_train[train_rating_idx], lambda, stepsize * gamma);					
				}
				*/

				//sample a value between 0 and 1
				double a = randa(gen);

				if(0<=a && a< alpha){ //optimizing personalized ranking
					//cout << "optimizing personalized ranking" << endl;
					int item_pair_idx = itemwise_randidx(gen);
					sgd_itemwise_step(model, prob.itemwise_train[item_pair_idx], lambda, stepsize);
				}
				else if(alpha <= a && a <= alpha + beta ){ //optimizing user targeting
					//cout << "optimizing user targeting" << endl;
					int user_pair_idx = userwise_randidx(gen);
					sgd_userwise_step(model, prob.userwise_train[user_pair_idx], lambda, stepsize);
				} else {
					//cout << "update regression"<<endl;
					if(prob.n_itemwise_train_comps < prob.n_userwise_train_comps){
						int item_pair_idx = itemwise_randidx(gen);
						comparison comp = prob.itemwise_train[item_pair_idx];

						rating rat1(comp.user_id, comp.item1_id, comp.item1_rating);
						rating rat2(comp.user_id, comp.item2_id, comp.item2_rating);

						sgd_reg_step_itemwise(model, rat1, lambda, stepsize * gamma);
						sgd_reg_step_itemwise(model, rat2, lambda, stepsize * gamma);
					}else{
						int user_pair_idx = userwise_randidx(gen);
						comparison comp = prob.userwise_train[user_pair_idx];
						rating rat1(comp.item1_id, comp.user_id, comp.item1_rating);
						rating rat2(comp.item2_id, comp.user_id, comp.item2_rating);
						sgd_reg_step_userwise(model, rat1, lambda, stepsize * gamma);
						sgd_reg_step_userwise(model, rat2, lambda, stepsize * gamma);
					}					
				}
				


/*
				if(alpha!=0 || beta!=0){
					double thd1,thd2,thd3;
					
					thd1 = alpha /(alpha + beta);

					if(alpha!=1){
						thd3 = beta / (1-alpha);
					}else{
						thd3 = 0.;
					}

					if(beta!=1)	{
						thd2 = alpha /(1-beta);
					}else{
						thd2 = 0.;
					}
					//cout << thd1 << " " << thd2 << " " << thd3 << endl;
					double a = randa(gen);	
					if(a < thd1){	
					//optimizing personalized ranking				
						double b = randb(gen);					
						if(b < thd2){ //update item pairwise loss
							int item_pair_idx = itemwise_randidx(gen);
							sgd_itemwise_step(model, prob.itemwise_train[item_pair_idx], lambda, stepsize);
						}else{ //update mf
						//	int item_pair_idx = itemwise_randidx(gen);
						//	sgd_itemwise_reg_step(model, prob.itemwise_train[item_pair_idx], lambda, stepsize);
							int train_rating_idx = rating_randidx(gen);
							sgd_reg_step(model, prob.rating_train[train_rating_idx], lambda, stepsize);
						}
					} 
					else { 
					//optimizing user targeting					
						double c = randc(gen);
						if(c < thd3){
							int user_pair_idx = userwise_randidx(gen);
							sgd_userwise_step(model, prob.userwise_train[user_pair_idx], lambda, stepsize);
						}else{
						//	int user_pair_idx = userwise_randidx(gen);
						//	sgd_userwise_reg_step(model, prob.userwise_train[user_pair_idx], lambda, stepsize);
							int train_rating_idx = rating_randidx(gen);
							sgd_reg_step(model, prob.rating_train[train_rating_idx], lambda, stepsize);
						}
					}
				}else{


				}	
			*/
			}
		}
		time = time + (omp_get_wtime() - time_single_iter);
		cout << iter+1 << " " << time << " ";
		//double f = prob.evaluate(model);
		//cout << model.n_users<< endl;
		eval->evaluate(model);
		cout << endl;
	}
 
}

void HybridRank::sgd_reg_step_itemwise(Model& model, const rating& rat, double lambda, double stepsize){

	int uid = rat.user_id;
	int iid = rat.item_id;
	double score = rat.score;

	//cout<< uid << ":" << iid << ":" << score << endl;
	double* user_vec  = &(model.U[ uid * model.rank]);
	double* item_vec = &(model.V[ iid * model.rank]);    

	int n_comps_user = n_comps_by_user_u[uid];
	int n_comps_item = n_comps_by_user_i[iid];

	//cout << "break point 1" << endl;

	if(n_comps_user < 1 || n_comps_item < 1) {
		n_comps_item = 1;
		n_comps_user = 1;
	}

	if(uid >= model.n_users || iid >= model.n_items){
		printf("Pointwise, id exceeds the maximum number\n");
		return ;
	}

	double rating_hat = 0.;

	for(int k=0; k < model.rank; k++)
		rating_hat += user_vec[k] * item_vec[k];

	for(int k=0; k < model.rank; k++){
		double user_dir = 2 * stepsize * ( (score - rating_hat) * -item_vec[k] + lambda / (double) n_comps_user * user_vec[k]);
		double item_dir = 2 * stepsize * ( (score - rating_hat) * -user_vec[k] + lambda / (double) n_comps_item * item_vec[k]);
	
		user_vec[k] -= user_dir;
		item_vec[k] -= item_dir;
	}
}

void HybridRank::sgd_reg_step_userwise(Model& model, const rating& rat, double lambda, double stepsize){

	int uid = rat.user_id;
	int iid = rat.item_id;
	double score = rat.score;

	//cout<< uid << ":" << iid << ":" << score << endl;
	double* user_vec  = &(model.U[ uid * model.rank]);
	double* item_vec = &(model.V[ iid * model.rank]);    

	int n_comps_user = n_comps_by_item_u[uid];
	int n_comps_item = n_comps_by_item_i[iid];

	//cout << "break point 1" << endl;

	if(n_comps_user < 1 || n_comps_item < 1) {
		n_comps_item = 1;
		n_comps_user = 1;
	}

	if(uid >= model.n_users || iid >= model.n_items){
		printf("Pointwise, id exceeds the maximum number\n");
		return ;
	}

	double rating_hat = 0.;
	
	for(int k=0; k < model.rank; k++)
		rating_hat += user_vec[k] * item_vec[k];

	for(int k=0; k < model.rank; k++){
		double user_dir = 2 * stepsize * ( (score - rating_hat) * -item_vec[k] + lambda / (double) n_comps_user * user_vec[k]);
		double item_dir = 2 * stepsize * ( (score - rating_hat) * -user_vec[k] + lambda / (double) n_comps_item * item_vec[k]);
	
		user_vec[k] -= user_dir;
		item_vec[k] -= item_dir;
	}
}

void HybridRank::sgd_itemwise_step(Model& model, const comparison& comp, double lambda, double stepsize){
	int uid = comp.user_id;
	int iid1= comp.item1_id;
	int iid2= comp.item2_id;
	double sc1 = comp.item1_rating;
	double sc2 = comp.item2_rating;

	double* user_vec  = &(model.U[ uid * model.rank]);
	double* item1_vec = &(model.V[ iid1 * model.rank]);    
	double* item2_vec = &(model.V[ iid2 * model.rank]);

	int n_comps_user = n_comps_by_user_u[uid];
	int n_comps_item1= n_comps_by_user_i[iid1];
	int n_comps_item2= n_comps_by_user_i[iid2];

	if(n_comps_user < 1 || n_comps_item1 < 1 || n_comps_item2 < 1) {
		cout << "No training pair Error" << endl;
		return ;
	}

	if(iid1 > model.n_items || iid2 > model.n_items) {
		printf(" Items id exceeds the maximum number of items\n");
		return ;
	}

	double prod = 0.;
	
	for( int k=0; k < model.rank; k++) 
		prod += user_vec[k] * (item1_vec[k] - item2_vec[k]);

	if(prod !=  prod){ 
	//	cout << "1.  Numerical Error! prod="<< prod <<endl;
		return;
	}

	double grad = 0.;
	grad = - 1./(1. + exp(prod));

	if(grad !=0.){
		 for(int k=0; k<model.rank; k++) {
		//	double user_dir  = stepsize * (grad * comp.comp * (item1_vec[k] - item2_vec[k]) + 2 * lambda /(double)n_comps_user * user_vec[k]);
			double item1_dir = stepsize * (grad * comp.comp * user_vec[k] + 2 * lambda / (double) n_comps_item1 * item1_vec[k]);
			double item2_dir = stepsize * (grad * -comp.comp * user_vec[k] + 2 * lambda  / (double)n_comps_item2 * item2_vec[k]);

		//	user_vec[k]  -= user_dir;
			item1_vec[k] -= item1_dir;
			item2_vec[k] -= item2_dir;
		}
	}
}

void HybridRank::sgd_userwise_step(Model& model, const comparison& comp, double lambda, double stepsize){
	int iid = comp.user_id;
	int uid1= comp.item1_id;
	int uid2= comp.item2_id;
	double sc1 = comp.item1_rating;
	double sc2 = comp.item2_rating;

	double* user1_vec = &(model.U[ uid1 * model.rank]);
	double* user2_vec = &(model.U[ uid2 * model.rank]);
	double* item_vec  = &(model.V[ iid * model.rank]);

	int n_comps_user1 = n_comps_by_item_u[uid1];
	int n_comps_user2 = n_comps_by_item_u[uid2];
	int n_comps_item  = n_comps_by_item_i[iid];

	if(n_comps_user1 < 1 || n_comps_user2 < 1 || n_comps_item < 1) {
		cout << "No training pair Error" << endl;
		return ;
	}

	if(uid1 > model.n_users || uid2 > model.n_users) {
		printf(" Items id exceeds the maximum number of items\n");
		return ;
	}

	double prod = 0.;
	for(int k=0; k<model.rank; k++)	
		prod += (user1_vec[k]-user2_vec[k])*item_vec[k]; 

	if(prod !=  prod){ 
	//	cout << "2.  Numerical Error! prod="<< prod <<endl;
		return;
	}
	
	double grad = 0.;
	grad = -1./(1. + exp(prod));  // - 1/ (1+e^(r_{u1i}-r_{u2i}));	

	if(grad != 0.){
		for(int k=0; k < model.rank; k++){
		//	double item_dir = stepsize * (grad * comp.comp * (user1_vec[k] - user2_vec[k]) + 2 * lambda / (double)n_comps_item * item_vec[k]);
			double user1_dir= stepsize * (grad * comp.comp * item_vec[k] + 2 * lambda / (double)n_comps_user1 * user1_vec[k]);
			double user2_dir= stepsize * (grad * -comp.comp* item_vec[k] + 2 * lambda / (double)n_comps_user2 * user2_vec[k]);

		//	item_vec[k] -= item_dir;
			user1_vec[k] -= user1_dir;
			user2_vec[k] -= user2_dir;
		}
	}
}

/*
void HybridRank::sgd_itemwise_reg_step(Model& model, const comparison& comp, double lambda, double stepsize){
	int uid = comp.user_id;
	int iid1= comp.item1_id;
	int iid2= comp.item2_id;
	double sc1 = comp.item1_rating;
	double sc2 = comp.item2_rating;

	double* user_vec  = &(model.U[ uid * model.rank]);
	double* item1_vec = &(model.V[ iid1 * model.rank]);    
	double* item2_vec = &(model.V[ iid2 * model.rank]);

	int n_comps_user = n_comps_by_user_u[uid];
	int n_comps_item1= n_comps_by_user_i[iid1];
	int n_comps_item2= n_comps_by_user_i[iid2];

	if(n_comps_user < 1 || n_comps_item1 < 1 || n_comps_item2 < 1) {
		cout << "Error: no training pair for user "<< uid << endl;
		return ;
	}

	if(iid1 > model.n_items || iid2 > model.n_items) {
		printf(" Items id exceeds the maximum number of items\n");
		return ;
	}

	double item1_rating_hat = 0.;
	for(int k=0; k < model.rank; k++)
		item1_rating_hat += user_vec[k] * item1_vec[k];

	double item2_rating_hat = 0.;
	for(int k=0; k < model.rank; k++)
		item2_rating_hat += user_vec[k] * item2_vec[k];

	for(int k=0; k < model.rank; k++){
		double user_dir = 2 * stepsize * ( (sc1 - item1_rating_hat) * comp.comp * (-item1_vec[k]) + 
			(sc2 - item2_rating_hat) * comp.comp * (-item2_vec[k]) + lambda / (double)n_comps_user * user_vec[k]);
		double item1_dir = 2 * stepsize * ( (sc1 - item1_rating_hat) * comp.comp * -user_vec[k] + lambda / (double)n_comps_item1 * item1_vec[k]);
		double item2_dir = 2 * stepsize * ( (sc2 - item2_rating_hat) * comp.comp * -user_vec[k] + lambda / (double)n_comps_item2 * item2_vec[k]);

		user_vec[k] -= user_dir;
		item1_vec[k]-= item1_dir;
		item2_vec[k]-= item2_dir;
	}

}




void HybridRank::sgd_userwise_reg_step(Model& model, const comparison& comp, double lambda, double stepsize){
	int iid = comp.user_id;
	int uid1= comp.item1_id;
	int uid2= comp.item2_id;
	double sc1 = comp.item1_rating;
	double sc2 = comp.item2_rating;

	double* user1_vec = &(model.U[ uid1 * model.rank]);
	double* user2_vec = &(model.U[ uid2 * model.rank]);
	double* item_vec  = &(model.V[ iid * model.rank]);

	int n_comps_user1 = n_comps_by_item_u[uid1];
	int n_comps_user2 = n_comps_by_item_u[uid2];
	int n_comps_item  = n_comps_by_item_i[iid];

	if(n_comps_user1 < 1 || n_comps_user2 < 1 || n_comps_item < 1) {
		cout << "Error: no training pair for item "<< iid << endl;
		return ;
	}

	if(uid1 > model.n_users || uid2 > model.n_users) {
		printf(" Items id exceeds the maximum number of items\n");
		return ;
	}

	double user1_rating_hat = 0.;
	for(int k=0; k < model.rank; k++)
		user1_rating_hat += user1_vec[k] * item_vec[k];

	double user2_rating_hat = 0.;
	for(int k=0; k < model.rank; k++)
		user2_rating_hat += user2_vec[k] * item_vec[k];

	for(int k=0; k < model.rank; k++){
		double item_dir = 2 * stepsize * ( (sc1 - user1_rating_hat) * comp.comp * (-user1_vec[k]) + 
			(sc2 - user2_rating_hat) * comp.comp * (-user2_vec[k]) + lambda / (double)n_comps_item * item_vec[k]);
		double user1_dir = 2 * stepsize * ( (sc1 - user1_rating_hat) * comp.comp * -item_vec[k] + lambda / (double)n_comps_user1 * user1_vec[k]);
		double user2_dir = 2 * stepsize * ( (sc2 - user2_rating_hat) * comp.comp * -item_vec[k] + lambda / (double)n_comps_user2 * user2_vec[k]);

		item_vec[k] -= item_dir;
		user1_vec[k]-= user1_dir;
		user2_vec[k]-= user2_dir;
	}
}

*/

/*

bool HybridRank::sgd_pair_step_by_choice(Model& model, const comparison& comp, const comparison& comp_user, double lambda, double step_size, int choice){
	switch(choice){
		case 0: return sgd_pair_step_0(model, comp, lambda, step_size); break;
		case 1: return sgd_pair_step_1(model, comp, lambda, step_size); break;
		case 2: return sgd_pair_step_2(model, comp, lambda, step_size); break;
		case 3: return sgd_pair_step_3(model, comp, comp_user, lambda, step_size); break;
		default:
			cout << "Input Learning Choice Error!" << endl;
			return false;
	}
	return false;
}

bool HybridRank::sgd_pair_step_0(Model& model, const comparison& comp, double lambda, double step_size){
	double* user_vec = &(model.U[comp.user_id * model.rank]);
	double* item1_vec = &(model.V[comp.item1_id * model.rank]);	
	double* item2_vec = &(model.V[comp.item2_id * model.rank]);

	int n_comps_user = n_comps_by_user[comp.user_id];
	int n_comps_item1= n_comps_by_item[comp.item1_id];
	int n_comps_item2= n_comps_by_item[comp.item2_id];

	if(n_comps_user < 1 || n_comps_item1 < 1 || n_comps_item2 < 1) cout << "Error" << endl;
	if(comp.item1_id > model.n_items || comp.item2_id > model.n_items) printf(" Items id exceeds the maximum number of items\n");

	double prod = 0.;
	for( int k=0; k < model.rank; k++) prod += user_vec[k] * (item1_vec[k] - item2_vec[k]);
	if(prod !=  prod){ 
		//cout << "Numerical Error!" <<endl;
		return false;
	}

	double grad = 0.;
	grad = - 1./(1. + exp(prod));

	if(grad !=0.){
		 for(int k=0; k<model.rank; k++) {
		//	double user_dir  = step_size * (grad * comp.comp * (item1_vec[k] - item2_vec[k]) + 2 * lambda /(double)n_comps_user * user_vec[k]);
			double item1_dir = step_size * (grad * comp.comp * user_vec[k] + 2 * lambda / (double) n_comps_item1 * item1_vec[k]);
			double item2_dir = step_size * (grad * -comp.comp * user_vec[k] + 2 * lambda  / (double)n_comps_item2 * item2_vec[k]);

		//	user_vec[k]  -= user_dir;
			item1_vec[k] -= item1_dir;
			item2_vec[k] -= item2_dir;
		}
	}
	return true;	
}

bool HybridRank::sgd_pair_step_1(Model& model, const comparison& comp, double lambda, double step_size){
	double* user_vec = &(model.U[comp.user_id * model.rank]);
	double* item1_vec = &(model.V[comp.item1_id * model.rank]);	
	double* item2_vec = &(model.V[comp.item2_id * model.rank]);

	int n_comps_user = n_comps_by_user[comp.user_id];
	int n_comps_item1= n_comps_by_item[comp.item1_id];
	int n_comps_item2= n_comps_by_item[comp.item2_id];

	if(n_comps_user < 1 || n_comps_item1 < 1 || n_comps_item2 < 1) cout << "Error" << endl;
	if(comp.item1_id > model.n_items || comp.item2_id > model.n_items) printf(" Items id exceeds the maximum number of items\n");

	double prod = 0.;
	for( int k=0; k < model.rank; k++) prod += user_vec[k] * (item1_vec[k] - item2_vec[k]);
	if(prod !=  prod){ 
		//cout << "Numerical Error!" <<endl;
		return false;
	}

	double grad = 0.;
	grad = - 1./(1. + exp(prod));

	if(grad !=0.){
		 for(int k=0; k<model.rank; k++) {
			double user_dir  = step_size * (grad * comp.comp * (item1_vec[k] - item2_vec[k]) + 2 * lambda /(double)n_comps_user * user_vec[k]);
			double item1_dir = step_size * (grad * comp.comp * user_vec[k] + 2 * lambda / (double) n_comps_item1 * item1_vec[k]);
			double item2_dir = step_size * (grad * -comp.comp * user_vec[k] + 2 * lambda  / (double)n_comps_item2 * item2_vec[k]);

			user_vec[k]  -= user_dir;
			item1_vec[k] -= item1_dir;
			item2_vec[k] -= item2_dir;
		}
	}

	return true;	

}*/

/**
* Update V using pairwise and update U using squared loss
*
*/
/*
bool HybridRank::sgd_pair_step_2(Model& model, const comparison& comp, double lambda, double step_size){
	double* user_vec = &(model.U[comp.user_id * model.rank]);
	double* item1_vec = &(model.V[comp.item1_id * model.rank]);	
	double* item2_vec = &(model.V[comp.item2_id * model.rank]);

	int n_comps_user = n_comps_by_user[comp.user_id];
	int n_comps_item1= n_comps_by_item[comp.item1_id];
	int n_comps_item2= n_comps_by_item[comp.item2_id];

	if(n_comps_user < 1 || n_comps_item1 < 1 || n_comps_item2 < 1) cout << "Error" << endl;
	if(comp.item1_id > model.n_items || comp.item2_id > model.n_items) printf(" Items id exceeds the maximum number of items\n");

	double prod = 0.;
	for( int k=0; k < model.rank; k++) prod += user_vec[k] * (item1_vec[k] - item2_vec[k]);
	if(prod !=  prod){ 
		//cout << "Numerical Error!" <<endl;
		return false;
	}

	double grad = 0.;
	grad = - 1./(1. + exp(prod));

	double item1_rating_hat = 0.; 
	for(int k=0; k < model.rank; k++) item1_rating_hat += user_vec[k] * item1_vec[k];

	double item2_rating_hat = 0.;
	for(int k=0; k < model.rank; k++) item2_rating_hat += user_vec[k] * item2_vec[k];

	if(grad !=0.){
		 for(int k=0; k<model.rank; k++) {
		//	double user_dir  = step_size * (grad * comp.comp * (item1_vec[k] - item2_vec[k]) + 2 * lambda /(double)n_comps_user * user_vec[k]);
			
			//Update U using regression loss
			double user_dir  = 2 * step_size * beta * ( (comp.item1_rating - item1_rating_hat) * comp.comp * (-item1_vec[k]) + 
					(comp.item2_rating - item2_rating_hat) * comp.comp * (-item2_vec[k]) + lambda /(double)n_comps_user * user_vec[k]);
            
			//update V using pairwise loss
			double item1_dir = step_size * (grad * comp.comp * user_vec[k] + 2 * lambda / (double) n_comps_item1 * item1_vec[k]);
			double item2_dir = step_size * (grad * -comp.comp * user_vec[k] + 2 * lambda  / (double)n_comps_item2 * item2_vec[k]);

			user_vec[k]  -= user_dir;
			item1_vec[k] -= item1_dir;
			item2_vec[k] -= item2_dir;
		}
	}

	return true;
}





bool HybridRank::sgd_pair_step_3(Model& model, const comparison& comp_item, const comparison& comp_user, double lambda, double step_size)
{
	

	return true;
}
*/

#endif





