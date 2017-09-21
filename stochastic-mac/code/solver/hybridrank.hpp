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
	     	double gamma; // the weights for squared penalty 
		int learn_choice;

   		vector<int> n_comps_by_user, n_comps_by_item;
	//	bool sgd_step(Model&, const comparison&, double, double, double);
   		bool sgd_pair_step_0(Model&, const comparison&, double, double);
		bool sgd_pair_step_1(Model&, const comparison&, double, double); // regular pairwise update
		bool sgd_pair_step_2(Model&, const comparison&, double, double); // update U by 
		bool sgd_pair_step_3(Model&, const comparison&, const comparison&, double, double);
		bool sgd_pair_step_by_choice(Model&, const comparison&, const comparison&, double, double, int);
		
	public:
		HybridRank(): Solver(){}
		HybridRank(double alp, double bet, double gam, init_option_t init, int n_th, int m_it=10, int update_choice=1):
			 Solver(init, m_it, n_th), alpha(alp), beta(bet), gamma(gam), learn_choice(update_choice) {}
		void solve(Problem&, Model&, Evaluator* eval);
};

void HybridRank::solve(Problem& prob, Model& model,  Evaluator* eval){
	n_users = prob.n_users;
	n_items = prob.n_items;
	n_train_comps = prob.n_train_comps;

	n_comps_by_user.resize(n_users, 0);
	n_comps_by_item.resize(n_items, 0);

	for(int i=0; i < n_train_comps; ++i){
		++n_comps_by_user[prob.train[i].user_id];
		++n_comps_by_item[prob.train[i].item1_id];
		++n_comps_by_item[prob.train[i].item2_id];
	}

	double time = omp_get_wtime();	
	initialize(prob, model, init_option);
	time = omp_get_wtime() - time;
	cout << "Parameter Initialization time cost .... " << time << endl;
	
	int n_max_updates = n_train_comps / n_threads;

	bool flag = false;
	cout << "Iteration " << "Time " << "PairErr " << "NDCG@10"<<endl;
	for(int iter = 0; iter < max_iter; ++iter){
		double time_single_iter = omp_get_wtime();
		#pragma omp parallel
		{
			std::mt19937 gen(n_threads*iter + omp_get_thread_num());
			std::uniform_int_distribution<int> randidx(0, n_train_comps - 1);

			for(int n_updates = 1; n_updates < n_max_updates; ++ n_updates){
				int idx = randidx(gen);	
				//double stepsize = alpha / (1. + beta*(double)((n_updates + n_max_updates*iter) * n_threads));
				double stepsize = alpha / pow(2.0, iter);
				//if(!sgd_step(model, prob.train[idx], prob.lambda, gamma, stepsize)){
				//	flag = true;  
				//	break;	
				//}	
				if(learn_choice != 3){
					if(!sgd_pair_step_by_choice(model, prob.train[idx], prob.train[idx], prob.lambda, stepsize, learn_choice)){
						flag = true;  
						break;	
					}
				}else{  // update U using column pairwise and update V using V using row pairwise
					sgd_pair_step_by_choice_4()	;
				}					
			}
		}
		
		if(flag) break;
		time = time + (omp_get_wtime() - time_single_iter);
		cout << iter+1 << " " << time << " ";
		double f = prob.evaluate(model);
		eval->evaluate(model);
		cout << endl;
	}
 
}


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

}

/**
* Update V using pairwise and update U using squared loss
*
*/
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



/*
bool HybridRank::sgd_step(Model& model, const comparison& comp, double lambda, double gam, double step_size){
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
		cout << "Numerical Error!" <<endl;
		return false;
	}

	double grad = 0.;
	grad = - 1./(1. + exp(prod)); // the derivative of logistic function


	double item1_rating_hat = 0.; 
	for(int k=0; k < model.rank; k++) item1_rating_hat += user_vec[k] * item1_vec[k];  // the estimated rating of item1

	double item2_rating_hat = 0.;
	for(int k=0; k < model.rank; k++) item2_rating_hat += user_vec[k] * item2_vec[k];  // the estimated rating of item2

	if(grad !=0.){
		 for(int k=0; k<model.rank; k++) {
			double user_dir_pair  = grad * comp.comp * (item1_vec[k] - item2_vec[k]) ;
			double item1_dir_pair = grad * comp.comp * user_vec[k];
			double item2_dir_pair = grad * -comp.comp * user_vec[k];

			double user_dir_squared  = 2 * ( gam *(comp.item1_rating - item1_rating_hat) * comp.comp * (-item1_vec[k]) + 
					gam * (comp.item2_rating - item2_rating_hat) * comp.comp * (-item2_vec[k]) + lambda / (double)n_comps_user * user_vec[k]);
            double item1_dir_squared = 2 * ( gam *(comp.item1_rating - item1_rating_hat) * comp.comp * -user_vec[k] + lambda  / (double)n_comps_item1 * item1_vec[k]);
            double item2_dir_squared = 2 * ( gam *(comp.item2_rating - item2_rating_hat) * comp.comp * -user_vec[k] + lambda  / (double)n_comps_item2 * item2_vec[k]);

			double user_dir = step_size * (user_dir_pair +  user_dir_squared);
			double item1_dir = step_size * (item1_dir_pair + item1_dir_squared);
			double item2_dir = step_size * (item2_dir_pair + item2_dir_squared);

			user_vec[k]  -= user_dir;
			item1_vec[k] -= item1_dir;
			item2_vec[k] -= item2_dir;
		}
	}

	return true;	
}
*/


#endif





