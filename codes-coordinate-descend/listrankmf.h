/*
   The MIT License (MIT)

   Copyright (c) 2014 Gabriel Poesia <gabriel.poesia at gmail.com>

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#ifndef LISTRANKMF_H_
#define LISTRANKMF_H_

#include <vector>
#include <utility>

class ListRankMF{
	public:
	int n_users,n_items;
	int ndcg_k;	
	
	ListRankMF(): n_users(0), n_items(0), ndcg_k(10){}
	ListRankMF(int n_user, int n_item, int k): n_users(n_user), n_items(n_item), ndcg_k(k){};

	double sigmoid(double x);
	double sigmoid_prime(double x);

	double compute_loss(
            const vector<vector<double>> &users_features,
            const vector<vector<double>> &items_features,
            const vector<vector<pair<int, double>>> &ratings_matrix,
            double lambda);
 
	void compute_gradient_ui(
            const vector<vector<double>> &users_features,
            const vector<vector<double>> &items_features,
            vector<vector<double>> &users_features_prime,
            int user_id,
            const vector<vector<pair<int, double>>> &ratings_matrix,
            double lambda); 

	void compute_gradient_vj(
            const vector<vector<double>> &users_features,
            const vector<vector<double>> &items_features,
            vector<vector<double>> &items_features_prime,
            int item_id,
            const vector<vector<pair<int, double>>> &ratings_matrix,
            const vector<vector<pair<int, double>>> &ratings_matrix_t,
            double lambda);

	void randomly_initialize(vector<vector<double>> &features);

	/// Find feature vectors for users and items using ListRank-MF
	/// \param ratings A sparse matrix of known ratings. For each user, it
	/// is expected to contain a vector of pairs (i, r) where i is an integer
	/// identifier of an item, and r is the rating the user assigned to that item.
	/// \param users_features An output vector containing, for each user, his/her
	/// extracted latent feature vector.
	/// \param items_features An output vector containing, for each item, its
	/// extracted latent feature vector.
	/// \param d Number of latent features to extract for each user and item.
	/// \param learning_rate Learning rate coefficient to use in Gradient Descent
	/// \param lambda Coefficient used for regularization of the output.
	/// \param eps Value to establish the stop criteria in the optimization. The
	/// optimization continues until the improvement observed in the loss function
	/// is less than this value.
	/// \param max_iterations Maximum number of iterations of Gradient Descent.
	/// A value of 0 means there is no limit set.
	/// \param initialize Whether this function should (randomly) initialize
	/// the latent factors (true) or use the given initial values instead (false).
	void list_rank_mf(
		const std::vector<std::vector<std::pair<int, double> > > &ratings,
		const std::vector<std::vector<std::pair<int, double> > > &test_matrix,
		std::vector<std::vector<double> > &users_features,
		std::vector<std::vector<double> > &items_features,
		unsigned int d = 10,
		double learning_rate = 0.1,
		double lambda = 0.1,
		double eps = 0.01,
		unsigned int max_iterations = 0,
		bool initialize = true);

	/// Returns the predicted score of an item for a given user
	/// \param user_features Latent features found by ListRank-MF for the user
	/// \param item_features Latent features found by ListRank-MF for the item
	/// user_features and item_features should have the same size. If one has
	/// more elements than the other, the smaller vector is assumed to have zeros
	/// in the remaining positions.
	/// \return The calculated score. This score can be used for ranking the items
	/// for a given user, but otherwise has no intrinsic meaning.
	double predict_score(
		const std::vector<double> &user_features,
		const std::vector<double> &item_features);

};

#endif  // LISTRANKMF_H_
