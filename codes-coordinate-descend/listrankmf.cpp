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

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <utility>
#include <vector>
#include <iostream>

using std::default_random_engine;
using std::exp;
using std::fill;
using std::log;
using std::make_pair;
using std::max;
using std::min;
using std::numeric_limits;
using std::pair;
using std::pow;
using std::random_device;
using std::uniform_real_distribution;
using std::vector;

#include "./listrankmf.h"
#include "./evaluate.h"

// Sigmoid (logistic) function
double ListRankMF::sigmoid(double x) {
   // return 1 / (1 + exp(-x));
    return x;
}

// Derivative of the sigmoid function
double ListRankMF::sigmoid_prime(double x) {
   // double s = sigmoid(x);
   // return s * (1 - s);
    return 1;
}

// Computes the values of the loss function ListRank-MF optimizes,
// that is, the cross-entropy between the top-1 probability distributions
// estimated by the observed ratings and by the predicted scores
double ListRankMF::compute_loss(
            const vector<vector<double>> &users_features,
            const vector<vector<double>> &items_features,
            const vector<vector<pair<int, double>>> &ratings_matrix,
            double lambda) {
        double loss = 0;
        const int LATENT_FEATURES = users_features.empty() ?
            0 : users_features[0].size();
        
        for (unsigned i = 0; i < users_features.size(); i++) {
            double den_predicted = 0;
            double den_ratings = 0;
        
            for (unsigned j = 0; j < ratings_matrix[i].size(); j++) {
                den_predicted += exp(sigmoid(predict_score(users_features[i],
                            items_features[ratings_matrix[i][j].first-1])));
                den_ratings += exp(ratings_matrix[i][j].second);
            }

            for (unsigned j = 0; j < ratings_matrix[i].size(); j++) {
                loss -= exp(ratings_matrix[i][j].second) / den_ratings *
                    log(exp(sigmoid(predict_score(users_features[i], items_features[ratings_matrix[i][j].first-1]))) / den_predicted);
            }
        }
        
        for (unsigned i = 0; i < users_features.size(); i++)
            for (int j = 0; j < LATENT_FEATURES; j++)
                loss += pow(users_features[i][j], 2) * lambda / 2;
        
        for (unsigned i = 0; i < items_features.size(); i++)
            for (int j = 0; j < LATENT_FEATURES; j++)
                loss += pow(items_features[i][j], 2) * lambda / 2;

        return loss;
}

void ListRankMF::compute_gradient_ui(
            const vector<vector<double>> &users_features,
            const vector<vector<double>> &items_features,
            vector<vector<double>> &users_features_prime,
            int user_id,
            const vector<vector<pair<int, double>>> &ratings_matrix,
            double lambda) {
        fill(users_features_prime[user_id].begin(),
                users_features_prime[user_id].end(),
                0.0);

        const int RATED_MOVIES = ratings_matrix[user_id].size();
        const int LATENT_FEATURES = users_features.empty() ?
            0 : users_features[0].size();

        double predicted_denominator = 0;
        double ratings_denominator = 0;

        for (int j = 0; j < RATED_MOVIES; j++) {
            predicted_denominator += exp(sigmoid(predict_score(
                            users_features[user_id],
                            items_features[ratings_matrix[user_id][j].first-1])));
            ratings_denominator += exp(ratings_matrix[user_id][j].second);
        }

        for (int j = 0; j < RATED_MOVIES; j++) {
            double multiplier = sigmoid_prime(predict_score(
                        users_features[user_id],
                        items_features[ratings_matrix[user_id][j].first-1])); //this line is corrected;
            multiplier *= (exp(sigmoid(predict_score(
                                users_features[user_id],
                                items_features[ratings_matrix[user_id][j].first-1])))
                    / predicted_denominator) -
                (exp(ratings_matrix[user_id][j].second) / ratings_denominator);

            for (int k = 0; k < LATENT_FEATURES; k++) {
                users_features_prime[user_id][k] +=
                    multiplier * items_features[ratings_matrix[user_id][j].first-1][k]; //this line is corrected;
            }
        }

        for (int k = 0; k < LATENT_FEATURES; k++)
            users_features_prime[user_id][k] +=
                lambda * users_features[user_id][k];
}

void ListRankMF::compute_gradient_vj(
            const vector<vector<double>> &users_features,
            const vector<vector<double>> &items_features,
            vector<vector<double>> &items_features_prime,
            int item_id,  // suppose item_id = 0;
            const vector<vector<pair<int, double>>> &ratings_matrix,
            const vector<vector<pair<int, double>>> &ratings_matrix_t,
            double lambda) {
        fill(items_features_prime[item_id].begin(),
                items_features_prime[item_id].end(),
                0.0);

        const int LATENT_FEATURES = users_features.empty() ?
            0 : users_features[0].size();

        int number_of_users = ratings_matrix_t[item_id].size();

        for (int u = 0; u < number_of_users; u++) {
            int user_id = ratings_matrix_t[item_id][u].first;

            int rated_items = ratings_matrix[user_id].size();
            double predicted_denominator = 0;
            double ratings_denominator = 0;

            for (int j = 0; j < rated_items; j++) {
                predicted_denominator += exp(sigmoid(predict_score(
                                users_features[user_id],
                                items_features[ratings_matrix[user_id][j].first-1])));
                ratings_denominator += exp(ratings_matrix[user_id][j].second);
            }

            double multiplier = sigmoid_prime(predict_score(
                        users_features[user_id], items_features[item_id]));
            multiplier *= (exp(sigmoid(predict_score(
                                users_features[user_id],
                                items_features[item_id]))) /
                    predicted_denominator) -
                (exp(ratings_matrix_t[item_id][u].second) /
                 ratings_denominator);

            for (int k = 0; k < LATENT_FEATURES; k++) {
                items_features_prime[item_id][k] +=
                    multiplier * users_features[user_id][k];
            }
        }

        for (int k = 0; k < LATENT_FEATURES; k++)
            items_features_prime[item_id][k] +=
                lambda * items_features[item_id][k];
}

void ListRankMF::randomly_initialize(vector<vector<double>> &features) {
        random_device rd;
        default_random_engine re(rd());

        // ``Good range'' defined by experiments
        uniform_real_distribution<> dis(0, 1);

        for (auto &v : features) {
            for (auto &f : v) {
                f = dis(re);
            }
        }
}

void ListRankMF::list_rank_mf(
        const vector<vector<pair<int, double>>> &ratings_matrix,
	const vector<vector<pair<int, double>>> &test_matrix,
        vector<vector<double>> &users_features,
        vector<vector<double>> &items_features,
        unsigned int d,
        double learning_rate,
        double lambda,
        double eps,
        unsigned int max_iterations,
        bool initialize) {
    int number_of_users = n_users;
    int number_of_items = n_items;

    // Transpose of the ratings matrix
    vector<vector<pair<int, double>>> ratings_matrix_t; //index from 0->n_items-1

    for (int user_id = 0; user_id < number_of_users; user_id++) {
        for (const auto &rating : ratings_matrix[user_id]) {
            //number_of_items = max(number_of_items, rating.first + 1);

            if (ratings_matrix_t.size() <=
                    static_cast<unsigned int>(rating.first)) {
                ratings_matrix_t.resize(rating.first);
            }

            ratings_matrix_t[rating.first-1].push_back(
                    make_pair(user_id, rating.second));
        }
    }
    
 //   std::cout << "ratings_matrix_transpose rows' number1: "<<ratings_matrix_t.size()<< std::endl;   
    ratings_matrix_t.shrink_to_fit();
 //   std::cout << "ratings_matrix_transpose rows' number2: "<<ratings_matrix_t.size()<< std::endl;   
 //   std::cout << "last row size of transpose matrix: "<< ratings_matrix_t[ratings_matrix_t.size()-1].size()<< std::endl;

    users_features.resize(number_of_users); //index: 0->n_users-1;
    items_features.resize(number_of_items); //index: 0->n_items-1 

    for (auto &f : users_features)
        f.resize(d);

    for (auto &f : items_features)
        f.resize(d);

    if (initialize) {
        randomly_initialize(users_features);
        randomly_initialize(items_features);
    }

    vector<vector<double>> users_features_prime(number_of_users, vector<double>(d)); //index: 0->n_users-1;
    vector<vector<double>> items_features_prime(number_of_items, vector<double>(d)); //index: 0->n_items-1; 
    
    double last_loss = numeric_limits<double>::infinity();

    // Stochastic Gradient Descent iterations
    for (unsigned it = 0; max_iterations == 0 || it < max_iterations; it++) {
        double loss = compute_loss(users_features, items_features,  ratings_matrix, lambda);
	double ndcg = compute_ndcg(test_matrix, users_features, items_features, ndcg_k);
	std::cout << "Iterations "<< it << "\tloss: "<< loss << "\t NDCG: "<<ndcg << std::endl;
        // std::cout << "Iterations "<< it << "\tloss: "<< loss << std::endl;
        if (loss > last_loss - eps)
            break;
        
        last_loss = loss;

        // Optimize the users' factors
        for (unsigned int i = 0; i < ratings_matrix.size(); i++) {
            compute_gradient_ui(users_features, items_features,
                    users_features_prime, i, ratings_matrix, lambda);
            for (unsigned k = 0; k < d; k++) {
                users_features[i][k] -=
                    learning_rate * users_features_prime[i][k];
            }
        }
        
     //	std::cout << "Successfully calculate derivative of U_i, loss=";
     //	loss = compute_loss(users_features, items_features, ratings_matrix, lambda);
      //  for (int i = 0; i < number_of_items; i++) {
	for(unsigned int j = 0; j < ratings_matrix_t.size(); j++){
            compute_gradient_vj(users_features, items_features, items_features_prime, j, ratings_matrix, ratings_matrix_t, lambda);

            for (unsigned k = 0; k < d; k++) {
                items_features[j][k] -=
                    learning_rate * items_features_prime[j][k];
            }
        }
     //	std::cout << "Successfully calculate derivative of V_j" << std::endl;
    }
}

double ListRankMF::predict_score(
        const vector<double> &user_features,
        const vector<double> &item_features) {
    double score = 0;

    unsigned size = min(user_features.size(), item_features.size());

    for (unsigned i = 0; i < size; i++)
        score += user_features[i] * item_features[i];

    return score;
}

