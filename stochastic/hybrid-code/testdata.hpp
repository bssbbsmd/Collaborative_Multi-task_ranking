#ifndef __TESTDATA_H__
#define __TESTDATA_H__

#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

#include "elements.hpp"
#include "model.hpp"

using namespace std;

class TestData {
public:
	int	n_users, n_items;

	std::vector<rating>	itemwise_ratings;  // item pairs
	std::vector<int> itemwise_idx;  

	std::vector<rating>	userwise_ratings;  // user pairs
	std::vector<int> userwise_idx;

	int ndcg_k=0;
	bool is_dcg_max_computed = false;
	std::vector<double>   dcg_max;

	void   compute_dcgmax(int);
	double compute_user_ndcg(int, const std::vector<double>&) const;

	TestData(): n_users(0), n_items(0) {}
	TestData(int nu, int ni): n_users(nu), n_items(ni) {}

	void read_itemwise_test(const string&);  // read test data for evaluating personalized ranking
	void read_userwise_test(const string&);  // read test data for evaluating user targeting
};

std::ifstream::pos_type filesize(const char* filename)
{
	std::ifstream in(filename, std::ios::binary | std::ios::ate);
	return in.tellg(); 
}

void TestData::read_itemwise_test(const string& filename) {
	itemwise_idx.clear();
	itemwise_ratings.clear();

	n_users = 0;
	n_item = 0;
}







#endif