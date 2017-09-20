#ifndef __INPUTDATA_H__
#define __INPUTDATA_H__

#include <vector>
#include <map>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <math.h>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>


class InputData{
    public:
    int n_users, n_items;
    std::vector<std::vector<std::pair<int, double>>> ratings_train; // iid start from 1;

    std::vector<std::vector<std::pair<int, int>>>    ratings_train_comp;
    std::vector<std::vector<std::pair<int, double>>> ratings_test;
    
    InputData(): n_users(0), n_items(0){};
    void readInputTrain(std::string& filename, std::string& comp_filename);
    void readInputTest(std::string& filename);
};


void InputData::readInputTrain(std::string& filename, std::string& comp_filename){
 //   std::vector<std::vector<std::pair<int, double>>> ratings;
    std::string user_str, user_str_comp, attribute_str, attribute_str_comp;
    std::stringstream attribute_sstr, attribute_sstr_comp;

    n_items = 0;    

    std::ifstream f;
    f.open(filename);
    std::ifstream f_comp;
    f_comp.open(comp_filename);

    if(f.is_open() && f_comp.is_open()){
    	int  iid;	
    	double sc;
    
        int iid_comp1;
        int iid_comp2;

        while(getline(f, user_str) && getline(f_comp, user_str_comp)){
    	    ratings_train.resize(ratings_train.size() + 1);
    	    ratings_train_comp.resize(ratings_train_comp.size() + 1);

            //read data from lsvm training file;
    	    size_t pos1 = 0, pos2;
    	    while(1){
        		pos2 = user_str.find(':', pos1);
        		if(pos2 == std::string::npos) break;
        		attribute_str = user_str.substr(pos1, pos2-pos1);
        		attribute_sstr.clear();
        		attribute_sstr.str(attribute_str);
        		attribute_sstr >> iid;
                        
        		n_items = std::max(n_items, iid);
        		//--iid; // let iid start from 0;
        		pos1 = pos2+1;
        		pos2 = user_str.find(' ', pos1);
        		attribute_str = user_str.substr(pos1, pos2-pos1);
        		attribute_sstr.clear();
        		attribute_sstr.str(attribute_str);
        		attribute_sstr >> sc;
        		pos1 = pos2+1;		
        		
        		ratings_train.back().push_back(make_pair(iid, sc));		                 
    	    }

	        //read data from order ralation input file		
	        pos1 = 0;
    	    while(1){
        		pos2 = user_str_comp.find(':', pos1);
        		if(pos2 == std::string::npos) break;
        		attribute_str_comp = user_str_comp.substr(pos1, pos2-pos1);
        		attribute_sstr_comp.clear();
        		attribute_sstr_comp.str(attribute_str_comp);
        		attribute_sstr_comp >> iid_comp1;
                        
        		//n_items = std::max(n_items, iid);
        		//--iid; // let iid start from 0;
        		pos1 = pos2+1;
        		pos2 = user_str_comp.find(' ', pos1);
        		attribute_str_comp = user_str_comp.substr(pos1, pos2-pos1);
        		attribute_sstr_comp.clear();
        		attribute_sstr_comp.str(attribute_str_comp);
        		attribute_sstr_comp >> iid_comp2;
        		pos1 = pos2+1;		
        		
        		ratings_train_comp.back().push_back(make_pair(iid_comp1, iid_comp2));		                 
    	    }
    	  //  if(ratings_train[ratings_train.size()-1].size()==0) break;
    	}
    }

    f.close();
    f_comp.close();
    n_users = ratings_train.size();
    std::cout << ratings_train.size() << " users in svm train file; " << ratings_train_comp.size() << " users in pairwise train file" << endl; 
    std::printf("%d users, %d items in training file \n", n_users, n_items);
}

void InputData::readInputTest(std::string& filename){
 // std::vector<std::vector<std::pair<int, double>>> ratings;
    std::string user_str, attribute_str;
    std::stringstream attribute_sstr; 

    std::ifstream f;
    f.open(filename);
    if(f.is_open()){
    	int  iid;	
    	double sc;
    
    	while(!f.eof()) {
    	    getline(f, user_str);
    	    if(!user_str.empty()) ratings_test.resize(ratings_test.size() + 1);
    	    size_t pos1 = 0, pos2;
    	    while(1){
        		pos2 = user_str.find(':', pos1);
        		if(pos2 == std::string::npos) break;
        		attribute_str = user_str.substr(pos1, pos2-pos1);
        		attribute_sstr.clear();
        		attribute_sstr.str(attribute_str);
        		attribute_sstr >> iid;
                        
        		n_items = std::max(n_items, iid);
        		//--iid; // let iid start from 0;
        		pos1 = pos2+1;
        		pos2 = user_str.find(' ', pos1);
        		attribute_str = user_str.substr(pos1, pos2-pos1);
        		attribute_sstr.clear();
        		attribute_sstr.str(attribute_str);
        		attribute_sstr >> sc;
        		pos1 = pos2+1;		
        		
        		ratings_test.back().push_back(make_pair(iid, sc));		                 
    	    }
    	  //  if(ratings_test[ratings_test.size()-1].size()==0) break;
    	}
    }
    f.close();
    if((int)ratings_test.size()!= n_users)
        std::cout << "Number of train users does not match number of test users" << std::endl;
    std::printf("%d users, %d items in test file \n", n_users, n_items);
}

#endif
