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

using namespace std;

#include "./hybridranking.h"
#include "./InputData.h"

struct configuration{
    //std::string algo="pairregssgd"
    //
    std::string train_file, train_comp_file, test_file;
    int ndcg_k = 10;
    unsigned int rank = 10;
    double learning_rate = 0.1;
    double relative_learning_rate = 1.;
    double alpha = 1.;
    double beta = 0;
    double lambda = 0.1;
    double eps = 0.01;
    unsigned int max_iterations = 0;
    bool initialized = true;
    bool show_seperate_loss = false;
    bool show_each_iteration =false;
    int pairwise_method = 1;
};

int readConf(struct configuration& conf, std::string conFile){
    std::ifstream infile(conFile);
    std::string line;
    while(std::getline(infile, line)){
        std::istringstream iss(line);
        if((line[0] == '#') || (line[0] == '[')) {
            continue;
        }

    	std::string key,equal,val;
    	if(iss >> key >> equal >> val){
	    if(equal != "="){
	        continue;	
	    }
	    if(key == "train_file"){
	        conf.train_file = val;	
	    }
	    if(key == "train_comp_file"){
	        conf.train_comp_file = val;	
	    }
	    if(key == "test_file"){
	        conf.test_file = val;	
	    }
            if(key == "ndcg_k"){
  	        conf.ndcg_k = std::stoi(val);   	
	    }	
	    if(key == "rank"){
  	        conf.rank = std::stoi(val);   	
	    }
	    if(key == "alpha"){
	        conf.alpha = std::stod(val);
	    }
	    if(key == "beta"){
	        conf.beta = std::stod(val);
	    }
	    if(key == "learning_rate"){
   	        conf.learning_rate = std::stod(val);	
	    }   
 	    if(key == "relative_learning_rate"){
   	        conf.relative_learning_rate = std::stod(val);	
	    } 
	    if(key == "lambda"){
	        conf.lambda = std::stod(val);		
	    }
	    if(key == "eps"){
	        conf.eps = std::stod(val);	
	    }  
	    if(key == "pairwise_method"){
		conf.pairwise_method = std::stoi(val);  
	    } 	
	    if(key == "max_iterations"){
	        conf.max_iterations = std::stoi(val);	
	    }
	    if(key == "show_seperate_loss"){
		if(val == "true") conf.show_seperate_loss = true;
	        if(val == "false") conf.show_seperate_loss = false;	
	    }
	    if(key == "show_each_iteration"){
		if(val == "true") conf.show_each_iteration = true;
	        if(val == "false") conf.show_each_iteration = false;
		cout << val << endl;	
	    }
	    if(key == "initialized"){
	        if(val == "true") conf.initialized = true;
	        if(val == "false") conf.initialized = false;	
	    }
	}
    }       
    return 1;
}



void printConf(const struct configuration& conf){
    std::cout << "train_file:"<<conf.train_file<<std::endl;
    std::cout << "test_file:"<<conf.test_file<<std::endl;
    std::cout << "rank:"<<conf.rank<<std::endl;
    std::cout << "learning_rate:"<<conf.learning_rate<<std::endl;
    std::cout << "relative_learning_rate:"<<conf.relative_learning_rate<<std::endl;
    std::cout << "alpha:"<<conf.alpha<<std::endl;
    std::cout << "beta:"<<conf.beta<<std::endl;
    std::cout << "lambda:"<<conf.lambda<<std::endl;
    std::cout << "eps:"<<conf.eps<<std::endl;
    std::cout << "max_iterations:"<<conf.max_iterations<<std::endl;
} 


void printVectorPair(const std::vector<std::pair<int, double>> &data){
    for(unsigned int i=0; i < data.size(); i++){
    	std::cout << data[i].first << ":" << data[i].second << " ";
    }
    std::cout<<std::endl;
}     

void printVectorPair(const std::vector<std::pair<int, int>> &data){
    for(unsigned int i=0; i < data.size(); i++){
    	std::cout << data[i].first << ":" << data[i].second << " ";
    }
    std::cout<<std::endl;
}    

int main(int argc, char* argv[]){
    struct configuration conf;
    std::string config_file = "config/default.cfg";
    if (argc > 2) {
        std::cerr << "Usage : " << std::string(argv[0]) << " [config_file]" << std::endl;
        return -1;
    }

    if (argc == 2) {
        config_file = std::string(argv[1]);
    }
    
    if(!readConf(conf, config_file)){
	std::cerr << "Usage : " << std::string(argv[0]) << " [config_file]" << std::endl;
    	return -1;
    }	

    HybridRanking hbranking;
    
    //assign the training and test data
    hbranking.f_lsvm = conf.train_file;
    hbranking.l_pair = conf.train_comp_file;
    hbranking.test = conf.test_file;
    
    hbranking.ndcg_k = conf.ndcg_k;
    hbranking.rank = conf.rank;
    hbranking.max_iterations = conf.max_iterations;
    
    hbranking.alpha = conf.alpha;
    hbranking.beta = conf.beta;
 //   hbranking.beta = 1 - conf.alpha;

    hbranking.learning_rate = conf.learning_rate;
    hbranking.relative_learning_rate =  conf.relative_learning_rate;
    hbranking.lambda = conf.lambda;
    
    hbranking.eps = conf.eps;
    hbranking.initialized = conf.initialized;
    
    hbranking.show_each_iteration = conf.show_each_iteration;
    hbranking.show_seperate_loss = conf.show_seperate_loss;

    hbranking.pairwise_method = conf.pairwise_method;   

    hbranking.train();
  
    
/* 
    InputData input;
    input.readInputTrain(conf.train_file, conf.train_comp_file);
   
    printVectorPair(input.ratings_train[input.ratings_train.size()-1]);
    printVectorPair(input.ratings_train_comp[input.ratings_train_comp.size()-1]);
*/
}







