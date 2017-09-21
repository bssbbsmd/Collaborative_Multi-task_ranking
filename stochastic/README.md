### Improved and Scalable Bradley-Terry (BT) Model for Collaborative Ranking
We combine a regression model to the BT model in order to resolve the "identifiability" issue and "numerical challenge".
We implemented two parallel solvers: SGD and SSGD (recommended) based on the Hogwild! parallel framework.

### Prerequisite for installation 
We develop the algorithms on a UNIX-based system with a C++11 supporting compiler and OpenMP API, hence users should install c++11 and OpenMP first!!!

compile using the Makefile 

$ make

### Preprocessing of Input Data
We provide a python script (util/num2coms.py) that transfers a (user, item, rating) dataset to a training set and test set: 
1. Prepare a dataset with (user, item, ratings) triple. (Example: data/movielens1m.txt)
2. Run util/num2comp.py to get training comparisons and test ratings. 
    
    $ python util/num2comps.py data/movielens1m.txt -o ml1m -n 50
    
    after "-o" is the name of output file, after "-n" is the number of ratings extracted for each user for training and hence users who has less than n+10 ratings will be removed.  
3. Set the configuration options. (Example: config/default.cfg)

    [input]
    type = numeric
    train_file = data/ml1m_train_comps_ratings_50.lsvm
    test_file  = data/ml1m_test_ratings_50.lsvm

#### Experiments 
Users can configures the settings in the config/default.cfg file. Some recommended parameter settings has been shown in that file.

Run the code

    $ ./pairrank

