## Prerequisite for installation 
We develop the algorithms on a UNIX-based system with a C++11 supporting compiler and OpenMP API, hence users should install c++11 and [OpenMP](https://bisqwit.iki.fi/story/howto/openmp/) first!!!

compile using the Makefile 

```
$ make
```

## Preprocessing of Input Data

check config/default.cfg for the introduction of input parameters. In particular, five input files need to be generated as the input data. Check [util](https://github.com/bssbbsmd/Collaborative_Multi-task_ranking/tree/master/stochastic2/util) about how to generate the following input files from source movielens dataset. 

```
train_rating_file = /datadisk/disk1/Jun/data/svm_data_userpairs/ml100k_new_train_20.rating
```
this is the raw training rating file, each rows contains tuples: userid itemid rating. e.g., [ml100k_new_train_20.rating](https://github.com/bssbbsmd/Collaborative_Multi-task_ranking/blob/master/stochastic/util/svm_data_itemwise/ml100k_new_train_20.rating). The first row contains number_users  number_items. 

```
itemwise_train_comps_file = /datadisk/disk1/Jun/data/svm_data_userpairs/ml100k_new_train_20.pair
```
this is the raw row-wise training pairs for pairwise optimization; e.g., [ml100k_new_train_20.pair](https://github.com/bssbbsmd/Collaborative_Multi-task_ranking/blob/master/stochastic/util/svm_data_itemwise/ml100k_new_train_10.pair). The first row contains number_users  number_items. other rows are formatted as one pair of items. 

```
userwise_train_comps_file = /datadisk/disk1/Jun/data/svm_data_userpairs/ml100k_train_50_add.pair
```
this is the raw column-wise training pairs for pairwise optimization. similar to previous one

```
itemwise_test_file = /datadisk/disk1/Jun/data/svm_data_userpairs/ml100k_test_50.lsvm
```
this is the input test file for item ranking; formatted as lbsvm. e.g., [ml100k_new_test_10.lsvm](https://github.com/bssbbsmd/Collaborative_Multi-task_ranking/blob/master/stochastic/util/svm_data_itemwise/ml100k_new_test_10.lsvm). First row contains number_users and number_items. 

```
userwise_test_file  = /datadisk/disk1/Jun/data/svm_data_userpairs/test.lsvm
```
this is the input test file for user ranking; 

## Experiments 
Users can configures the settings in the config/default.cfg file. Some recommended parameter settings has been shown in that file.

Run the execute file (highly recommend to generate the execute file from source code, since in different c++ or computer enviroment, the provided execute may not run successfully): 

    $ ./pairrank

