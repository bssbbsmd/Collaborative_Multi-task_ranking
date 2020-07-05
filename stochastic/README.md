## Prerequisite for installation 
We develop the algorithms on a UNIX-based system with a C++11 supporting compiler and OpenMP API, hence users should install c++11 and [OpenMP](https://bisqwit.iki.fi/story/howto/openmp/) first!!!

compile using the Makefile 

```
$ make
```

### Preprocessing of Input Data

check config/default.cfg for the introduction of input parameters. In particular, five input files are needed to be generated as the input data

```
train_rating_file = /datadisk/disk1/Jun/data/svm_data_userpairs/ml100k_new_train_20.rating
```
this is the raw training rating file, each rows contains tuples: userid itemid rating. e.g., [ml100k_new_train_20.rating](https://github.com/bssbbsmd/Collaborative_Multi-task_ranking/blob/master/stochastic/util/svm_data_itemwise/ml100k_new_train_20.rating)

```
itemwise_train_comps_file = /datadisk/disk1/Jun/data/svm_data_userpairs/ml100k_train_50.pair
```
this is the raw training pairs for 

trai
userwise_train_comps_file = /datadisk/disk1/Jun/data/svm_data_userpairs/ml100k_train_50_add.pair
itemwise_test_file = /datadisk/disk1/Jun/data/svm_data_userpairs/ml100k_test_50.lsvm
userwise_test_file  = /datadisk/disk1/Jun/data/svm_data_userpairs/test.lsvm
```

#### Experiments 
Users can configures the settings in the config/default.cfg file. Some recommended parameter settings has been shown in that file.

Run the code

    $ ./pairrank

