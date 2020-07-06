## GENERATE ITEMPAIR FILES for PERSONALIZED RANKING Optimization

```
python rating2input_itempairs.py ../../data/raw_rating_data/ml100k.txt -o ../../data/svm_data_itempairs/ml100k -n 50 -t 10
```
'-o': Prefix for the output files;

'-n': Number of training items per user (Default 50);

'-t': Minimum number of test items per user (Default 10);

'-d1': if delete the first line: 1 delete; 0 (default) keep it; 

The output of the above commend contains 4 files: 
+ ml100k_train_50.pair: the training item pairs for each user;
+ ml100k_test_50.lsvm: the extracted test data;
+ ml100k_train_50.rating: the training file, formatted as uid, iid, ratings;
+ ml100k_train_50_add.pair: the training user pairs for each item. It should be mentioned that the users pairs here are obtained from ml100k_train_50.pair. Therefore, for each item the number of users which rated this item are not set. If we target at user ranking task, we need to generate the training data using the following command. 

## GENERATE USERPAIR FILES for USER TARGETING Optimization

```
python rating2input_userpairs.py ../../data/raw_rating_data/ml1m.txt -o ../../data/svm_data_userpairs/ml1m -n 50 -t 10
```



## Other Useful Tool (not related to input data preprocessing)

### GENERATE FILES FOR COMPARIOSN

Generate two (training + test) lsvm file (for cofirank type algorithms) 

**The following programs are run on the output of the previous command**

1. gen svm test file from test.lsvm file

```
python svm2svm.py -i ../../data/svm_data_userpairs/ml100k_test_10.lsvm -o ../../data/compare_ut/ml100k_test_10
```

2. gen svm training file from train.rating file

```
python rating2svm.py -i ../../data/svm_data_userpairs/ml100k_train_50.rating -o ../../data/compare_ut/ml100k_train_50
```

### Generate two (training + test) rating file (for FM-like methods)


```
 python svm2rating.py -i ../../data/compare_ut/ml1m_train_10.svm -o ../../data/compare_ut/ml1m_train_10 
```
