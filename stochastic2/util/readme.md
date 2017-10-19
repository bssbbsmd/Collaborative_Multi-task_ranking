## Generate Userpairs files

```
python rating2input_userpairs.py ../../data/raw_rating_data/ml1m.txt -o ../../data/svm_data_userpairs/ml1m -n 10 -t 10
```


## Generate two lsvm file (for cofirank type algorithms) 

**The following programs are run on the output of the previous command**

1. gen svm test file from test.lsvm file

```
python svm2svm.py -i ../../data/svm_data_userpairs/ml100k_test_10.lsvm -o ../../data/svm_data_userpairs/compare_data/ml100k_test_10
```

2. gen svm training file from train.rating file

```
python rating2svm.py -i ../../data/svm_data_userpairs/ml100k_train_50.rating -o ../../data/svm_data_userpairs/compare_data/ml100k_train_50
```
