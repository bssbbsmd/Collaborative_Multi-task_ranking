### Generate Userpairs files

```
python rating2input_userpairs.py ../../data/raw_rating_data/ml1m.txt -o ../../data/svm_data_userpairs/ml1m -n 10 -t 10
```



## GENERATE FILES FOR COMPARIOSN

### Generate two (training + test) lsvm file (for cofirank type algorithms) 

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