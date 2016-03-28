# LinearSVMImpl
Implementation of linear svm with l2-penalty and hinge loss, without interception. 

Results of the perforamce on text classification tasks. The Web Track from TREC is used, to compare against 
the sklearn.svm.LinearSVM(C=10000, class_weight=None, dual=True). The documents are spitted, and 60% documents are used as training data.

The reported gain/loss is based on average F1 score from binary classification with GD for 50 queries in each year.

wt11 -7.1%
wt12 -5.2%
wt13 -1.4%
wt14 +0.7%
