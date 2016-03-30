# LinearSVMImpl
Implementation of linear svm with l2-penalty and hinge loss, without interception. 

Results of the perforamce on text classification tasks. The Web Track from TREC is used, to compare against 
the sklearn.svm.LinearSVM(C=10000, class_weight=None, dual=True). The documents are spitted, and 60% documents are used as training data.

The reported gain/loss is based on average F1 score from binary classification with GD for 50 queries with 300 iterations.

                                                                              | Gain (+) /Loss (-)
------------------------------------------------------------------------------|--------
[TREC 2011 Web Track: Topics 101-150](http://trec.nist.gov/data/web2011.html)  | -7.1%
[TREC 2012 Web Track: Topics 151-200](http://trec.nist.gov/data/web2012.html) | -5.2%
[TREC 2013 Web Track: Topics 201-251](http://trec.nist.gov/data/web2013.html) | -1.4%
[TREC 2014 Web Track: Topics 251-300](http://trec.nist.gov/data/web2014.html) | +0.7%
