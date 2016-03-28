# %load_ext Cython
# %reload_ext Cython
import os
import pyximport;pyximport.install(reload_support=True, build_dir=os.getcwd() + "/build")
from lossfuncs.lossfuncs import HingeLoss
from svmbase.svmbase import SvmBase
from svmbase.svmbase cimport SvmBase
from svmtrain import SvmTrainGD, SvmTrainSGD
from sklearn.datasets import make_classification
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
cimport numpy as np

# data generation
n_features=200
X, y = make_classification(n_samples=1000, n_features=n_features, n_redundant=20, n_informative=100,
                           random_state=1, n_clusters_per_class=1)
X = StandardScaler().fit_transform(X)
from sklearn.preprocessing import normalize
normX = normalize(X, norm='l2')
X_train, X_test, y_train, y_test = train_test_split(normX, y, test_size=.6)

alpha=0.0001
# do the test
def gdtest(X_train, y_train, X_test, y_test, step=0.01):
    ys_revised = [1 if y > 0 else -1 for y in y_train]
    cdef np.ndarray[np.float64_t,ndim=2] X = X_train.astype('float64')
    cdef np.ndarray[np.float64_t,ndim=1] ys = np.array(ys_revised).astype('float64')
    svb = SvmBase(alpha=alpha)
    svt =SvmTrainSGD(alpha=alpha, iteration=5)
    w = svt.train(X_train, y_train)
    for i in range(100):
        w = svt.train(X_train, y_train, w)
        evalv = svt.evalweight(X_test, w, y_test)
        doptdg = svb.doptgd(w, X, ys)
        deltaw = doptdg * step
        optv1 = svb.opt(w + deltaw*0.5, X, ys)
        optv0 = svb.opt(w - deltaw*0.5, X, ys)
        dgv = optv1 - optv0 - np.dot(deltaw, doptdg)
        print(i, evalv, dgv, abs(dgv)/optv1)
# call the test function
gdtest(X_train, y_train, X_test, y_test, step=0.0001)