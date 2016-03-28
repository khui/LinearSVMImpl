# %%cython
import os
import pyximport;pyximport.install(reload_support=True, build_dir=os.getcwd() + "/build")
from svmbase.svmbase import SvmBase
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from lossfuncs.lossfuncs import HingeLoss
import random

class SvmTrain(SvmBase):
    
    def __init__(self, alpha=0.0001, \
            class_weight='balanced', point_weight=None):
        self.class_weight=class_weight
        self.point_weight=point_weight
        self.alpha = alpha
        super(SvmTrain, self).__init__(alpha=alpha)
    
    def _initeta(self):
        hloss = HingeLoss()
        typw = np.sqrt(1.0 / np.sqrt(self.alpha))
        # computing eta0, the initial learning rate
        initial_eta0 = typw / max(1.0, hloss.dloss(-typw, 1.0))
        # initialize t such that eta at first sample equals eta0
        self.optimal_init = 1.0 / (initial_eta0 * self.alpha)

    def _updateeta(self, t):
        return 1.0 / (self.alpha * (self.optimal_init + t - 1))
        

    def train(self, X_train, ys_train, wt0=None):
        ys_revised = [1 if y > 0 else -1 for y in ys_train]
        X = X_train.astype('float64')
        ys = np.array(ys_revised)
        if self.class_weight == 'balanced':
            labels = set(ys)
            n_samples = float(X.shape[0])
            n_classes = float(len(labels))
            class_counts = {l:ys[ys==l].shape[0] for l in labels}
            self.class_weight = {l:n_samples/(c * n_classes) for l, c in class_counts.items()}
        else:
            self.class_weight={-1:1.0,1:1.0}
        return self._train(X, ys.astype('float64'), wt0)
    
    def confidence(self, X, w):
        confs = np.dot(w, X.transpose())
        return confs
    
    def predict(self, X, w):
        confs = self.confidence(X, w)
        return [1 if c > 0 else 0 for c in confs]
    
    def evalweight(self, X, w, y_true):
        y_pred = self.predict(X, w)
        f1 = f1_score(y_true, y_pred, pos_label=1)
        precision = precision_score(y_true, y_pred, pos_label=1)
        recall = recall_score(y_true, y_pred, pos_label=1)
        return f1, precision, recall
    
    def evalpredict(self, X_train, y_train, X_test, y_test):
        w = self.train(X_train, y_train)
        return self.evalweight(X_test, w, y_test)
    
    def getopt(self, wt, X_train, ys_train):
        ys_revised = [1 if y > 0 else -1 for y in ys_train]
        X = X_train.astype('float64')
        w = wt.astype('float64').reshape((X_train.shape[1],))
        ys = np.array(ys_revised).astype('float64')
        return SvmBase.opt(self, w, X, ys)
    
    def _train(self, X, ys, wt0=None):
        raise NotImplementedError()

class SvmTrainGD(SvmTrain):
    
    def __init__(self, alpha=0.0001, iteration=1000, \
            class_weight='balanced', point_weight=None):
        self.iteration = iteration
        self.alpha = alpha
        super(SvmTrainGD, self).__init__(alpha=alpha, class_weight=class_weight, point_weight=point_weight)
    
    def _train(self, X, ys, wt0=None):
        SvmTrain._initeta(self)
        fnum = X.shape[1]
        wt = np.zeros((fnum,), dtype=np.float64) if wt0 is None else wt0
        cweights = np.array([self.class_weight[y] for y in ys]).astype('float64')
        pweights = np.ones((X.shape[0],), dtype=np.float64) if self.point_weight is None else self.point_weight
        t = 1
        while t <= self.iteration:
            eta = SvmTrain._updateeta(self, t)
            dwt = SvmBase.doptgd(self, wt, X, ys, cweights, pweights)
            wt = np.add(wt, -eta * dwt)
            t += 1
        return wt

class SvmTrainSGD(SvmTrain):
    
    def __init__(self, alpha=0.0001, iteration=1, shuffle=True, \
            class_weight='balanced', point_weight=None):
        self.iteration = iteration
        self.alpha = alpha
        self.shuffle=shuffle
        super(SvmTrainSGD, self).__init__(alpha=alpha, class_weight=class_weight, point_weight=point_weight)
    
    def _train(self, X, ys, wt0=None):
        SvmTrain._initeta(self)
        fnum = X.shape[1]
        wt = np.zeros((fnum,), dtype=np.float64) if wt0 is None else wt0
        cweights = [self.class_weight[y] for y in ys]
        pweights = [1.0] * X.shape[0] if self.point_weight is None else self.point_weight
        t = 1
        datapoints = zip(X, ys, cweights, pweights)
        while t <= self.iteration:
            if self.shuffle:
                random.shuffle(datapoints)
            for x, y, cw, pw in datapoints:
                eta = SvmTrain._updateeta(self, t)
                dwt = SvmBase.doptsgd(self, wt, x, y, cw, pw)
                wt = np.add(wt, -eta * dwt)
                t += 1
        return wt
