# %%cython
import pyximport;pyximport.install(reload_support=True)
from toolkit cimport *
from lossfuncs.lossfuncs import HingeLoss
from lossfuncs.lossfuncs cimport HingeLoss
import numpy as np
cimport numpy as np

cdef class SvmBase:
    cdef double alpha
    cdef HingeLoss lossfunc
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.lossfunc = HingeLoss()

    # L(w)
    def opt(self, 
            np.ndarray[np.float64_t,ndim=1] w, 
            np.ndarray[np.float64_t,ndim=2] X, 
            np.ndarray[np.float64_t,ndim=1] ys, 
            np.ndarray[np.float64_t,ndim=1] classw=np.empty((0,)), 
            np.ndarray[np.float64_t,ndim=1] pointw=np.empty((0,))
            ):
        # point weight
        cdef np.ndarray[np.float64_t,ndim=1] pweights = \
                np.ones((X.shape[0],)) if  pointw.shape[0] != X.shape[0] \
                else pointw
        # class weight
        cdef np.ndarray[np.float64_t,ndim=1] cweights = \
                np.ones((X.shape[0],)) if  classw.shape[0] != X.shape[0] \
                else classw
        cdef np.ndarray[np.float64_t,ndim=1] x
        # alpha w^2 /2
        cdef double norm2Penalty = self.alpha * np.dot(w, w) * 0.5
        cdef double hingeloss = 0
        cdef double p, pw, y
        # \sum_{all points} {loss(y, w^T x) * datapointweight}
        for x, y, cw, pw in zip(X, ys, cweights, pweights):
            p = np.dot(w, x)
            hingeloss += self.lossfunc.loss(p, y) * pw * cw
        return hingeloss + norm2Penalty
    
    # dL(w) / dw on all data points
    def doptgd(self, np.ndarray[np.float64_t,ndim=1] w, 
            np.ndarray[np.float64_t,ndim=2] X, 
            np.ndarray[np.float64_t,ndim=1] ys,
            np.ndarray[np.float64_t,ndim=1] classw=np.empty((0,)), 
            np.ndarray[np.float64_t,ndim=1] pointw=np.empty((0,))
            ):
        # point weight
        cdef np.ndarray[np.float64_t,ndim=1] pweights = \
                np.ones((X.shape[0],)) if  pointw.shape[0] != X.shape[0] \
                else pointw
        # class weight
        cdef np.ndarray[np.float64_t,ndim=1] cweights = \
                np.ones((X.shape[0],)) if  classw.shape[0] != X.shape[0] \
                else classw
        cdef np.ndarray[np.float64_t,ndim=1] x
        cdef double dhingeloss = 0
        cdef double p, pw, y
        # alpha w
        cdef np.ndarray[np.float64_t,ndim=1] dw = self.alpha * w
        # \sum_{all data points} {dloss(y,w^Tx) * datapointweight}
        for x, y, cw, pw in zip(X, ys, cweights, pweights):
            p = np.dot(w, x)
            dhingeloss = self.lossfunc.dloss(p, y) 
            if dhingeloss != 0:
                dw += cw * pw * dhingeloss * x
        return dw
    
    # dL(w) / dw on single data pints
    def doptsgd(self, np.ndarray[np.float64_t,ndim=1] w, 
            np.ndarray[np.float64_t,ndim=1] xpoint, 
            double ypoint,
            double classweight=1.0,
            double pointweight=1.0
            ):
        cdef np.ndarray[np.float64_t,ndim=1] x = xpoint
        cdef double dhingeloss = 0
        cdef double p, pw = pointweight, cw = classweight, y = ypoint
        # alpha w
        cdef np.ndarray[np.float64_t,ndim=1] dw = self.alpha * w
        # dloss(y,w^Tx) * datapointweight
        p = np.dot(w, x)
        dhingeloss = self.lossfunc.dloss(p, y)
        if dhingeloss != 0:
            dw += cw * pw * dhingeloss * x
        return dw
