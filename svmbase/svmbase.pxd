# %%cython
from lossfuncs.lossfuncs cimport *

cdef class SvmBase:
    cdef double alpha
    cdef HingeLoss lossfunc
