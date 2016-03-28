cdef class HingeLoss:
    cdef double loss(self, double p, double y) except *
    # the feature vector should be normalized to 1 (2norm)
    # this is the negative derivation 
    cdef double dloss(self, double p, double y) except *
