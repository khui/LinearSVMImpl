from toolkit cimport *
cdef class HingeLoss:
    cdef double loss(self, double p, double y) except *:
        cdef double z = p * y
        if z < 1.0:
            return (1 - z)
        return 0
    cpdef double dloss(self, double p, double y) except *:
        cdef double z = p * y
        if z < 1.0:
            return -y
        return 0

