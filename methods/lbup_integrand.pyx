# The integrand here is phi(x, n, rhos, eta)

# NB: this has to be a cdef, not a cpdef or def, because these add
# extra stuff to the argument list to help python. LowLevelCallable
# does not like these things...

# You can however increase the number of arguments (remember also to
# update test.pxd)

import numpy as np
from libc.math cimport exp, log

cdef double phi(int m, double* args):
    x = args[0]
    n = int(args[1])  # order
    rhos = list()
    for i in range(2 * n - 1):
        rhos.append(args[i + 2])
    eta = args[2 * n + 1]

    return exp(np.sum([rhos[k - 1] * ((1 - x) ** k) / k
                       for k in range(1, 2 * n)]) + eta * log(x))
