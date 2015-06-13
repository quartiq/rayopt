# -*- coding: utf8 -*-
#
#   pyrayopt - raytracing for optical imaging systems
#   Copyright (C) 2015 Robert Jordens <jordens@phys.ethz.ch>
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function, absolute_import, division
import cython
import numpy as np
cimport numpy as np

np.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
def simplex_mul(np.ndarray[np.uint16_t, ndim=3] abi,
                np.ndarray[np.double_t, ndim=1] a,
                np.ndarray[np.double_t, ndim=1] b,
               ):
    cdef int n, j, m = a.shape[0]
    cdef np.ndarray[np.double_t, ndim=1] c = np.zeros((m,), np.double)
    for n in range(m):
        for j in range(1, abi[n, 0, 0] + 1):
            c[n] += a[abi[n, j, 0]] * b[abi[n, j, 1]]
    return c


@cython.boundscheck(False)
@cython.wraparound(False)
cdef long binom_(long n, long k):
    cdef long p = 1, q = 1, i
    if k < 0 or k > n:
        return 0
    for i in range(min(k, n - k)):
        p *= n - i
        q *= i + 1
    return p//q


@cython.boundscheck(False)
@cython.wraparound(False)
def simplex_transform_fast(np.ndarray[np.uint16_t, ndim=3] idx,
                           np.ndarray[np.uint16_t, ndim=2] jdx,
                           double r, double u, double w,
                           np.ndarray[np.double_t, ndim=1] s,
                           np.ndarray[np.double_t, ndim=1] t):
    cdef int i, j, k, l, m, n, p, v, q = s.shape[0]
    cdef double s1, t1, c
    cdef np.ndarray[np.double_t, ndim=2] bst = np.zeros((q, 2), np.double)
    for p in range(q):
        i = jdx[p, 0]
        j = jdx[p, 1]
        k = jdx[p, 2]
        s1 = r*s[p] - u*t[p]
        t1 = -w*t[p]
        for l in range(k + 1):
            for m in range(j + 1):
                for n in range(m + 1):
                    c = (r**(2*i + k)*u**(2*j - 2*m + l + n)*w**(2*m + k - l - n)*
                         2**n*(-1)**k*binom_(k, l)*binom_(j, m)*binom_(m, n))
                    v = idx[i + j + l - m, m - n, k - l + n]
                    bst[v, 0] += c*s1
                    bst[v, 1] += c*t1
    return bst
