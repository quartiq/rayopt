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

#cython: boundscheck=False, wraparound=False, cdivision=True,
#cython: embedsignature=True, initializedcheck=False

from __future__ import print_function, absolute_import, division

import cython
import numpy as np
cimport numpy as np

np.import_array()


cpdef inline int simplex_mul_i(unsigned short[:, :, ::1] abi,
                               double[::1] a, double[::1] b, double[::1] c) nogil:
    cdef int i, j, m = a.shape[0]
    for i in range(m):
        c[i] = 0
        for j in range(1, abi[i, 0, 0] + 1):
            c[i] += a[abi[i, j, 0]] * b[abi[i, j, 1]]
    return 0


cpdef simplex_mul(unsigned short[:, :, ::1] abi,
                  double[::1] a, double[::1] b):
    """Multiply two d-n simplex multinomials, `a * b`"""
    cdef int m = a.shape[0]
    cd = np.zeros((m,), np.double)
    cdef double[::1] c = cd
    with nogil:
        simplex_mul_i(abi, a, b, c)
    return cd


cpdef simplex_pow(unsigned short[:, :, ::1] abi,
                  int m, double[::1] a, double p):
    """Return the `p`th power of the d-n simplex multinomial `a`"""
    cdef int i, j, n = a.shape[0]
    cdef double[::1] x = a.copy()
    cdef double[::1] w = np.empty((n,), np.double)
    cdef double[::1] y
    zd = np.empty((n,), np.double)
    cdef double[::1] z = zd
    x[0] = 0.
    y = x.copy()
    for i in range(n):
        y[i] *= p
    z[:] = y.copy()
    z[0] += 1.
    with nogil:
        for i in range(1, m):
            simplex_mul_i(abi, x, y, w)
            for j in range(n):
                y[j] = (p - i)/(i + 1.)*w[j]
                z[j] += y[j]
        for i in range(n):
            z[i] *= a[0]**p
    return zd


cpdef simplex_eval(unsigned short[:, ::1] jdx, double[::1] a, double[:, ::1] x):
    """Evaluate d-n simplex multinomial `a` at points `x` in R^d"""
    cdef int i, j, k
    cdef int m = a.shape[0], d = x.shape[0], e = x.shape[1], n = jdx[-1, -1]
    cdef double yi
    cdef double [:, :, ::1] xp = np.empty((e, d, n), np.double)
    yd = np.zeros((e,), np.double)
    cdef double[::1] y = yd
    with nogil:
        for k in range(e):
            for i in range(d):
                xp[k, i, 0] = x[i, k]
                for j in range(1, n):
                    xp[k, i, j] = xp[k, i, j - 1]*x[i, k]
        for k in range(e):
            for i in range(m):
                yi = a[i]
                for j in range(d):
                    if jdx[i, j] > 0:
                        yi *= xp[k, j, jdx[i, j] - 1]
                y[k] += yi
    return yd


cpdef inline int binom(int n, int k) nogil:
    cdef int r = 1, i
    if k < 0 or k > n:
        return 0
    for i in range(1, min(k, n - k) + 1):
        r *= n
        r /= i
        n -= 1
    return r


cpdef inline int multinom(unsigned short[::1] k) nogil:
    cdef int t = k[0], r = 1
    cdef int i, n = k.shape[0]
    for i in range(1, n):
        t += k[i]
        r *= binom(t, k[i])
    return r


cpdef inline int multinom_next(int p, unsigned short[::1] q) nogil:
    cdef int p0 = p, n = q.shape[0], i
    if q[n - 1] == p:
        return 0
    for i in reversed(range(n)):
        if q[i] == p0:
            q[i + 1] += 1
            q[i] = 0
            q[0] = p0 - 1
            break
        p0 -= q[i]
    return multinom(q)


cpdef inline int supernom_next(unsigned short[::1] p,
                               unsigned short[:, ::1] q,
                               unsigned short[::1] m) nogil:
    cdef int d = p.shape[0], i, j
    for i in reversed(range(d)):
        m[i] = multinom_next(p[i], q[i])
        if m[i]:
            return True
        else:
            q[i, 0] = p[i]
            for j in range(1, d):
                q[i, j] = 0
            m[i] = 1
    return False


cpdef simplex_transform(unsigned short[::1] idx,
                        unsigned short[:, ::1] jdx,
                        double[:, ::1] x, double[:, ::1] t):
    """Transform d-n simplex multinomial `x` with the linear transformation
    matrix `t`: R^d -> R^d"""
    cdef int n = jdx.shape[0], d = jdx.shape[1], r = x.shape[0]
    cdef int s = jdx[n - 1, d - 1] + 1
    cdef int i, j, k, l
    cdef double q
    yd = np.zeros((r, n), np.double)
    cdef double[:, ::1] y = yd
    cdef unsigned short[:, ::1] p = np.zeros((d, d), np.uint16)
    cdef unsigned short[::1] m = np.ones((d,), np.uint16)
    cdef unsigned short[::1] ji
    with nogil:
        for i in range(n):
            ji = jdx[i]
            for j in range(d):
                p[j, 0] = ji[j]
            while True:
                q = 1.
                l = 0
                for j in range(d):
                    q *= m[j]
                    l *= s
                    for k in range(d):
                        q *= t[k, j]**p[k, j]
                        l += p[k, j]
                l = idx[l]
                for j in range(r):
                    y[j, l] += q*x[j, i]
                if not supernom_next(ji, p, m):
                    break
    return yd


cpdef finite_object_fast(unsigned short[:, :, :] idx,
                    unsigned short[:, :] jdx,
                    double r, double u, double w,
                    double[:] s, double[:] t):
    """Special case of simplex_transform"""
    cdef int i, j, k, l, m, n, p, v, q = s.shape[0]
    cdef double s1, t1, c
    bstd = np.zeros((q, 2), np.double)
    cdef double[:, :] bst = bstd
    with nogil:
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
                             2**n*(-1)**k*binom(k, l)*binom(j, m)*binom(m, n))
                        v = idx[i + j + l - m, m - n, k - l + n]
                        bst[v, 0] += c*s1
                        bst[v, 1] += c*t1
    return bstd
