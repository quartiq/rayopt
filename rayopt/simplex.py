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

import numpy as np

from .simplex_accel import simplex_mul


def simplex_iter(d, m):
    if d == 0:
        yield ()
    else:
        for i in range(m):
            for j in simplex_iter(d - 1, i + 1):
                yield (i - sum(j),) + j


def simplex_size(d, m):
    n = 1
    p = 1
    for i in range(d):
        n *= m + i
        p *= i + 1
    return n//p


def simplex_enum(d, m):
    idx = np.zeros((m,)*d, dtype=np.uint16)
    jdx = np.zeros((simplex_size(d, m), d), dtype=np.uint16)
    for j, i in enumerate(simplex_iter(d, m)):
        idx[i] = j
        jdx[j] = i
    assert jdx.shape[0] == j + 1, (jdx.shape, j)
    return idx, jdx


def simplex_idx(d, m):
    i, j = simplex_enum(d, m)
    a = (m - 1)//3
    b = (m - a - 1)//2
    r = (a + 1)*(b + 1)*(m - a - b)
    abi = np.zeros((j.shape[0], r + 1, 2), dtype=np.uint16)
    for pq in simplex_iter(2*d, m):
        p, q = pq[0::2], pq[1::2]
        l = i[tuple(pi + qi for pi, qi in zip(p, q))]
        abi[l, 0, 0] += 1
        abi[l, abi[l, 0, 0]] = i[p], i[q]
    maxl = abi[:, 0, 0].max()
    assert maxl == abi.shape[1] - 1, (maxl, abi.shape)
    return i, j, abi


def simplex_pow(abi, m, a, p):
    x = a.copy()
    x[0] = 0.
    y = p*x
    z = y.copy()
    z[0] += 1.
    for i in range(1, m):
        y = simplex_mul(abi, x, y)
        y *= (p - i)/(i + 1.)
        z += y
    return a[0]**p*z


def simplex_eval(n, j, a, x):
    x = np.broadcast_arrays(*x)
    y = np.zeros_like(x[0])
    xp = []
    for xi in x:
        xpi = [1, xi]
        for i in range(n - 2):
            xpi.append(xpi[-1]*xi)
        xp.append(xpi)
    for p, ji in zip(a, j):
        yi = p
        for k, jik in enumerate(ji):
            if jik:
                yi *= xp[k][jik]
        y += yi
    return y


def make_simplex(d0, n0):
    class Simplex(np.ndarray):
        d, n = d0, n0
        i, j, abi = simplex_idx(d, n)
        q = j.shape[0]

        def __new__(cls, t=None):
            if t is None:
                t = np.zeros(cls.q, np.double)
            else:
                t = np.asarray(t).astype(np.double)
                assert t.shape[0] == cls.q
            return t.view(cls)

        def __array_finalize__(self, obj):
            if obj is None:
                return
            assert obj.shape == (self.q,)
            assert obj.dtype == np.double

        def __mul__(self, other):
            cls = self.__class__
            if isinstance(other, cls):
                p = simplex_mul(self.abi, self, other)
                return p.view(cls)
            return np.ndarray.__mul__(self, other)

        def __pow__(self, other):
            if np.isscalar(other):
                p = simplex_pow(self.abi, self.n, self, float(other))
                return p.view(self.__class__)
            return np.ndarray.__pow__(self, other)

        def shift(self, a):
            self[0] += a
            return self

        def __call__(self, *x):
            assert len(x) == self.d
            return simplex_eval(self.n, self.j, self, x)

    return Simplex
