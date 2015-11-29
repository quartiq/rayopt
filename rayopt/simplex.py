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

from .simplex_accel import *

"""Truncated multinomial tools.

Notes on tests and functionality:

Forward and backward transformations:

    S = make_simplex(3, 7)
    a = np.random.randn(1, S.q) #.view(S)
    t = random_rotation_matrix()[:3, :3].copy()
    b = simplex_transform(S.i.ravel(), S.j, a, t)
    c = simplex_transform(S.i.ravel(), S.j, b, t.T.copy())
    nptest.assert_allclose(a, c)

Benchmarking and size scaling:

    S = make_simplex(3, 11)
    print([simplex_size(3, n) for n in np.arange(S.n + 1)])
    print(S.abi[:, 0].max())
    a = S().shift(3)
    b = a*10
    %timeit a*b
    %timeit a**-.5

    [0, 1, 4, 10, 20, 35, 56, 84, 120, 165, 220, 286]
    80
    10000 loops, best of 3: 20.2 µs per loop
    10000 loops, best of 3: 154 µs per loop
"""


def simplex_iter(d, m):
    """Yield index tuples covering the m-scaled d+1 simplex (
    d+1 cube corner N^d with edge length m - 1."""
    if d == 0:
        yield ()
    else:
        for i in range(m):
            for j in simplex_iter(d - 1, i + 1):
                yield (i - sum(j),) + j


def simplex_size(d, m):
    """Count points in the d-m simplex."""
    n = 1
    p = 1
    for i in range(d):
        n *= m + i
        p *= i + 1
    return n//p


def simplex_enum(d, m):
    """Return an ordered forward and backward mapping of the points in the d-m
    simplex.

    idx[j] == (i_0, i_1, ..., i_{d-1})
    jdx[i_0, i_1, ..., i_{d-1}] == j (only the simplex close to the origin is
    valid).
    """
    idx = np.zeros((m,)*d, dtype=np.uint16)
    jdx = np.zeros((simplex_size(d, m), d), dtype=np.uint16)
    for j, i in enumerate(simplex_iter(d, m)):
        idx[i] = j
        jdx[j] = i
    assert jdx.shape[0] == j + 1, (jdx.shape, j)
    return idx, jdx


def simplex_idx(d, m):
    """Build index arrays for multiplication. See `simplex_mul_i`.
    """
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


def make_simplex(d0, n0):
    class Simplex(np.ndarray):
        """
        Truncated multinomial over R^d of maximal order n.

        The coefficients cover the n-scaled (d+1)-simplex (cube corner in
        N^d with edge length n - 1).

        p(x_0, x_1, ..., x_{d - 1}) =
            \Sum_{i_{d - 1} = 0}^{n - 1 - i_0 - i_1 - ... - i_{d - 2}}
            ...
            \Sum_{i_1 = 0}^{n - 1 - i_0}
            \Sum_{i_0 = 0}^{n - 1}
            p_{i_0, i_1, ..., i_{d - 1}}
            \Prod_{j = 0}^{d - 1} x_{j}^{i_j}

        Number of coefficients in `p.q`.
        Allowed `i_0, i_1, ..., i_{d - 1}` indices are listed in `p.i`.
        Their reverse mapping is in `p.j`.

        Operations supported are addition, difference, products, rational
        powers, evaluation, shifting, and linear transformations.
        Division is not supported.

        Multiplication and power are implemented in simplex_accel.pyx for
        speed.
        """
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
            """Shift multinomial by constant.

            `p.shift(a)` offsets the multinomial `p` by constant `a`
            in-place and returns the shifted multinomial.
            """
            self[0] += a
            return self

        def __call__(self, *x):
            """Evaluate multinomial.

            `p(*x)` evaluates the multinomial at `x` (first dimensions
            being `d`): R^d x R^m -> R^m.
            """
            assert len(x) == self.d
            x = np.array(np.broadcast_arrays(*x))
            return simplex_eval(self.j, self, x)

        def transform(self, t):
            """Linear transformation with matrix `t`
            """
            p = simplex_transform(self.i.ravel(), self.j, self, t)
            return p.view(self.__class__)


    Simplex.__name__ = "Simplex{}d{}n".format(d0, n0)
    return Simplex
