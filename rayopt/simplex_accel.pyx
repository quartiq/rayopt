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


@cython.boundscheck(False)
@cython.wraparound(False)
def simplex_mul(np.ndarray[np.uint16_t, ndim=3] abi,
                np.ndarray[np.double_t, ndim=1] a,
                np.ndarray[np.double_t, ndim=1] b,
               ):
    cdef int n, j, m = a.shape[0]
    cdef np.ndarray[np.double_t, ndim=1] c = np.zeros((m,), dtype=np.double)
    for n in range(m):
        for j in range(1, abi[n, 0, 0] + 1):
            c[n] += a[abi[n, j, 0]] * b[abi[n, j, 1]]
    return c
