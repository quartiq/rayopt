# -*- coding: utf8 -*-
#
#   pyrayopt - raytracing for optical imaging systems
#
#   Copyright (C) 2011-2013 Robert Jordens <jordens@phys.ethz.ch>
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

# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

from __future__ import division
import cython
import numpy as np
cimport numpy as np
np.import_array()
from libc.math cimport sqrt, fabs, M_PI

dtype = np.double
ctypedef np.double_t dtype_t
ctypedef int intc_t

cdef double NAN
NAN = float("nan")


def propagate(object obj not None,
        np.ndarray[dtype_t, ndim=2] y0 not None,
        np.ndarray[dtype_t, ndim=2] u0 not None,
        double n0, double l, int clip=True,
        np.ndarray[dtype_t, ndim=2] y=None,
        np.ndarray[dtype_t, ndim=2] u=None,
        np.ndarray[dtype_t, ndim=1] t=None):
    cdef int i
    cdef int m
    cdef double n
    cdef double r, c, k, d
    r = obj.radius
    c = obj.curvature
    d = obj.thickness
    k = obj.conic
    m = y0.shape[0]
    if y is None:
        y = np.empty([m, 3], dtype=dtype)
    if u is None:
        u = np.empty([m, 3], dtype=dtype)
    if t is None:
        t = np.empty([m], dtype=dtype)
    if obj.material is not None:
        n = obj.material.refractive_index(l)
    else:
        n = n0
    for i in range(m):
        _propagate(&y0[i, 0], &u0[i, 0], &y[i, 0], &u[i, 0], &t[i],
        d, r, n0, n, c, k, clip)
    return y, u, n, t

cdef inline void _propagate(double *y0, double *u0, double *y, double *u,
        double *t, double d, double r, double n0, double n,
        double c, double k, int do_clip):
    y[0] = y0[0]
    y[1] = y0[1]
    y[2] = y0[2] - d
    t[0] = intercept(y, u0, c, k)
    y[0] += t[0]*u0[0]
    y[1] += t[0]*u0[1]
    y[2] += t[0]*u0[2]
    refract(y, u0, n0/n, c, k, u)
    if do_clip:
        clip(y, u, r)
    t[0] *= n0

cdef inline void clip(double *y, double *u, double radius):
    cdef double r2
    r2 = y[0]**2 + y[1]**2
    if r2 > radius**2:
        u[2] = NAN

#cdef double sag(double *y, double c, double k):
#    cdef double r2, e
#    r2 = y[0]**2 + y[1]**2
#    e = c*r2/(1 + sqrt(1 - k*c**2*r2))
#    return y[2] - e

cdef inline void sag_normal(double *y, double c, double k, double *dy):
    cdef double r2, e
    r2 = y[0]**2 + y[1]**2
    e = c/sqrt(1 - k*c**2*r2)
    dy[0] = -y[0]*e
    dy[1] = -y[1]*e
    dy[2] = 1

cdef inline double sign(double y):
    if y > 0:
        return 1.
    elif y < 0:
        return -1.
    else:
        return 0.

cdef inline double intercept(double *y, double *u, double c, double k):
    cdef double d, e, f, s
    if c == 0:
        return -y[2]/u[2]
    d = c*(u[0]*y[0] + u[1]*y[1] + u[2]*k*y[2]) - u[2]
    e = c*(u[0]*u[0] + u[1]*u[1] + u[2]*k*u[2])
    f = c*(y[0]*y[0] + y[1]*y[1] + y[2]*k*y[2]) - 2*y[2]
    s = (-d - sign(u[2])*sqrt(d**2 - e*f))/e
    return s

cdef inline void refract(double *y, double *u0, double mu, double c, double k, double *u):
    # G. H. Spencer and M. V. R. K. Murty
    # General Ray-Tracing Procedure
    # JOSA, Vol. 52, Issue 6, pp. 672-676 (1962)
    # doi:10.1364/JOSA.52.000672
    cdef double r[3], r2, muf, a, b, g
    if mu == 1:
        u[0] = u0[0]
        u[1] = u0[1]
        u[2] = u0[2]
        return
    sag_normal(y, c, k, r)
    r2 = r[0]**2 + r[1]**2 + r[2]**2
    muf = fabs(mu)
    a = muf*(u0[0]*r[0] + u0[1]*r[1] + u0[2]*r[2])/r2
    # solve g**2 + 2*a*g + b=0
    if mu == -1: # reflection
        u[0] = u0[0] - 2*a*r[0]
        u[1] = u0[1] - 2*a*r[1]
        u[2] = u0[2] - 2*a*r[2]
    else: # refraction
        b = (mu**2 - 1)/r2
        g = -a + sign(mu)*sqrt(a**2 - b)
        u[0] = muf*u0[0] + g*r[0]
        u[1] = muf*u0[1] + g*r[1]
        u[2] = muf*u0[2] + g*r[2]
