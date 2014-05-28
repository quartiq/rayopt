# -*- coding: utf8 -*-
#
#   pyrayopt - raytracing for optical imaging systems
#   Copyright (C) 2012 Robert Jordens <jordens@phys.ethz.ch>
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

import logging
logging.basicConfig(level=logging.WARNING)

import numpy as np
from numba import jit


def no_jit(*args):
    def w(f):
        return f
    return w

#jit = no_jit

@jit("bool_(double[:], double)")
def clip(y, radius):
    r2 = y[0]*y[0] + y[1]*y[1]
    return r2 > radius**2

@jit("double(double[:], double, double)")
def sag(y, c, k):
    r2 = y[0]*y[0] + y[1]*y[1]
    e = c*r2/(1 + np.sqrt(1 - (1 + k)*c*c*r2))
    return y[2] - e

@jit("double[:](double[:], double, double)")
def sag_normal(y, c, k):
    r2 = y[0]*y[0] + y[1]*y[1]
    e = c/np.sqrt(1 - (1 + k)*c*c*r2)
    n = np.array([-y[0]*e, -y[1]*e, 1])
    return n

@jit("double(double[:], double[:], double, double)")
def intercept(y, u, c, k):
    if c == 0:
        return -y[2]/u[2]
    d = c*(u[0]*y[0] + u[1]*y[1] + u[2]*(1 + k)*y[2]) - u[2]
    e = c*(u[0]*u[0] + u[1]*u[1] + u[2]*(1 + k)*u[2])
    f = c*(y[0]*y[0] + y[1]*y[1] + y[2]*(1 + k)*y[2]) - 2*y[2]
    s = (-d - np.sign(u[2])*np.sqrt(d*d - e*f))/e
    return s

@jit("double[:](double[:], double[:], double, double, double)")
def refract(y, u0, mu, c, k):
    if mu == 1:
        return u0
    r = sag_normal(y, c, k)
    r2 = np.dot(r, r)
    muf = np.fabs(mu)
    a = muf*np.dot(u0, r)/r2
    # solve g**2 + 2*a*g + b=0
    if mu == -1: # reflection
        u = u0 - 2*a*r
    else: # refraction
        b = (mu**2 - 1)/r2
        g = -a + np.sign(mu)*np.sqrt(a**2 - b)
        u = muf*u0 + g*r
    return u

@jit("void(double, double, double, double, double[:, :], double[:, :], double, double, "
        "bool_, double[:, :], double[:, :], double[:])")
def fast_propagate(curvature, conic, radius, mu, y0, u0, n0, l, do_clip, y, u, t):
    for i in range(y0.shape[0]):
        y0i, u0i = y0[i], u0[i]
        s = intercept(y0i, u0i, curvature, conic)
        t[i] = n0*s
        y[i] = y0i + s*u0i
        if do_clip and clip(y[i], radius):
            u[i] = np.nan
        else:
            u[i] = refract(y[i], u0i, mu, curvature, conic)
