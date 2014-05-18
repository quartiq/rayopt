# -*- coding: utf8 -*-
#
#   pyrayopt - raytracing for optical imaging systems
#   Copyright (C) 2013 Robert Jordens <jordens@phys.ethz.ch>
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

import sys

import numpy as np
from scipy.special import orthogonal


def public(f):
  """Use a decorator to avoid retyping function/class names.

  * Based on an idea by Duncan Booth:
  http://groups.google.com/group/comp.lang.python/msg/11cbb03e09611b8a
  * Improved via a suggestion by Dave Angel:
  http://groups.google.com/group/comp.lang.python/msg/3d400fb22d8a42e1
  """
  all = sys.modules[f.__module__].__dict__.setdefault('__all__', [])
  if f.__name__ not in all:  # Prevent duplicates if run from an IDE.
      all.append(f.__name__)
  return f

public(public)  # Emulate decorating ourself


@public
def simple_cache(f):
    cache = {}
    def wrapper(*args):
        try:
            return cache[args]
        except KeyError:
            cache[args] = v = f(*args)
            return v
    wrapper.cache = cache
    return wrapper


@public
def tanarcsin(u, v=None):
    u = np.asanyarray(u)
    if u.ndim == 2 and u.shape[1] == 3:
        u1 = u[:, :2]/u[:, 2:]
        if v is not None:
            return u1, np.sign(u[:, 2])
        else:
            return u1
    u2 = np.square(u)
    if u2.ndim == 2:
        u2 = (u2[:, 0] + u2[:, 1])[:, None]
    u1 = u/np.sqrt(1 - u2)
    if v is not None:
        return u1, np.sign(v)
    else:
        return u1


@public
def sinarctan(u, v=None):
    u2 = np.square(u)
    if u2.ndim == 2:
        if u2.shape[1] >= 3:
            v = u[:, 3]
            u, u2 = u[:, :2], u2[:, :2]
        u2 = u2.sum(1)[:, None]
    u2 = 1/np.sqrt(1 + u2)
    u1 = u*u2
    if v is not None:
        u1 = np.concatenate((u1, np.sign(v)[:, None]*u2), axis=1)
    return u1


@public
def sfloat(a):
    try:
        return float(a)
    except ValueError:
        return None


@public
def sint(a):
    try:
        return int(a)
    except ValueError:
        return None


@public
def pupil_distribution(distribution, nrays):
    # TODO apodization
    """returns nrays in normalized aperture coordinates x/meridional
    and y/sagittal according to distribution, all rays are clipped
    to unit circle aperture.
    Returns center ray index, x, y

    meridional: equal spacing line
    sagittal: equal spacing line
    cross: meridional-sagittal cross
    tee: meridional (+-) and sagittal (+ only) tee
    random: random within aperture
    square: regular square grid
    triangular: regular triangular grid
    hexapolar: regular hexapolar grid
    """
    d = distribution
    n = nrays
    if n == 1:
        return 0, np.zeros((n, 2))
    elif d == "half-meridional":
        return 0, np.c_[np.zeros(n), np.linspace(0, 1, n)]
    elif d == "meridional":
        n -= n % 2
        return n/2, np.c_[np.zeros(n + 1), np.linspace(-1, 1, n + 1)]
    elif d == "sagittal":
        n -= n % 2
        return n/2, np.c_[np.linspace(-1, 1, n + 1), np.zeros(n + 1)]
    elif d == "cross":
        n -= n % 4
        return n/4, np.concatenate([
            np.c_[np.zeros(n/2 + 1), np.linspace(-1, 1, n/2 + 1)],
            np.c_[np.linspace(-1, 1, n/2 + 1), np.zeros(n/2 + 1)],
            ])
    elif d == "tee":
        n = (n - 2)/3
        return 2*n + 1, np.concatenate([
            np.c_[np.zeros(2*n + 1), np.linspace(-1, 1, 2*n + 1)],
            np.c_[np.linspace(0, 1, n + 1), np.zeros(n + 1)],
            ])
    elif d == "random":
        r, phi = np.random.rand(2, n)
        xy = np.exp(2j*np.pi*phi)*np.sqrt(r)
        xy = np.c_[xy.real, xy.imag]
        return 0, np.concatenate([[[0, 0]], xy])
    elif d == "square":
        n = int(np.sqrt(n*4/np.pi))
        xy = np.mgrid[-1:1:1j*n, -1:1:1j*n].reshape(2, -1)
        xy = xy[:, (xy**2).sum(0)<=1].T
        return 0, np.concatenate([[[0, 0]], xy])
    elif d == "triangular":
        n = int(np.sqrt(n*4/np.pi))
        xy = np.mgrid[-1:1:1j*n, -1:1:1j*n]
        xy[0] += (np.arange(n) % 2.)*(2./n)
        xy = xy.reshape(2, -1)
        xy = xy[:, (xy**2).sum(0)<=1].T
        return 0, np.concatenate([[[0, 0]], xy])
    elif d == "hexapolar":
        n = int(np.sqrt(n/3.-1/12.)-1/2.)
        l = [np.zeros((2, 1))]
        for i in np.arange(1, n + 1.):
            a = np.linspace(0, 2*np.pi, 6*i, endpoint=False)
            l.append([np.sin(a)*i/n, np.cos(a)*i/n])
        return 0, np.concatenate(l, axis=1).T


@public
def gaussian_roots(n, m=None):
    """returns r[n, 1], phi[1, n],  weights[n, 1]
    gaussian quadrature points (r, phi) with given weights
    covering the [0, pi] semi-unit-circle.
    """
    r, weight = orthogonal.ps_roots(n)
    m = m or n
    phi = np.pi/(1 + m)*np.arange(1, m + 1)
    return r[:, None]**.5, phi[None, :], weight[:, None]/2


@public
def lobatto_roots(n, m=None):
    raise NotImplementedError
    m = m or n
    phi = np.pi/(1 + m)*np.arange(1, m + 1)
    return r[:, None]**.5, phi[None, :], weight[:, None]/2

