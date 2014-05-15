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

import numpy as np
import sys


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
