# -*- coding: utf-8 -*-
#
#   rayopt - raytracing for optical imaging systems
#   Copyright (C) 2014 Robert Jordens <jordens@phys.ethz.ch>
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

from __future__ import (absolute_import, print_function,
                        unicode_literals, division)

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial.qhull import QhullError

from .utils import public


@public
class CacheND(object):
    def __init__(self, solver, guess=None, **kwargs):
        self.solver = solver
        self.interpolator = None
        self.kwargs = kwargs
        self.cache = {}
        self.clear(guess)

    def clear(self, guess=None):
        self.cache.clear()
        self.guess = None

    def __call__(self, *args):
        try:
            return self.cache[args]
        except KeyError:
            pass
        guess = self.guess
        if self.interpolator:
            guess = self.interpolator(*args)
            if np.any(np.isnan(guess)):
                guess = self.guess
        value = self.solver(*args, guess=guess, **self.kwargs)
        self.cache[args] = value
        self._update()
        return value

    def _update(self):
        raise NotImplementedError


@public
class NearestCacheND(CacheND):
    def _update(self):
        xy = list(self.cache.items())
        x = np.array([_[0] for _ in xy])
        y = np.array([_[1] for _ in xy])
        i = NearestNDInterpolator(x, y)
        self.interpolator = i


@public
class LinearCacheND(CacheND):
    def _update(self):
        if len(self.cache) < 4:
            return
        xy = list(self.cache.items())
        x = np.array([_[0] for _ in xy])
        y = np.array([_[1] for _ in xy])
        try:
            i = LinearNDInterpolator(x, y)
        except QhullError:
            i = None
        self.interpolator = i


@public
class PolarCacheND(CacheND):
    def _update(self):
        xy = list(self.cache.items())
        x = np.array([_[0] for _ in xy])
        y = np.array([_[1] for _ in xy])
        r = np.sqrt(np.square(x).sum(1))
        i = np.argsort(r)
        self.r = r.take(i)
        # self.p = np.arctan2(x[:, 1], x[:, 0]).take(i)
        self.y = y.take(i, axis=0)
        self.interpolator = self._interpolator

    def _interpolator(self, xo, yo):
        r = np.sqrt(xo**2 + yo**2)
        if r <= self.r[0]:
            return self.y[0]
        if r >= self.r[-1]:
            return self.y[-1]
        i = np.searchsorted(self.r, r)
        ra, rb = self.r[i - 1], self.r[i]
        ya, yb = self.y[i - 1], self.y[i]
        return ya + (yb - ya)*(r - ra)/(rb - ra)
