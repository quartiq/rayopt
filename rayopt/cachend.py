# -*- coding: utf8 -*-
#
#   pyrayopt - raytracing for optical imaging systems
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

from __future__ import print_function, absolute_import, division

import warnings

import numpy as np
from scipy.interpolate import LinearNDInterpolator
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
