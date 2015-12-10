# -*- coding: utf8 -*-
#
#   rayopt - raytracing for optical imaging systems
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

from __future__ import (absolute_import, print_function,
                        unicode_literals, division)

import unittest

import numpy as np
from numpy import testing as nptest


from rayopt import FiniteConjugate, InfiniteConjugate
from rayopt.utils import tanarcsin, sinarctan


class ConjugatesCase(unittest.TestCase):
    def test_finite(self):
        c = FiniteConjugate(radius=.1,
                            pupil=dict(type="slope", distance=6., slope=2./6))
        self.assertAlmostEqual(c.pupil.na,
                               sinarctan(c.pupil.radius/c.pupil.distance))
        self.some_aims(c)

    def test_infinite(self):
        c = InfiniteConjugate(
            angle=.1, pupil=dict(type="radius", distance=6., radius=2/6.))
        self.some_aims(c)

    def some_aims(self, c):
        y, p = [], []
        for i in 0, 1, -1:
            for j in 0, 1, -1:
                y.extend([(0, i), (i, 0), (0, i), (i, 0)])
                p.extend([(0, j), (0, j), (j, 0), (0, j)])
        for a, b in zip(y, p):
            # print(a, b)
            self.assert_aims(c, a, b)

    def assert_aims(self, c, yo, yp):
        yo, yp = np.broadcast_arrays(*np.atleast_2d(yo, yp))
        y, u = c.aim(yo, yp)
        nptest.assert_allclose(1., np.square(u).sum(-1))
        p = np.arctan2(yo[0, 0], yo[0, 1])
        r = np.array([[np.cos(p), -np.sin(p)], [np.sin(p), np.cos(p)]])
        y1 = np.dot(yp*c.pupil.radius, r)
        # print(yo, yp, y, u, y1)
        self.assert_hits(y, u, c.pupil.distance, y1)

    def assert_hits(self, y, u, z, yp):
        y1 = y[:, :2] + (z - y[:, 2])*tanarcsin(u)
        nptest.assert_allclose(y1, yp, atol=1e-14, rtol=1e-2)
