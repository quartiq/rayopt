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

from __future__ import (absolute_import, print_function,
        unicode_literals, division)

import os
import unittest

from scipy import constants as ct
import numpy as np
from numpy import testing as nptest


from rayopt import FiniteConjugate, InfiniteConjugate
from rayopt.utils import tanarcsin, sinarctan


class ConjugatesCase(unittest.TestCase):
    def test_finite(self):
        c = FiniteConjugate(entrance_distance=1e6,
                entrance_radius=2., pupil_distance=2e6,
                radius=.5)
        self.assertEqual(c.pupil_radius,
                c.entrance_radius/c.entrance_distance*c.pupil_distance)
        self.assertEqual(c.na,
                sinarctan(c.entrance_radius/c.entrance_distance))
        self.some_aims(c)

    def test_infinite(self):
        c = InfiniteConjugate(entrance_distance=30.,
                entrance_radius=2., pupil_distance=40.,
                angle=.2)
        self.assertEqual(c.pupil_radius, c.entrance_radius)
        self.some_aims(c)

    def some_aims(self, c):
        y, p = [], []
        for i in 0, 1, -1:
            for j in 0, 1, -1:
                y.extend([(0, i), (i, 0), (0, i), (i, 0)])
                p.extend([(0, j), (0, j), (j, 0), (0, j)])
        for a, b in zip(y, p):
            print(a, b)
            self.assert_aims(c, a, b)

    def assert_hits(self, y, u, z, yp):
        y1 = y[:, :2] + (z - y[:, 2])*tanarcsin(u)
        nptest.assert_allclose(y1, yp, atol=1e-15)

    def assert_aims(self, c, yo, yp):
        yo, yp = np.broadcast_arrays(*np.atleast_2d(yo, yp))
        y, u = c.aim(yo, yp)
        nptest.assert_allclose(1., np.square(u).sum(-1))
        self.assert_hits(y, u, c.pupil_distance,
                yp*c.pupil_radius)
