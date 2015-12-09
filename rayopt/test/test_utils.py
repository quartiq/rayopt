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

import os
import unittest

import numpy as np
from numpy import testing as nptest

from rayopt.utils import *


class MiscCase(unittest.TestCase):
    def test_trigs(self):
        for j in 0, .1, .9:
            for i in -j, j:
                self.assertAlmostEqual(sinarctan(tanarcsin(i)), i)
                self.assertAlmostEqual(tanarcsin(sinarctan(i)), i)
                self.assertAlmostEqual(-sinarctan(i), sinarctan(-i))
                self.assertAlmostEqual(-tanarcsin(i), tanarcsin(-i))
        for j in 2., 1e3:
            for i in -j, j:
                self.assertAlmostEqual(tanarcsin(sinarctan(i)), i)

    def test_sag_mer(self):
        u = np.array((0, 3., 3.))
        z = np.array((0, 0, 3.))
        s, m = sagittal_meridional(u, z)
        nptest.assert_allclose(s, (1, 0, 0))
        nptest.assert_allclose(m, (0, 2**-.5, -2**-.5))

    def test_zero_sag_mer(self):
        u = np.array((0, 0, 2.))
        s, m = sagittal_meridional(u, u)
        nptest.assert_allclose(s, (1, 0, 0))
        nptest.assert_allclose(m, (0, 1, 0))

    def test_random_sag_mer(self):
        n = 10
        u = np.random.randn(n, 3)
        z = np.array((0, 0, 3.))
        s, m = sagittal_meridional(u, z)
        un = u.copy()
        normalize(un)
        for i in range(n):
            nptest.assert_allclose(np.dot(u[i], s[i]), 0, atol=1e-13)
            nptest.assert_allclose(np.dot(u[i], m[i]), 0, atol=1e-13)
            nptest.assert_allclose(np.dot(s[i], m[i]), 0, atol=1e-13)
            nptest.assert_allclose(np.cross(s[i], m[i]), un[i])

    def test_radau(self):
        i, xy, w = pupil_distribution("radau", 7)
        self.assertEqual(i, 0)
        self.assertEqual(len(xy), 7)
        r = np.hypot(xy[:, 0], xy[:, 1])
        phi = np.arctan(xy[:, 1], xy[:, 0])
        nptest.assert_allclose(np.unique(r),
                [0, .596, .919], atol=1e-3)
        nptest.assert_allclose(np.unique(w),
                [.111, .188/3*2, .256/3*2], atol=1e-3)
