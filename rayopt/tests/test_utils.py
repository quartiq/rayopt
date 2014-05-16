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

    def test_gauss(self):
        r, p, w = gaussian_roots(4)
        nptest.assert_allclose(r,
                np.array([.2635, .5745, .8185, .9647])[:, None],
                atol=1e-4)
        nptest.assert_allclose(p,
                (np.arange(4)[None, :] + 1)*np.pi/5)
        nptest.assert_allclose(w,
                np.array([.087, .163, .163, .087])[:, None],
                atol=1e-4)
