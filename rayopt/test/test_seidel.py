# -*- coding: utf-8 -*-
#
#   rayopt - raytracing for optical imaging systems
#   Copyright (C) 2015 Robert Jordens <jordens@phys.ethz.ch>
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

import rayopt as ro


parabolic = """
object:
  type: infinite
  angle_deg: 1
  pupil:
    radius: 1
    distance: 1
stop: 1
elements:
- {material: vacuum}
- {material: mirror, distance: 1, roc: -200, conic: -1}
- {material: vacuum, distance: -100}
"""


class ParabolicCase(unittest.TestCase):
    def setUp(self):
        self.s = ro.system_from_yaml(parabolic)
        self.s.update()

    def test_zero_spherical(self):
        nptest.assert_allclose(self.s.paraxial.transverse3[1, 0], 0)

    def test_hyperbolic(self):
        self.s[1].conic = 0
        self.s.update()
        sph = self.s.paraxial.transverse3[1, 0]
        self.s[1].conic = -2
        self.s.update()
        hyp = self.s.paraxial.transverse3[1, 0]
        nptest.assert_allclose(sph, -hyp)
