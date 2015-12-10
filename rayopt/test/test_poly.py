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

from numpy import testing as nptest


from rayopt import system_from_yaml, PolyTrace


doublet = """
description: "test doublet"
object:
  type: finite
  radius: 1.0
  pupil:
    type: slope
    slope: .001
    distance: 100.
    update_distance: False
elements:
- {material: vacuum}
- {material: 1.51872, distance: 99.9, curvature: 1.611356421}
- {material: 1.66238, distance: .1, curvature: -2.455396159}
- {material: vacuum, distance: 0.0661308, curvature: -0.786448792}
- {distance: 0.93402287}
"""


class DoubletCase(unittest.TestCase):
    def setUp(self):
        self.s = system_from_yaml(doublet)

    def test_poly(self):
        p = PolyTrace(self.s, 5)
        nptest.assert_allclose(self.s.object.pupil.slope, .001)
        nptest.assert_allclose(self.s.object.slope, .01)
        nptest.assert_allclose(self.s.object.pupil.distance, 100.)
        nptest.assert_allclose(self.s.object.pupil.radius, .1)
        nptest.assert_allclose(self.s.object.radius, 1.)
        str(p)
        # print("\n".join(p.print_trace("st")))
        s, t = p.st()
        # print(s[0])
        # print(p.evaluate([[1.], [0]], [[0, 1], [0, 0]]))
        nptest.assert_allclose(p.stvwof[-1, 0, :20], [
          5.560e-03,   6.672e-02,  -7.896e-01,  -3.607e-02,
          8.647e+00,  -2.132e-01,  -8.588e+00,   4.489e-02,
          1.240e+00,   9.228e-01,   8.649e+01,   1.436e+01,
         -1.081e+02,   3.086e-01,  -9.204e+00,   4.208e+01,
         -9.268e-02,   6.684e-01,  -6.419e-01,  -4.079e+00
        ], atol=0, rtol=1e-3)
        nptest.assert_allclose(p.stvwof[-1, 1, :20], [
          1.010e+00,  -1.602e-02,   9.027e-02,  -8.481e-01,
         -2.161e+00,  -3.484e-01,   9.602e-01,   6.039e-02,
         -6.956e-01,   3.486e-01,  -2.187e+01,  -4.773e+00,
          2.988e+01,  -2.475e-01,   1.584e+00,  -1.450e+01,
         -2.700e-03,  -1.123e-01,   1.680e+00,   1.265e+00
        ], atol=0, rtol=1e-3)
        nptest.assert_allclose(s.base[:17], [
          1.566e-03,   6.604e-05,  -7.887e-06,  -6.033e-06,
          8.539e-05,  -2.043e-08,  -8.564e-06,   4.549e-11,
          1.235e-08,   9.510e-08,   8.520e-06,   1.422e-08,
         -1.067e-06,   3.128e-12,  -9.188e-10,   4.163e-08,
         -9.270e-15
        ], atol=0, rtol=1e-3)
        nptest.assert_allclose(t.base[:17], [
          1.010e-02,  -2.450e-06,   9.027e-08,  -8.463e-06,
         -2.152e-06,  -3.554e-09,   9.600e-08,   6.039e-12,
         -6.932e-10,   3.347e-09,  -2.157e-07,  -4.757e-10,
          2.949e-08,  -2.486e-13,   1.608e-11,  -1.443e-09,
         -2.700e-17
        ], atol=0, rtol=1e-3)
