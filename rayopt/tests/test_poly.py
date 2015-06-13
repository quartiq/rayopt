# -*- coding: utf8 -*-
#
#   pyrayopt - raytracing for optical imaging systems
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


from rayopt import system_from_yaml, PolyTrace, GeometricTrace


doublet = """
description: 'doublet'
object: {type: finite, slope: .001, pupil_distance: 100.1}
stop: 2
elements:
- {material: 1.}
- {material: 1.51872, distance: 100., curvature: 1.611356421}
- {material: 1.66238, distance: .1, curvature: -2.455396159}
- {material: 1., distance: 0.0661308, curvature: -0.786448792}
- {material: 1., distance: 0.93402287}
"""


class DoubletCase(unittest.TestCase):
    def setUp(self):
        self.s = system_from_yaml(doublet)
        self.s.object.chief_slope = .01
        self.s.update()
    
    def test_poly(self):
        p = PolyTrace(self.s)
        #print(p.stvwo[-1, 0, p.Simplex.i[2, 2, 2]])
        #print(p)
        nptest.assert_allclose(self.s.object.slope, .001)
        nptest.assert_allclose(self.s.object.chief_slope, .01)
        nptest.assert_allclose(self.s.object.pupil_distance, 100.1)
        #nptest.assert_allclose(self.s.object.radius, 1.)
        print("\n".join(p.print_trace("st", [-1], 3)))
        s, t = p.transform()
        print(s[0])
        print(p.evaluate([[1.], [0]], [[0, 1], [0, 0]]))
        nptest.assert_allclose(p.stvwo[-1, 0, :20],
[  5.560e-03,   6.672e-02,  -7.896e-01,  -3.607e-02,   8.647e+00,  -2.132e-01,
   -8.588e+00,   4.489e-02,   1.240e+00,   9.228e-01,   8.649e+01,   1.436e+01,
   -1.081e+02,   3.086e-01,  -9.204e+00,   4.208e+01,  -9.268e-02,   6.684e-01,
   -6.419e-01,  -4.079e+00], atol=0, rtol=1e-3)
        nptest.assert_allclose(p.stvwo[-1, 1, :20],
[  1.010e+00,  -1.602e-02,   9.027e-02,  -8.481e-01,  -2.161e+00,  -3.484e-01,
    9.602e-01,   6.039e-02,  -6.956e-01,   3.486e-01,  -2.187e+01,  -4.773e+00,
    2.988e+01,  -2.475e-01,   1.584e+00,  -1.450e+01,  -2.700e-03,  -1.123e-01,
    1.680e+00,   1.265e+00], atol=0, rtol=1e-3)

