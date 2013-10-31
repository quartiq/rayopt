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


from rayopt import system_from_text, ParaxialTrace, GeometricTrace


class DemotripCase(unittest.TestCase):
    def setUp(self):
        description = "oslo cooke triplet example 50mm f/4 20deg"
        columns = "type roc distance radius material"
        text = """
O 0       0     .364 AIR
S 21.25   5     6.5  SK16
S -158.65 2     6.5  AIR
S -20.25  6     5    F4
S 19.3    1     5    AIR
A 0       0     4.75 AIR
S 141.25  6     6.5  SK16
S -17.285 2     6.5  AIR
I 0       42.95 .364 AIR
"""
        self.s = system_from_text(text, columns.split(),
            description=description)

    def test_from_text(self):
        self.assertEqual(len(self.s), 9)
        self.assertFalse(self.s[0].finite)
        for i, el in enumerate(self.s):
            self.assertGreater(el.radius, 0)
            if i not in (0, 5):
                self.assertGreater(el.distance, 0)
            if i not in (0, 5, 8):
                self.assertGreater(abs(el.curvature), 0)
            if i not in (5,):
                self.assertIsNot(el.material, None)

    def test_system(self):
        s = self.s
        self.assertEqual(len(str(s).splitlines()), 14)
        o, a, i = s[0], s[5], s[-1]
        self.assertIs(s.object, o)
        self.assertIs(s.aperture, a)
        self.assertIs(s.image, i)
        s.object = o
        s.image = i
        self.assertIs(s.object, o)
        self.assertIs(s.aperture, a)
        self.assertIs(s.image, i)
        self.assertEqual(len(self.s), 9)

    def test_reverse(self):
        s = self.s
        s.reverse()
        s.reverse()
        self.test_from_text()
        self.test_system()

    def test_rescale(self):
        l = [el.distance for el in self.s]
        self.s.rescale(123)
        nptest.assert_allclose([el.distance/123 for el in self.s], l)
        self.s.rescale()
        nptest.assert_allclose([el.distance for el in self.s], l)

    def test_funcs(self):
        self.s.fix_sizes()
        list(self.s.surfaces_cut(axis=1, points=11))
        self.s.paraxial_matrices(self.s.wavelengths[0], start=1, stop=None)
        self.s.paraxial_matrix(self.s.wavelengths[0], start=1, stop=None)
        self.s.track
        self.s.origins
        self.s.mirrored
        self.s.align(np.ones_like(self.s.track))

    def test_paraxial(self):
        p = ParaxialTrace(self.s)
        print(unicode(p).encode("ascii", errors="replace"))
        unicode(p)

    def traces(self):
        p = ParaxialTrace(self.s)
        g = GeometricTrace(self.s)
        return p, g
    
    def test_aim(self):
        p, g = self.traces()
        z = p.pupil_distance[0] + p.z[1]
        a = np.arctan2(p.pupil_height[0], z)
        print(z, a)
        z, a = g.aim_pupil(1., z, a)
        print(z, a)
    
    def test_aim_point(self):
        p, g = self.traces()
        g.rays_paraxial_clipping(p)
        g.rays_paraxial_point(p)
        g.rays_paraxial_line(p)

    def test_aim_point(self):
        p, g = self.traces()
        i = self.s.aperture_index
        r = np.array([el.radius for el in self.s[1:-1]])
        g.rays_paraxial_clipping(p)
        nptest.assert_allclose(g.y[i, 0, 1], 0, atol=1e-7)
        nptest.assert_allclose(max(g.y[1:-1, 2, 1] - r), 0, atol=1e-7)
        nptest.assert_allclose(min(g.y[1:-1, 1, 1] + r), 0, atol=1e-7)
        g.rays_paraxial_point(p)
        g.rays_paraxial_line(p)


