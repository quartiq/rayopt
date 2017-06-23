# -*- coding: utf-8 -*-
#
#   rayopt - raytracing for optical imaging systems
#   Copyright (C) 2013 Robert Jordens <robert@joerdens.org>
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU Lesser General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import (absolute_import, print_function,
                        unicode_literals, division)

import unittest

import numpy as np
from numpy import testing as nptest

from rayopt.cachend import LinearCacheND


class CacheCase(unittest.TestCase):
    def test_make(self):
        LinearCacheND(lambda a: a)

    def test_random(self):
        n = 10

        def solver(a, b, guess):
            if guess is not None:
                nptest.assert_allclose((a, b), guess)
            return a, b
        c = LinearCacheND(solver)
        for x in np.random.randn(n, 2):
            nptest.assert_equal(c(*x), x)
