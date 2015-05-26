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

from __future__ import print_function, absolute_import, division

import itertools

import numpy as np

from .utils import sinarctan, tanarcsin, public
from .raytrace import Trace
from .simplex import make_simplex



@public
class PolyTrace(Trace):
    def __init__(self, system, kmax=7):
        super(PolyTrace, self).__init__(system)
        self.Simplex = make_simplex(3, kmax)
        self.allocate(kmax)
        self.rays()
        self.propagate()

    def allocate(self, kmax):
        super(PolyTrace, self).allocate()
        l = self.system.wavelengths
        n = self.length

    def rays(self):
        pass

    def propagate(self, start=1, stop=None):
        super(PolyTrace, self).propagate()
        init = start - 1
        # FIXME not really round for gen astig...
        yu = np.vstack((self.y[init], self.y[init],
                        self.u[init], self.u[init])).T
        n = self.n[init]
        for j, (yu, n) in enumerate(self.system.propagate_paraxial(
                yu, n, self.l, start, stop)):
            j += start
            self.y[j], self.u[j] = np.vsplit(yu[:, self.axis::2].T, 2)
            self.n[j] = n

    def print_trace(self):
        c = np.c_[self.z, self.y[:, 0], self.u[:, 0],
                self.y[:, 1], self.u[:, 1]]
        return self.print_coeffs(c,
                "track/axial y/axial u/chief y/chief u".split("/"),
                sum=False)

    def __str__(self):
        "\n".join(itertools.chain(
                #self.print_params(), ("",),
                self.print_trace(), ("",),
                ))

    def andersen(S, mdu, pos=0.):
        mp = 1.

        r0, p0, k0 = S(), S(), S()
        r0[1], p0[2], k0[3] = 1, 1, 1

        r, p, k = S(), S(), S()
        r[1], p[2], k[3] = 1, 1, 1
        r[2] = pos**2
        r[3] = 2*pos
        k[2] = pos

        s = S().shift(1)
        t = S().shift(pos)
        v = S()
        w = S().shift(1)

        o = S().shift(1)
        o[2] = 1
        o = pos*o**.5

        for m, d, u in mdu:
