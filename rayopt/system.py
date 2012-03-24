# -*- coding: utf8 -*-
#
#   pyrayopt - raytracing for optical imaging systems
#   Copyright (C) 2012 Robert Jordens <jordens@phys.ethz.ch>
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

import numpy as np
from traits.api import (HasTraits, Str, Array, Float, Instance, List,
        Property)

from scipy.optimize import (newton, fsolve)

from .elements import Element, Object, Image

class System(HasTraits):
    name = Str
    wavelengths = Array(dtype=np.float64, shape=(None,))
    heights = Array(dtype=np.float64, shape=(None, 2))
    temperature = Float(21.)
    scale = Float(1e-3)
    object = Instance(Object)
    elements = List(Element)
    image = Instance(Image)
    all = Property()

    def revert(self):
        m = self.object.material
        self.object.material = self.elements[-1].material
        for e in self.elements:
            if hasattr(e, "material"):
                m, e.material = e.material, m
        d = self.image.origin
        self.image.origin = self.elements[0].origin
        self.elements.reverse()
        for e in self.elements:
            e.revert()
            d, e.origin = e.origin, d

    def _get_all(self):
        return [self.object] + self.elements + [self.image]

    def __str__(self):
        s = ""
        s += "System: %s\n" % self.name
        s += "Scale: %g m\n" % self.scale
        s += "Temperature: %g C\n" % self.temperature
        s += "Wavelengths: %s nm\n" % ",".join("%.0f" % (w/1e-9)
                    for w in self.wavelengths)
        s += "Surfaces:\n"
        s += "%2s %1s %12s %12s %10s %15s %5s %5s\n" % (
                "#", "T", "Distance to", "ROC", "Diameter", 
                "Material after", "N", "V")
        if self.object:
            dia = (self.object.radius == np.inf and
                self.object.field_angle*2 or self.object.radius*2)
            s += "%-2s %1s %-12s %-12s %10.5g %15s %5.2f %5.2f\n" % (
                "", self.object.typestr, "", "", dia,
                self.object.material,
                self.object.material.nd, self.object.material.vd)
        for i,e in enumerate(self.elements):
            curv = getattr(e, "curvature", 0)
            roc = curv == 0 and np.inf or 1/curv
            mat = getattr(e, "material", None)
            n = getattr(mat, "nd", np.nan)
            v = getattr(mat, "vd", np.nan)
            s += "%-2i %1s %12.7g %12.6g %10.5g %15s %5.2f %5.2f\n" % (
                i+1, e.typestr, e.origin[2], roc, e.radius*2, mat, n, v)
        if self.image:
            s += "%2s %1s %12.7g %-12s %10.5g %15s %-5s %-5s\n" % (
                "", self.image.typestr, self.image.origin[2], "",
                self.image.radius*2, "", "", "")
        return s

    def surfaces(self, axis, n=20):
        p = [0, 0, 0]
        l = None
        for e in [self.object] + self.elements + [self.image]:
            xi, zi = e.surface(axis, n)
            xi += p[axis]
            zi += p[2]
            p += e.origin
            if l is not None:
                if xi[0] < l[0, 0]:
                    cl = ([xi[0]], [l[1, 0]])
                else:
                    cl = ([l[0, 0]], [zi[0]])
                if xi[-1] > l[0, -1]:
                    cu = ([xi[-1]], [l[1, -1]])
                else:
                    cu = ([l[0, -1]], [zi[-1]])
                yield np.c_[l[:, (0,)], cl, (xi, zi), cu, l[:, ::-1]]
            elif not e.material.solid:
                yield xi, zi
            if e.material.solid:
                l = np.array([xi, zi])
            else:
                l = None

    def solve(self):
        pass

    def optimize(self, rays, parameters, demerits, constraints=(),
            method="ralg"):

        def objective_function(x):
            for i,p in enumerate(parameters):
                p.set_value(self, x[i])
            p = self.paraxial_trace()
            r = [self.propagate_through(ir) for ir in rays]
            d = [np.array(de(self, p, r)).reshape((-1,))*de.weight for de in demerits]
            return np.concatenate(d)

        x0 = np.array([p.get_value(self) for p in parameters])
        # bs = 2
        # bounds = [(min(p/bs, p*bs), max(p/bs, p*bs)) for p in x0]
        #from numpy.random import randn
        #x0 *= 1+randn(len(x0))*.1

        eqs = [c for c in constraints if c.equality]
        ineqs = [c for c in constraints if not c.equality]

        def equality_constraints(x):
            return np.concatenate([c(self) for c in eqs])
        def inequality_constraints(x):
            return np.concatenate([c(self) for c in ineqs])

        from openopt import NLP
        problem = NLP(objective_function, x0,
                c=ineqs and inequality_constraints or None,
                h=eqs and equality_constraints or None,
                lb=np.array([p.bounds[0] for p in parameters]),
                ub=np.array([p.bounds[1] for p in parameters]),
                #scale=[p.scale for p in parameters],
                diffInt=[p.scale*1e-2 for p in parameters],
                ftol=1e-10, gtol=1e-10, xtol=1e-14,
                maxCPUTime=2e3, maxNonSuccess=30,
                maxFunEvals=2000, iprint=1, plot=1)
        res = problem.solve(method)
        print res
        x, f = res.xf, res.ff
        for i,p in enumerate(parameters):
             p.set_value(self, x[i])
        return x0,x,f
        


