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

from .elements import Element, Object, Image

class System(HasTraits):
    name = Str
    temperature = Float(21.)
    scale = Float(1e-3)
    elements = List(Element)
    object = Property()
    image = Property()
    aperture = Property()
    aperture_index = Property()

    def reverse(self):
        # reverse surface order
        self.elements[1:-1] = self.elements[-2:0:-1]
        # swap i/o radii
        self.object.radius, self.image.radius = (
                self.image.radius, self.object.radius)
        # image origin is old first surface origin
        d, self.image.origin = (
                self.image.origin, self.elements[-2].origin)
        # object material is old last surface material
        m, self.object.material = (
                self.object.material, self.elements[1].material)
        for e in self.elements[1:-1]:
            e.reverse()
            # origin is old preceeding origin
            d, e.origin = e.origin, d
            # material is old preceeding material
            m, e.material = e.material, m

    def _get_aperture(self):
        for e in self.elements:
            if e.typestr == "A":
                return e

    def _get_aperture_index(self):
        for i, e in enumerate(self.elements):
            if e.typestr == "A":
                return i
        raise KeyError

    def _get_object(self):
        e = self.elements[0]
        assert e.typestr == "O"
        return e

    def _set_object(self, e):
        assert e.typestr == "O"
        if self.elements and self.elements[0].typestr == "O":
            del self.elements[0]
        self.elements.insert(0, e)

    def _get_image(self):
        e = self.elements[-1]
        assert e.typestr == "I"
        return e

    def _set_image(self, e):
        assert e.typestr == "I"
        if self.elements and self.elements[-1].typestr == "I":
            del self.elements[-1]
        self.elements.append(e)

    def __str__(self):
        return "\n".join(self.text())

    def text(self):
        yield "System: %s" % self.name
        yield "Scale: %g m" % self.scale
        yield "Temperature: %g C" % self.temperature
        yield "Wavelengths: %s nm" % ", ".join("%.0f" % (w/1e-9)
                    for w in self.object.wavelengths)
        yield "Surfaces:"
        yield "%2s %1s %12s %12s %10s %15s %5s %5s" % (
                "#", "T", "Distance", "ROC", "Diameter", 
                "Material", "N", "V")
        for i,e in enumerate(self.elements):
            curv = getattr(e, "curvature", 0)
            roc = curv == 0 and np.inf or 1/curv
            mat = getattr(e, "material", None)
            n = getattr(mat, "nd", np.nan)
            v = getattr(mat, "vd", np.nan)
            yield "%2i %1s %12.7g %12.6g %10.5g %15s %5.3f %5.2f" % (
                    i, e.typestr, e.origin[2], roc,
                    e.radius*2, mat or "", n, v)

    def surfaces(self, axis, n=20):
        p = [0, 0, 0]
        l = None
        for e in self.elements:
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

    def paraxial_matrices(self, l):
        n0 = self.object.material.refractive_index(l)
        for e in self.elements:
            n0, m = e.paraxial_matrix(l, n0)
            yield m

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
