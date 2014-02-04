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

from __future__ import print_function, absolute_import, division

import numpy as np
from scipy import optimize

from .elements import get_element
from .material import fraunhofer


class System(list):
    def __init__(self, elements=None, description="", scale=1e-3,
            wavelengths=None, stop=1, wavelength_weights=None,
            fields=None, field_weights=None, finite=False,
            pupil=None, pupil_distance=None,
            pickups=None, validators=None, solves=None):
        elements = map(get_element, elements or [])
        super(System, self).__init__(elements)
        self.description = description
        self.scale = scale
        self.stop = stop
        self.wavelengths = wavelengths or [fraunhofer[i] for i in "dCF"]
        self.wavelength_weights = wavelength_weights or []
        self.fields = fields or [0.]
        self.field_weights = field_weights or []
        self.pickups = pickups or []
        self.validators = validators or []
        self.solves = solves or []

        # aperture: radius, object na, slope, image na, working fno
        # field: object angle, object radius, image radius
        # conjugate: object distance, image distance, magnification

    def dict(self):
        dat = {}
        # dat["type"] = "system"
        if self.description:
            dat["description"] = self.description
        dat["stop"] = self.stop
        if self.wavelengths:
            dat["wavelengths"] = [float(w) for w in self.wavelengths]
        if self.scale != 1e-3:
            dat["scale"] = float(self.scale)
        if self.pickups:
            dat["pickups"] = [dict(p) for p in self.pickups]
        if self.validators:
            dat["validators"] = [dict(v) for v in self.validators]
        if self.solves:
            dat["solves"] = [dict(s) for s in self.solves]
        if self:
            dat["elements"] = [e.dict() for e in self]
        return dat

    @property
    def object(self):
        return self[0]

    @property
    def aperture(self):
        return self[self.stop]

    @aperture.setter
    def aperture(self, a):
        self.stop = self.index(a)

    @property
    def aperture_index(self):
        return self.stop

    @aperture_index.setter
    def aperture_index(self, i):
        self.stop = int(i)

    @property
    def image(self):
        return self[-1]

    def groups(self):
        """yield lists of element indices that form lens "elements"
        (singlets, multiplets, mirrors)
        """
        group = []
        for i, el in enumerate(self):
            if hasattr(el, "material"):
                if getattr(el.material, "solid", False):
                    group.append(i)
                elif group or getattr(el.material, "mirror", False):
                    group.append(i)
                    yield group
                    group = []
            elif group:
                group.append(i)
        if group:
            yield group

    def get_path(self, path):
        v = self
        for k in path:
            if isinstance(k, str):
                v = getattr(v, k)
            else:
                v = v[k]
        return v

    def set_path(self, path, value):
        v = self
        for k in path[:-1]:
            if isinstance(k, str):
                v = getattr(v, k)
            else:
                v = v[k]
        k = path[-1]
        if isinstance(k, str):
            setattr(v, k, value)
        else:
            v[k] = value

    def pickup(self):
        for pickup in self.pickups:
            value = None
            if "get" in pickup:
                value = self.get_path(pickup["get"])
            if "get_eval" in pickup:
                value = eval(pickup["get_eval"])
            if "get_func" in pickup:
                value = pickup["get_func"](self, pickup, value)
            if "factor" in pickup:
                value *= pickup["factor"]
            if "offset" in pickup:
                value += pickup["offset"]
            if "set" in pickup:
                self.set_path(pickup["set"], value)
            if "set_exec" in pickup:
                exec pickup["set_exec"]

    def solve(self):
        for solve in self.solves:
            if "get" in solve:
                getter = lambda: self.get_path(solve["get"])
            elif "get_eval" in solve:
                def getter():
                    loc = dict(self=self, solve=solve)
                    return eval(solve["get_eval"], loc, globals())
            elif "get_func" in solve:
                getter = lambda: solve["get_func"](self, solve)
            if "set" in solve:
                setter = lambda x: self.set_path(solve["set"], x)
            elif "set_exec" in solve:
                def setter(value):
                    loc = dict(value=value, self=self, solve=solve)
                    exec solve["set_exec"] in loc, globals()
            elif "set_func" in solve:
                setter = lambda x: solve["set_func"](self, solve, x)
            target = solve.get("target", 0.)
            if "init" in solve:
                init = solve["init"]
            elif "set" in solve:
                init = self.get_path(solve["set"])
            else:
                init = 0.
            def func(x):
                setter(x)
                self.pickup()
                return target - getter()
            x = optimize.newton(func, init, tol=solve.get("tol", 1e-8),
                    maxiter=solve.get("maxiter", 20))
            func(x)
            if "init_current" in solve:
                solve["init"] = float(x)

    def update(self):
        self.solve()
        self.pickup()

    def validate(self):
        for validator in self.validators:
            value = None
            if "get" in validator:
                value = self.get_path(validator["get"])
            if "get_eval" in validator:
                value = eval(validator["get_eval"])
            if "get_func" in validator:
                value = validator["get_func"](self, validator, value)
            if "minimum" in validator:
                v = validator["minimum"]
                if value < v:
                    raise ValueError("%s < %s (%s)" % (value, v, validator))
            if "maximum" in validator:
                v = validator["maximum"]
                if value > v:
                    raise ValueError("%s > %s (%s)" % (value, v, validator))
            if "equality" in validator:
                v = validator["equality"]
                if value != v:
                    raise ValueError("%s != %s (%s)" % (value, v, validator))

    def reverse(self):
        # reverse surface order
        self[:] = self[::-1]
        for e in self:
            e.reverse()
        # shift distances forwards
        d = 0.
        for e in self:
            d, e.distance = e.distance, d
        # shift materials backwards
        m = self[0].material
        for e in self[::-1]:
            if hasattr(e, "material"):
                # material is old preceeding material
                m, e.material = e.material, m

    def rescale(self, scale=None):
        if scale is None:
            scale = self.scale/1e-3
        self.scale /= scale
        for e in self:
            e.rescale(scale)

    def __str__(self):
        return "\n".join(self.text())

    def text(self):
        yield u"System: %s" % self.description
        yield u"Scale: %s mm" % (self.scale/1e-3)
        yield u"Wavelengths: %s nm" % ", ".join("%.0f" % (w/1e-9)
                    for w in self.wavelengths)
        yield u"Elements:"
        yield u"%2s %1s %10s %10s %10s %17s %7s %7s %7s" % (
                "#", "T", "Distance", "Rad Curv", "Diameter",
                "Material", "n", "nd", "Vd")
        for i,e in enumerate(self):
            curv = getattr(e, "curvature", 0)
            roc = curv == 0 and np.inf or 1./curv
            mat = getattr(e, "material", "")
            rad = e.radius if e.finite else e.angular_radius
            nd = getattr(mat, "nd", np.nan)
            vd = getattr(mat, "vd", np.nan)
            if mat:
                n = mat.refractive_index(self.wavelengths[0])
            else:
                n = nd
            yield u"%2i %1s %10.5g %10.4g %10.5g %17s %7.3f %7.3f %7.2f" % (
                    i, e.typ, e.distance, roc, rad*2, mat, n, nd, vd)

    def edge_thickness(self, axis=1):
        """list of the edge thicknesses"""
        t = []
        dz0 = 0.
        for el in self:
            try:
                dz = el.edge_sag(axis)
            except AttributeError:
                dz = 0.
            t.append(el.distance - dz + dz0)
            dz0 = dz
        return t

    def resize_convex(self):
        """ensure convex surfaces are at least as large as their
        corresponding closing surface, enabling standard manufacturing"""
        pending = None
        c0 = None
        for el in self[1:-1]:
            if not hasattr(el, "material"):
                continue
            c = getattr(el, "curvature", 0)
            if pending is not None:
                r = max(el.radius, pending.radius)
                if c < 0:
                    el.radius = r
                if c0 > 0:
                    pending.radius = r
                pending = None
                if el.material.solid:
                    pending = el
            if el.material.solid:
                pending, c0 = el, c

    def fix_sizes(self):
        self.resize_convex()

    def surfaces_cut(self, axis=1, points=31):
        """yields cut outlines of surfaces. solids are closed"""
        # FIXME: not really a global cut, but a local one
        pos = np.zeros(3)
        pending = None
        for e in self:
            pos += e.offset
            xyz = pos + e.from_normal(e.surface_cut(axis, points))
            x, z = xyz[:, axis], xyz[:, 2]
            if getattr(e, "material", None) is None:
                yield x, z
                continue
            if pending:
                px, pz = pending
                if x[0] < px[0]: # lower right
                    cl = x[0], pz[0]
                else: # lower left
                    cl = px[0], z[0]
                if x[-1] > px[-1]: # upper right
                    cu = x[-1], pz[-1]
                else: # upper left
                    cu = px[-1], z[-1]
                yield np.c_[(px, pz), cu, (x[::-1], z[::-1]), cl,
                        (px[0], pz[0])]
            elif not e.material.solid or e.material.mirror:
                yield x, z
            if e.material.solid or (pending and e.material.mirror):
                pending = x, z
            else:
                pending = None
        if pending:
            yield pending

    def plot(self, ax, axis=1, npoints=31, adjust=True, **kwargs):
        kwargs.setdefault("color", "black")
        if adjust:
            ax.set_aspect("equal")
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_xticks(())
            ax.set_yticks(())
        for x, z in self.surfaces_cut(axis, npoints):
            ax.plot(z, x, **kwargs)
        o = np.cumsum([e.offset for e in self], axis=0)
        ax.plot(o[:, 2], o[:, axis], ":", **kwargs)

    def paraxial_matrices(self, l, start=1, stop=None):
        n = self[start - 1].refractive_index(l)
        for e in self[start:stop or len(self)]:
            n, m = e.paraxial_matrix(n, l)
            yield n, m

    def paraxial_matrix(self, l, start=1, stop=None):
        m = np.eye(4)
        for n, mi in self.paraxial_matrices(l, start, stop):
            m = np.dot(mi, m)
        return m

    @property
    def origins(self):
        return np.cumsum([el.offset for el in self], axis=0)

    def close(self, index=-1):
        """close of the system such that image is at object using
        element at index"""
        self[index].offset -= self.origins[-1]

    @property
    def track(self):
        return np.cumsum([el.distance for el in self])

    def align(self, n):
        n0 = n[0]
        for i, (el, n) in enumerate(zip(self[:-1], n[:-1])):
            mu = n0/n
            el.align(self[i + 1].direction, mu)
            n0 = n
        self[-1].angles = 0, 0, 0.

    @property
    def mirrored(self):
        return np.cumprod([-1 if getattr(getattr(el, "material", None),
            "mirror", False) else 1 for el in self])
