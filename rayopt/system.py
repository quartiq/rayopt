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

import warnings

import numpy as np
from scipy.optimize import newton

from .elements import Element
from .conjugates import Conjugate, FiniteConjugate, InfiniteConjugate
from .material import fraunhofer
from .utils import public, tanarcsin, sinarctan, simple_cache


@public
class System(list):
    def __init__(self, elements=None,
            description="", scale=1e-3, wavelengths=None,
            stop=None, object=None, image=None,
            pickups=None, validators=None, solves=None):
        elements = map(Element.make, elements or [])
        super(System, self).__init__(elements)
        self.description = description
        self.scale = scale
        self.wavelengths = wavelengths or [fraunhofer[i] for i in "dCF"]
        self.stop = stop
        if object:
            self.object = Conjugate.make(object)
        else:
            self.object = InfiniteConjugate(angle=0.)
        if image:
            self.image = Conjugate.make(image)
        else:
            self.image = FiniteConjugate(radius=0.)
        self.pickups = pickups or []
        self.validators = validators or []
        self.solves = solves or []
        self.update()

    def dict(self):
        dat = {}
        # dat["type"] = "system"
        if self.description:
            dat["description"] = self.description
        if self.stop is not None:
            dat["stop"] = self.stop
        if self.scale != 1e-3:
            dat["scale"] = float(self.scale)
        if self.wavelengths:
            dat["wavelengths"] = [float(w) for w in self.wavelengths]
        if self.object:
            dat["object"] = self.object.dict()
        if self.image:
            dat["image"] = self.image.dict()
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
    def aperture(self):
        return self[self.stop]

    @aperture.setter
    def aperture(self, a):
        self.stop = self.index(a)

    @property
    def aperture_index(self):
        warnings.warn("use system.stop", DeprecationWarning)
        return self.stop

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
            x = newton(func, init, tol=solve.get("tol", 1e-8),
                    maxiter=solve.get("maxiter", 20))
            func(x)
            if "init_current" in solve:
                solve["init"] = float(x)

    def update(self):
        self.solve()
        self.pickup()
        self.object.refractive_index = \
                self[0].refractive_index(self.wavelengths[0])
        self.object.entrance_distance = self[1].distance
        self.object.entrance_radius = self[1].radius
        self.image.refractive_index = \
                self[-2].refractive_index(self.wavelengths[0])
        self.image.entrance_distance = self[-1].distance
        self.image.entrance_radius = self[-2].radius

    def validate(self, fix=False):
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
                    if fix and "get" in validator:
                        self.set_path(validator["get"], v)
                    else:
                        raise ValueError("%s < %s (%s)" % (value, v, validator))
            if "maximum" in validator:
                v = validator["maximum"]
                if value > v:
                    if fix and "get" in validator:
                        self.set_path(validator["get"], v)
                    else:
                        raise ValueError("%s > %s (%s)" % (value, v, validator))
            if "equality" in validator:
                v = validator["equality"]
                if value != v:
                    if fix and "get" in validator:
                        self.set_path(validator["get"], v)
                    else:
                        raise ValueError("%s != %s (%s)" % (value, v, validator))

    def reverse(self):
        # i-1|material_i-1,distance_i|i|material_i,distance_i+1|i+1
        # ->
        # i-1|material_i,distance_i-1|i|material_i+1,distance_i|i+1
        # 
        d = [e.distance for e in self] + [0.]
        m = [None] + [getattr(e, "material", None) for e in self]
        for i, e in enumerate(self):
            e.reverse()
            e.distance = d[i + 1]
            e.material = m[i]
        self.object, self.image = self.image, self.object
        self[:] = reversed(self)

    def rescale(self, scale=None):
        if scale is None:
            scale = self.scale/1e-3
        self.scale /= scale
        for e in self:
            e.rescale(scale)
        self.object.rescale(scale)
        self.image.rescale(scale)

    def __str__(self):
        return "\n".join(self.text())

    def text(self):
        yield u"System: %s" % self.description
        yield u"Scale: %s mm" % (self.scale/1e-3)
        yield u"Wavelengths: %s nm" % ", ".join("%.0f" % (w/1e-9)
                    for w in self.wavelengths)
        yield u"Object:"
        for line in self.object.text():
            yield u" " + line
        yield u"Image:"
        for line in self.image.text():
            yield u" " + line
        yield u"Elements:"
        yield u"%2s %1s %10s %10s %10s %17s %7s %7s %7s" % (
                "#", "T", "Distance", "Rad Curv", "Diameter",
                "Material", "n", "nd", "Vd")
        for i,e in enumerate(self):
            curv = getattr(e, "curvature", 0)
            roc = curv == 0 and np.inf or 1./curv
            mat = getattr(e, "material", "")
            rad = e.radius
            nd = getattr(mat, "nd", np.nan)
            vd = getattr(mat, "vd", np.nan)
            if mat:
                n = mat.refractive_index(self.wavelengths[0])
            else:
                n = nd
            yield u"%2i %1s %10.5g %10.4g %10.5g %17s %7.3f %7.3f %7.2f" % (
                    i, e.typeletter, e.distance, roc, rad*2, mat, n, nd, vd)

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

    def propagate_paraxial(self, yu, n, l, start=1, stop=None):
        stop = stop or len(self)
        for e in self[start:stop]:
            yu, n = e.propagate_paraxial(yu, n, l)
            yield yu, n

    def propagate_gaussian(self, q, n, l, start=1, stop=None):
        stop = stop or len(self)
        for e in self[start:stop]:
            qi, n = e.propagate_gaussian(qi, n, l)
            yield qi, n

    def propagate(self, y, u, n, l, start=1, stop=None, clip=False):
        stop = stop or len(self)
        for e in self[start:stop]:
            y, i = e.to_normal(y - e.offset, u)
            y, u, n, t = e.propagate(y, i, n, l, clip)
            yield y, u, n, i, t
            y, u = e.from_normal(y, u)

    def aim(self, yo, yp, z, p, l=None, axis=1, stop=None,
            tol=1e-3, maxiter=100):
        """aims ray at aperture center (or target)
        changing angle (in case of finite object) or
        position in case of infinite object"""
        # yo 2d fractional object coordinate (object knows meaning)
        # yp 2d fractional angular pupil coordinate (since object points
        # emit into solid angles)
        # z pupil distance from object apex
        # a pupil angular half aperture (from z=0 even in infinite case)

        # get necessary y for finite object and u for infinite
        # get guess u0/y0
        # setup vary functions that change u/y (angle around u0 and pos
        # ortho to u)
        # setup distance function that measures distance to aperture
        # point or max(yi/target - radii) (if stop==-1)
        # find first, then minimize
        # return apparent z and a

        if l is None:
            l = self.wavelengths[0]
        y, u = self.object.aim(yo, yp, z, p)
        n = self[0].refractive_index(l)

        if np.allclose(yp, 0):
            # aim chief and determine pupil distance
            def vary(a):
                z1 = z*a
                y[0], u[0] = self.object.aim(yo, yp, z1, p)
                return z1
        else:
            # aim marginal and determine pupil aperture
            p1 = np.array(p) # copies
            def vary(a):
                p1[axis] = p[axis]*a
                y[0], u[0] = self.object.aim(yo, yp, z, p1)
                return p1[axis]

        if stop is -1:
            # return clipping ray
            radii = np.array([e.radius for e in self[1:-1]])
            target = np.sign(yp[axis])
            @simple_cache
            def distance(a):
                vary(a)
                ys = [yunit[0][0, axis]/target for yunit in self.propagate(
                    y, u, n, l, clip=False, stop=-1)]
                return max(ys - radii)
        else:
            # return pupil ray
            if stop is None:
                stop = self.stop
            target = yp[axis]*self[stop].radius
            @simple_cache
            def distance(a):
                vary(a)
                res = [yunit[0] for yunit in self.propagate(
                    y, u, n, l, stop=stop + 1, clip=False)][-1][0, axis]
                return res - target

        def find_start(fun, a0=1.):
            f0 = fun(a0)
            if not np.isnan(f0):
                return a0, f0
            for scale in np.logspace(-1, 2, 16):
                for ai in -scale, scale:
                    fi = fun(a0 + ai)
                    if not np.isnan(fi):
                        return a0 + ai, fi
            raise RuntimeError("no starting ray found")

        a, f = find_start(distance)
        if abs(f - target) > tol:
            a = newton(distance, a, tol=tol, maxiter=maxiter)
        return vary(a)
