# -*- coding: utf-8 -*-
#
#   rayopt - raytracing for optical imaging systems
#   Copyright (C) 2012 Robert Jordens <robert@joerdens.org>
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

import itertools

import numpy as np
from scipy.optimize import newton, brentq
from fastcache import clru_cache

from .elements import Element
from .conjugates import Conjugate, FiniteConjugate, InfiniteConjugate
from .material import fraunhofer
from .utils import public
from .cachend import PolarCacheND
from .paraxial_trace import ParaxialTrace
from .pupils import RadiusPupil


@public
class System(list):
    def __init__(self, elements=None, description="", scale=1e-3,
                 wavelengths=None, stop=1, fields=None,
                 object=None, image=None,
                 pickups=None, validators=None, solves=None):
        elements = [Element.make(_) for _ in elements or []]
        super(System, self).__init__(elements)
        self.description = description
        self.scale = scale
        self.wavelengths = wavelengths or [fraunhofer[i] for i in "dCF"]
        self.stop = stop
        if object:
            self.object = Conjugate.make(object)
        else:
            self.object = InfiniteConjugate(
                angle=0., pupil=RadiusPupil(
                    radius=1., update_distance=True, update_radius=True))
        if image:
            self.image = Conjugate.make(image)
        else:
            self.image = FiniteConjugate(
                radius=0., update_radius=True, pupil=RadiusPupil(
                    radius=1., update_distance=True, update_radius=True))
        if fields is None:
            if self.object.point:
                fields = [0.]
            else:
                fields = [0., .7, 1.]
        self.fields = fields
        self.pickups = pickups or []
        self.validators = validators or []
        self.solves = solves or []
        self._pupil_cache = {}
        self.paraxial = ParaxialTrace(self, update=False)

    def dict(self):
        return {
            "description": self.description,
            "stop": self.stop,
            "scale": float(self.scale),
            "wavelengths": [float(w) for w in self.wavelengths],
            "object": self.object.dict(),
            "image": self.image.dict(),
            "pickups": [dict(p) for p in self.pickups],
            "validators": [dict(v) for v in self.validators],
            "solves": [dict(s) for s in self.solves],
            "elements": [e.dict() for e in self],
        }

    @property
    def aperture(self):
        return self[self.stop]

    @aperture.setter
    def aperture(self, a):
        self.stop = self.index(a)

    def groups(self):
        """yield lists of element indices that form lens "elements"
        (singlets, multiplets, mirrors)

        a group is: gas solid+ (mirror solid+)+ gas or a single mirror
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
                value = eval(pickup["get_func"])(self, pickup, value)
            if "factor" in pickup:
                value = value * pickup["factor"]
            if "offset" in pickup:
                value = value + pickup["offset"]
            if "set" in pickup:
                self.set_path(pickup["set"], value)
            if "set_exec" in pickup:
                exec(pickup["set_exec"])

    def solve(self):
        for solve in self.solves:
            if "get" in solve:
                def getter():
                    return self.get_path(solve["get"])
            elif "get_eval" in solve:
                def getter():
                    loc = dict(self=self, solve=solve)
                    return eval(solve["get_eval"], loc, globals())
            elif "get_func" in solve:
                def getter():
                    return solve["get_func"](self, solve)
            if "set" in solve:
                def setter(x):
                    self.set_path(solve["set"], x)
            elif "set_exec" in solve:
                def setter(value):
                    loc = dict(value=value, self=self, solve=solve)
                    exec(solve["set_exec"], globals(), loc)
            elif "set_func" in solve:
                def setter(x):
                    solve["set_func"](self, solve, x)
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
                return getter() - target

            x = newton(func, init, tol=solve.get("tol", 1e-8),
                       maxiter=solve.get("maxiter", 20))
            func(x)
            if "init_current" in solve:
                solve["init"] = float(x)

    def refractive_index(self, wavelength, index):
        for element in self[index::-1]:
            try:
                return element.refractive_index(wavelength)
            except AttributeError:
                pass
        return 1.

    def update(self):
        self._pupil_cache.clear()
        self.pickup()
        self.solve()
        self.object.pupil.refractive_index = \
            self.refractive_index(self.wavelengths[0], 0)
        self.image.pupil.refractive_index = \
            self.refractive_index(self.wavelengths[0], -1)
        self.paraxial.update_conjugates()
        self.paraxial.update()
        self.validate()

    def validate(self, fix=False):
        for validator in self.validators:
            value = None
            if "get" in validator:
                value = self.get_path(validator["get"])
            if "get_eval" in validator:
                value = eval(validator["get_eval"])
            if "get_func" in validator:
                value = eval(validator["get_func"])(self, validator, value)
            if "exec" in validator:
                exec(validator["exec"])
            if "minimum" in validator:
                v = validator["minimum"]
                if value < v:
                    if fix and "get" in validator:
                        self.set_path(validator["get"], v)
                    else:
                        raise ValueError("%s < %s (%s)" %
                                         (value, v, validator))
            if "maximum" in validator:
                v = validator["maximum"]
                if value > v:
                    if fix and "get" in validator:
                        self.set_path(validator["get"], v)
                    else:
                        raise ValueError("%s > %s (%s)" %
                                         (value, v, validator))
            if "equality" in validator:
                v = validator["equality"]
                if value != v:
                    if fix and "get" in validator:
                        self.set_path(validator["get"], v)
                    else:
                        raise ValueError("%s != %s (%s)" %
                                         (value, v, validator))

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
        return itertools.chain(
            self.base_text(), ("",),
            # self.paraxial.text(), ("",)
        )

    def base_text(self):
        yield "System: %s" % self.description
        yield "Scale: %s mm" % (self.scale/1e-3)
        yield "Wavelengths: %s nm" % ", ".join("%.0f" % (w/1e-9)
                                               for w in self.wavelengths)
        yield "Fields: %s" % ", ".join("%g" % _ for _ in self.fields)
        yield "Object:"
        for line in self.object.text():
            yield " " + line
        yield "Image:"
        for line in self.image.text():
            yield " " + line
        yield "Stop: %i" % self.stop
        yield "Elements:"
        yield "%2s %1s %10s %10s %10s %17s %7s %7s %7s" % (
                "#", "T", "Distance", "Rad Curv", "Diameter",
                "Material", "n", "nd", "Vd")
        for i, e in enumerate(self):
            curv = getattr(e, "curvature", 0)
            roc = curv == 0 and np.inf or 1./curv
            rad = e.radius
            mat = getattr(e, "material", "")
            nd = getattr(mat, "nd", np.nan)
            vd = getattr(mat, "vd", np.nan)
            n = nd
            if mat:
                n = self.refractive_index(self.wavelengths[0], i)
            yield "%2i %1s %10.5g %10.4g %10.5g %17s %7.3f %7.3f %7.2f" % (
                    i, e.typeletter, e.distance, roc, rad*2, mat, n, nd, vd)

    def edge_thickness(self, axis=1):
        """list of the edge thicknesses"""
        # FIXME: account for differing radius
        t = []
        dz0 = 0.
        for el in self:
            try:
                dz = el.edge_sag(axis)
            except AttributeError:
                dz = 0.
            t.append(el.distance - dz + dz0)
            dz0 = dz
        return np.array(t)

    @property
    def edge_y(self):
        return np.array(self.edge_thickness(axis=1))

    @property
    def edge_x(self):
        return np.array(self.edge_thickness(axis=0))

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
                if c <= 0:
                    el.radius = r
                if c0 > 0:
                    pending.radius = r
                pending = None
                if not el.material or el.material.solid:
                    pending = el
            if not el.material or el.material.solid:
                pending, c0 = el, c

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
                if x[0] < px[0]:  # lower right
                    cl = x[0], pz[0]
                else:  # lower left
                    cl = px[0], z[0]
                if x[-1] > px[-1]:  # upper right
                    cu = x[-1], pz[-1]
                else:  # upper left
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
        n = self.refractive_index(l, start-1)
        for e in self[start:stop]:
            n, m = e.paraxial_matrix(n, l)
            yield n, m

    def paraxial_matrix(self, l, start=1, stop=None):
        n = 1.
        m = np.eye(4)
        for n, mi in self.paraxial_matrices(l, start, stop):
            m = np.dot(mi, m)
        return n, m

    @property
    def origins(self):
        return np.cumsum([el.offset for el in self], axis=0)

    def close(self, index=-1):
        """close of the system such that image is at object using
        element at index"""
        self[index].offset -= self.origins[-1]

    @property
    def path(self):
        return np.cumsum([el.distance for el in self])

    @property
    def track(self):
        return self.origins[:, 2]

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
                                         "mirror", False) else 1
                           for el in self])

    def propagate_paraxial(self, yu, n, l, start=1, stop=None):
        for e in self[start:stop]:
            yu, n = e.propagate_paraxial(yu, n, l)
            yield yu, n

    def propagate_gaussian(self, q, n, l, start=1, stop=None):
        for e in self[start:stop]:
            q, n = e.propagate_gaussian(q, n, l)
            yield q, n

    def propagate_poly(self, state, l, start=1, stop=None):
        for e in self[start:stop]:
            state = e.propagate_poly(state, l)
            yield state

    def propagate(self, y, u, n, l, start=1, stop=None, clip=False):
        for e in self[start:stop]:
            y, i = e.to_normal(y - e.offset, u)
            y, u, n, t = e.propagate(y, i, n, l, clip)
            yield y, u, n, i, t
            y, u = e.from_normal(y, u)

    def solve_newton(self, merit, a=0., tol=1e-3, maxiter=30):
        def find_start(fun, a0):
            f0 = fun(a0)
            if not np.isnan(f0):
                return a0, f0
            for scale in np.arange(1, maxiter):
                for ai in -scale, scale:
                    fi = fun(a0 + ai)
                    if not np.isnan(fi):
                        return a0 + ai, fi
            raise ValueError("no starting ray found")

        a, f = find_start(merit, a)
        if abs(f) > tol:
            a = newton(merit, a, tol=tol, maxiter=maxiter)
        return a

    def solve_brentq(self, merit, a=0., b=1., tol=1e-3, maxiter=30):
        for i in range(maxiter):
            fb = merit(b)
            if abs(fb) <= tol:
                return b
            elif np.isnan(fb):
                b /= 2
            elif fb < 0:
                a = b
                b *= 1 - fb
            else:
                break
        if i == maxiter - 1:
            raise ValueError("no viable interval found", a, b, fb)
        fa = merit(a)
        if abs(fa) <= tol:
            return a
        assert fa < 0
        a = brentq(merit, a, b, rtol=tol, xtol=tol, maxiter=maxiter)
        return a

    def aim(self, *args, **kwargs):
        return self.object.aim(*args, surface=self[0], **kwargs)

    def aim_chief(self, yo, z, p, l=None, stop=None, **kwargs):
        assert p
        if self.object.pupil.telecentric or not self.object.pupil.aim:
            return z
        if l is None:
            l = self.wavelengths[0]
        n = self.refractive_index(l, 0)
        if stop in (-1, None):
            stop = self.stop
        rad = self[self.stop].radius
        assert rad

        @clru_cache(maxsize=1024)
        def dist(a):
            y, u = self.aim(yo, None, z + a*p, filter=False)
            for yunit in self.propagate(y, u, n, l, stop=stop + 1):
                y = yunit[0]
            return (yo*y[0, :2]).sum()/rad
        a = self.solve_newton(dist, **kwargs)
        return z + a*p

    def aim_marginal(self, yo, yp, z, p, l=None, stop=None, **kwargs):
        assert p
        rim = stop == -1
        if not self.object.pupil.aim and not rim:
            return p
        if l is None:
            l = self.wavelengths[0]
        n = self.refractive_index(l, 0)
        if rim:
            stop = len(self) - 1
        elif stop is None:
            stop = self.stop + 1
        r2 = np.square([e.radius for e in self[1:stop]])

        @clru_cache(maxsize=1024)
        def dist(a):
            y, u = self.aim(yo, yp, z, a*p, filter=False)
            ys = [y]
            for yunit in self.propagate(y, u, n, l, stop=stop):
                ys.append(yunit[0])
            d = np.square(ys)[1:, 0, :2].sum(1)/r2 - 1
            if rim:
                return d.max()
            else:
                return d[-1]
        a = self.solve_brentq(dist, **kwargs)
        assert a
        return a*p

    def _aim_pupil(self, xo, yo, guess, **kwargs):
        y = np.array((xo, yo))
        if guess is None:
            if not self.object.finite and self.object.wideangle and False:
                # FIXME: wideangle!
                z = self.object.entrance_distance
            else:
                z = self.object.pupil.distance
            a = self.object.pupil.radius
            a = a*np.ones((2, 2))
        else:
            z, a = guess[0], guess[1:].reshape(2, 2)
        if not np.allclose(y, 0):
            z1 = self.aim_chief(y, z, np.fabs(a).max(), **kwargs)
            if self.object.finite:
                a *= np.fabs(z1/z)  # improve guess
            z = z1
        for ax, sig in (1, 1), (1, 0), (0, 1), (0, 0):
            yp = [0, 0]
            yp[ax] = 2*sig - 1.
            a1 = self.aim_marginal(y, yp, z, a[sig, ax], **kwargs)
            a[sig, ax] = a1
            if sig == 1:  # and guess is None
                a[0, ax] = -a[1, ax]
            if (sig, ax) == (1, 1) and guess is None:
                a[:, 0] = a[:, 1]
        return np.r_[z, a.flat]

    def pupil(self, yo, l=None, stop=None, **kwargs):
        k = l, stop
        try:
            c = self._pupil_cache[k]
        except KeyError:
            c = self._pupil_cache[k] = PolarCacheND(self._aim_pupil,
                                                    l=l, stop=stop, **kwargs)
        q = c(*yo)
        return q[0], q[1:].reshape(2, 2)
