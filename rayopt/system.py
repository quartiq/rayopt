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

from .elements import Object, Image, Aperture


class System(list):
    def __init__(self, elements=[], description="", scale=1e-3):
        if elements is None:
            elements = [Object(), Aperture(), Image()]
        super(System, self).__init__(elements)
        self.description = description
        self.scale = scale

    @property
    def object(self):
        assert isinstance(self[0], Object)
        return self[0]

    @object.setter
    def object(self, value):
        assert isinstance(value, Object)
        if isinstance(self[0], Object):
            self[0] = value
        else:
            self.insert(0, value)

    @property
    def aperture(self):
        for el in self:
            if isinstance(el, Aperture):
                return el

    @property
    def aperture_index(self):
        return self.index(self.aperture)

    @property
    def image(self):
        assert isinstance(self[-1], Image)
        return self[-1]

    @image.setter
    def image(self, value):
        assert isinstance(value, Image)
        if isinstance(self[-1], Image):
            self[-1] = value
        else:
            self.append(value)

    def reverse(self):
        o, i = self.object, self.image
        # swap i/o radii
        o.radius, i.radius = i.radius, o.radius
        # reverse surface order
        self[1:-1] = self[-2:0:-1]
        # shift thicknesses forward
        for e in self:
            e.reverse()
        # shift thicknesses forwards
        d = i.thickness
        for e in self[1:]:
            d, e.thickness = e.thickness, d
        # shift materials backwards
        m = o.material
        for e in self[-2::-1]:
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
        yield "System: %s" % self.description
        yield "Scale: %s mm" % (self.scale/1e-3)
        yield "Wavelengths: %s nm" % ", ".join("%.0f" % (w/1e-9)
                    for w in self.object.wavelengths)
        yield "Surfaces:"
        yield "%2s %1s %10s %10s %10s %10s %5s %5s" % (
                "#", "T", "Thickness", "Rad Curv", "Diameter", 
                "Material", "n", "V")
        for i,e in enumerate(self):
            curv = getattr(e, "curvature", 0)
            roc = curv == 0 and np.inf or 1./curv
            mat = getattr(e, "material", None)
            n = getattr(mat, "nd", np.nan)
            v = getattr(mat, "vd", np.nan)
            yield "%2i %1s %10.5g %10.4g %10.5g %10s %5.3f %5.2f" % (
                    i, e.typ, e.thickness, roc,
                    e.radius*2, mat or "-", n, v)

    def resize_convex(self):
        """ensure convex surfaces are at least as large as their
        corresponding closing surface"""
        pending = None
        c0 = None
        for el in self[1:-1]:
            if not hasattr(el, "material"):
                continue
            c = getattr(el, "curvature", 0)
            if pending is not None:
                if c < 0:
                    el.radius = max(el.radius, pending.radius)
                if c0 > 0:
                    pending.radius = max(pending.radius, el.radius)
                pending = None
                if el.material.solid:
                    pending = el
            if el.material.solid:
                pending, c0 = el, c

    def fix_sizes(self):
        self.resize_convex()

    def surfaces_cut(self, axis, points):
        """yields cut outlines of surfaces. solids are closed"""
        z0 = 0.
        pending = None
        for e in self:
            z0 += e.thickness
            xyz = e.transform_from(e.surface_cut(axis, points))
            x, z = xyz[:, axis], xyz[:, 2]
            z += z0
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
            elif not e.material.solid:
                yield x, z
            if e.material.solid:
                pending = x, z
            else:
                pending = None

    def plot(self, ax, axis=1, npoints=31, adjust=True, **kwargs):
        kwargs.setdefault("linestyle", "-")
        kwargs.setdefault("color", "black")
        if adjust:
            ax.set_aspect("equal")
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            #ax.spines["bottom"].set_position("zero")
            #ax.spines["bottom"].set_smart_bounds(True)
            #ax.xaxis.set_ticks_position("bottom")
            #ax.set_xticklabels(())
            ax.set_xticks(())
            ax.set_yticks(())
        for x, z in self.surfaces_cut(axis, npoints):
            ax.plot(z, x, **kwargs)
        ax.plot((0, sum(e.thickness for e in self)), (0, 0), "k--")

    def paraxial_matrices(self, l, start=0, stop=None, n=None):
        for e in self[start:stop or len(self)]:
            n, m = e.paraxial_matrix(n, l)
            yield n, m

    def paraxial_matrix(self, l, start=0, stop=None, n0=None):
        m = np.eye(2)
        for n, mi in self.paraxial_matrices(l, start, stop):
            m = np.dot(mi, m)
        return m
