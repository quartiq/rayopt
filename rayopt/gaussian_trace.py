# -*- coding: utf-8 -*-
#
#   rayopt - raytracing for optical imaging systems
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

from __future__ import (absolute_import, print_function,
                        unicode_literals, division)

import itertools

import numpy as np

from .utils import public
from .raytrace import Trace


@public
class GaussianTrace(Trace):
    # qi[i] is valid after the ith element perpendicular to/along
    # the excidence direction (assumes aligned()
    def __init__(self, system):
        super(GaussianTrace, self).__init__(system)
        self.allocate()
        self.rays()
        self.propagate()

    def allocate(self):
        super(GaussianTrace, self).allocate()
        self.qi = np.empty((self.length, 2, 2), dtype=np.complex_)
        self.n = np.empty(self.length)

    def make_qi(self, l, n, waist, position=(0, 0.), angle=0.):
        z0 = np.pi*np.array(waist)**2*self.system.scale/l
        z = np.array(position)/n
        qi = 1/(z + 1j*z0)
        qq = np.eye(2)*qi
        ca, sa = np.cos(angle), np.sin(angle)
        a = np.array([[ca, -sa], [sa, ca]])
        qq = np.dot(a.T, np.dot(qq, a))
        return qq

    def rays(self, qi=None, l=None):
        # 1/q = 1/R - i*lambda/(pi*n*w**2)
        # q = z + i*z0
        # z0 = pi*n*w0**2/lambda
        if l is None:
            l = self.system.wavelengths[0]
        n = self.system.refractive_index(l, 0)
        if qi is None:
            obj = self.system.object
            if obj.finite:
                qi = self.make_qi(l, n, obj.radius)
            else:
                qi = self.make_qi(l, n, obj.pupil.radius,
                                  (-obj.pupil.distance,
                                   -obj.pupil.distance))
        assert np.allclose(qi.T, qi), qi
        self.wavelength = l
        self.n[0] = n
        self.qi[0] = qi

    def propagate(self, start=1, stop=None):
        super(GaussianTrace, self).propagate()
        init = start - 1
        qi, n = self.qi[init], self.n[init]
        for j, (qi, n) in enumerate(self.system.propagate_gaussian(
                qi, n, self.wavelength, start, stop)):
            j += start
            self.qi[j], self.n[j] = qi, n

    def qin_at(self, z=None):
        if z is None:
            return self.qi, self.n
        else:
            # qi[i] is valid after element i
            # element indices for z[i-1] <= z < z[i]
            # returns the qi right after element i if z == z[i]
            i = np.searchsorted(self.path, z) - 1
            i = np.where(i < 0, 0, i)
            qi = self.qi[i, :]
            ni = self.n[i, ]
            dz = (z - self.path[i, ])/ni
            # have to do full freespace propagation here
            # simple astigmatic have just q = q0 + dz
            qixx, qixy, qiyy = qi[:, 0, 0], qi[:, 0, 1], qi[:, 1, 1]
            qixy2 = qixy**2
            n = 1/((1 + dz*qixx)*(1 + dz*qiyy) - dz**2*qixy2)
            qi1 = np.empty_like(qi)
            qi1[:, 0, 0] = n*(qixx*(1 + dz*qiyy) - dz*qixy2)
            qi1[:, 1, 0] = qi1[:, 0, 1] = n*qixy
            qi1[:, 1, 1] = n*(qiyy*(1 + dz*qixx) - dz*qixy2)
            return qi1, ni

    def angle(self, qi):
        qixx, qixy, qiyy = qi[:, 0, 0], qi[:, 0, 1], qi[:, 1, 1]
        if np.iscomplexobj(qi):
            a = np.arctan(2*qixy/(qixx - qiyy))/2
            # a = np.where(np.isnan(a), 0, a)
        else:
            a = np.arctan2(2*qixy, qixx - qiyy)/2
        a = (a + np.pi/4) % (np.pi/2) - np.pi/4
        return a

    def normal(self, qi):
        a = self.angle(qi)
        ca, sa = np.cos(a), np.sin(a)
        o = np.array([[ca, -sa], [sa, ca]])
        # qi = np.where(np.isnan(qi), 0, qi)
        qi = np.einsum("jki,ikl,lmi->ijm", o, qi, o)
        assert np.allclose(qi[:, 0, 1], 0), qi
        assert np.allclose(qi[:, 1, 0], 0), qi
        return np.diagonal(qi, 0, 1, 2), a

    def spot_radius_at(self, z=None, normal=False):
        qi, n = self.qin_at(z)
        c = self.wavelength/(self.system.scale*np.pi)
        if normal:
            r, a = self.normal(-qi.imag)
            return np.sqrt(c/r), a
        else:
            r = np.diagonal(-qi.imag, 0, 1, 2)
            return np.sqrt(c/r)

    def curvature_radius_at(self, z=None, normal=False):
        qi, n = self.qin_at(z)
        c = n[:, None]
        if normal:
            r, a = self.normal(qi.real)
            return c/r, a
        else:
            r = np.diagonal(qi.real, 0, 1, 2)
            return c/r

    @property
    def curvature_radius(self):  # on element
        return self.curvature_radius_at()

    @property
    def spot_radius(self):  # on element
        return self.spot_radius_at()

    @property
    def waist_position(self):  # after element relative to element
        w = -(1/np.diagonal(self.qi, 0, 1, 2)).real*self.n[:, None]
        return w

    @property
    def rayleigh_range(self):  # after element
        z = (1/np.diagonal(self.qi, 0, 1, 2)).imag*self.n[:, None]
        return z

    @property
    def waist_radius(self):  # after element
        n = self.n[:, None]
        r = self.rayleigh_range/np.pi/n*self.wavelength/self.system.scale
        return r**.5

    @property
    def diverging(self):
        return self.curvature_radius > 0

    @property
    def confined(self):
        return self.rayleigh_range > 0

    @property
    def intensity_max(self, lambd):
        return (2/np.pi)**.5/self.waist_radius

    def is_stigmatic(self, m):
        return np.allclose(m[::2, ::2], m[1::2, 1::2])

    def is_simple_astigmatic(self, m):
        # does not mix the two axes
        return np.allclose(m[(0, 0, 1, 1, 2, 2, 3, 3),
                             (1, 3, 0, 2, 1, 3, 0, 2)], 0)

    @property
    def eigenmodes(self):
        n, m = self.system.paraxial_matrix(self.wavelength)
        # FIXME only know how to do this for simple astigmatic matrices
        # otherwise, solve qi*b*qi + qi*a - d*qi - c = 0
        assert self.is_simple_astigmatic(m)
        q = []
        for axis in (0, 1):
            a, b, c, d = m[axis::2, axis::2].flat
            q.append(np.roots((c, d - a, -b)))
        q = np.eye(2)[None, :]/np.array(q).T[:, :, None]  # mode, axis
        return q

    def is_proper(self):  # Nemes checks
        n, m = self.system.paraxial_matrix(self.wavelength)
        a, b, c, d = m[:2, :2], m[:2, 2:], m[2:, :2], m[2:, 2:]
        for i, (v1, v2) in enumerate([
                (np.dot(a, d.T) - np.dot(b, c.T), np.eye(2)),
                (np.dot(a, b.T), np.dot(b, a.T)),
                (np.dot(c, d.T), np.dot(d, c.T)),
                ]):
            assert np.allclose(v1, v2), (i, v1, v2)

    @property
    def m(self):
        n, m = self.system.paraxial_matrix(self.wavelength)
        assert self.is_simple_astigmatic(m)
        a0, a1, d0, d1 = np.diag(m)
        m = np.array([a0 + d0, a1 + d1])/2
        return m

    @property
    def eigenvalues(self):
        m = self.m
        m1 = (m**2 - 1+0j)**.5
        return m + m1, m - m1

    @property
    def real(self):
        return (self.m**2).imag == 0

    @property
    def stable(self):
        return (self.m**2).real < 1

    # TODO: sagittal, meridional, angled, make_complete

    def print_trace(self):
        # c, rc = self.curvature_radius_at(z=None, normal=True)
        s, rs = self.spot_radius_at(z=None, normal=True)
        sa, sb = s.T
        wpx, wpy = self.waist_position.T  # assumes simple astig
        wrx, wry = self.waist_radius.T  # assumes simple astig
        c = np.c_[self.path, sa, sb, np.rad2deg(rs), wpx, wpy, wrx, wry]
        return self.print_coeffs(
            c, "path/spot a/spot b/spot ang/waistx dz/waisty dz/"
            "waist x/waist y".split("/"), sum=False)

    def __str__(self):
        t = itertools.chain(
                self.print_trace(), ("",),
                )
        return "\n".join(t)

    def resize(self, waists=3):
        w, a = self.spot_radius_at(z=None, normal=True)
        for e, y in zip(self.system[1:], w.max(1)[1:]):
            e.radius = y*waists

    def refocus(self, axis=1):
        self.system[-1].distance += self.waist_position[-1, axis]
        self.system.update()
        self.propagate()

    def plot(self, ax, axis=1, npoints=5001, waist=True, scale=10, **kwargs):
        kwargs.setdefault("color", "red")
        z = np.linspace(self.path[0], self.path[-1], npoints)
        i = np.searchsorted(self.path, z) - 1
        m = self.mirrored[i, ]
        wx, wy = self.spot_radius_at(z).T*scale*m
        y = np.array([
            [wx, wx, z], [wy, wy, z],
            [-wx, -wx, z], [-wy, -wy, z],
            ]).transpose(2, 0, 1)
        y = self.from_axis(y)
        for i, ci in zip((axis, 0 if axis else 1), ("-", "--")):
            ax.plot(y[:, i::2, 2], y[:, i::2, axis], ci, **kwargs)
        if waist:
            p = self.waist_position.T
            w = self.waist_radius.T*scale
            r = self.rayleigh_range.T
            for i, ci in zip((axis, 0 if axis else 1), ("-", "--")):
                for j, (el, oi) in enumerate(zip(self.system[1:],
                                                 self.origins[1:])):
                    for z, h, cj in [(0, w[i, j], ci),
                                     (r[i, j], 2**.5*w[i, j], ":"),
                                     (-r[i, j], 2**.5*w[i, j], ":"),
                                     ]:
                        v = p[i, j] + z - el.distance
                        if v >= -el.distance and v <= 0:
                            y = np.array([[h, h, v], [-h, -h, v]])
                            y = el.from_axis(y) + oi
                            ax.plot(y[:, 2], y[:, axis], cj, **kwargs)
