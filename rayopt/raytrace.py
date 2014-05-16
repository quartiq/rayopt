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

"""
Raytracing like Spencer and Murty 1962, J Opt Soc Am 52, 6
with some improvements
"""

from __future__ import print_function, absolute_import, division

import itertools

import numpy as np
from scipy.optimize import newton
from scipy.interpolate import griddata

from .aberration_orders import aberration_extrinsic
from .elements import Spheroid
from .utils import sinarctan, tanarcsin, simple_cache, public
from .transformations import rotation_matrix


@public
class Trace(object):
    def __init__(self, system):
        self.system = system

    def allocate(self):
        self.length = len(self.system)

    def propagate(self):
        self.z = self.system.track
        self.origins = self.system.origins
        self.mirrored = self.system.mirrored

    def from_axis(self, y, i=None, ref=0):
        y = np.atleast_3d(y) # zi, rayi, xyz
        if i is None:
            i = np.searchsorted(y[:, ref, 2], self.z)
        ys = []
        for j, yi in enumerate(np.vsplit(y, i)):
            if yi.ndim <= 1:
                continue
            j = min(self.length - 1, j)
            zi, ei, oi = self.z[j], self.system[j], self.origins[j]
            yj = yi.reshape(-1, 3)
            yj = oi + ei.from_axis(yj - (0, 0, zi))
            ys.append(yj.reshape(yi.shape))
        ys = np.vstack(ys)
        return ys

    def print_coeffs(self, coeff, labels, sum=True):
        yield (u"%2s %1s" + u"% 10s" * len(labels)) % (
                (u"#", u"T") + tuple(labels))
        fmt = u"%2s %1s" + u"% 10.4g" * len(labels)
        for i, a in enumerate(coeff):
            yield fmt % ((i, self.system[i].type) + tuple(a))
        if sum:
            yield fmt % ((u" ∑", u"") + tuple(coeff.sum(0)))

    def align(self):
        self.system.align(self.n)
        self.propagate()


@public
class ParaxialTrace(Trace):
    # y[i] is ray height after the ith element perpendicular to the
    # excidence direction (assumes excidence and offset of the
    # next element coincide: use el.align())
    # u[i] is tan(u) angle after the ith element: always a slope.
    # a.k.a. "the paraxial u is tan u"
    #
    # calculations assume aplanatic (and not paraxial system
    # oslo also assumes aplanatic). aplanatic is equivalent to the abbe
    # sine condition, magnification is equal to optical input ray sine
    # over optical output sine for all rays:
    # m = n0 sin u0/ (nk sin uk)
    def __init__(self, system, aberration_orders=3, axis=1):
        super(ParaxialTrace, self).__init__(system)
        self.axis = axis
        self.allocate(aberration_orders)
        self.rays()
        self.propagate()
        self.aberrations()

    def allocate(self, k):
        super(ParaxialTrace, self).allocate()
        l = self.system.wavelengths
        self.l = l[0]
        self.lmin = min(l)
        self.lmax = max(l)
        n = self.length
        self.n = np.empty(n)
        self.y = np.empty((n, 2))
        self.u = np.empty((n, 2))
        self.v = np.empty(n)
        self.c = np.empty((n, 2, 2, k, k, k))
        self.d = np.empty_like(self.c)

    def __aim(self):
        ai = self.system.stop
        m = self.system.paraxial_matrix(self.l, stop=ai + 1)
        m = m[self.axis::2, self.axis::2]
        a, b, c, d = m.flat
        r = self.system[ai].radius
        self.system.object.pupil_distance = b/a
        self.system.object.pupil_radius = r/b

    def __rays(self):
        y, u = self.y, self.u
        ai = self.system.stop
        y, u = self.system.object.aim([0, 0], [], )
        self.y[0] = 0
        self.n[0] = self.system[0].refractive_index(self.l)

    def rays(self):
        y, u = self.y, self.u
        ai = self.system.stop
        m = self.system.paraxial_matrix(self.l, stop=ai + 1)
        a, b, c, d = m[self.axis::2, self.axis::2].flat
        #mi = np.linalg.inv(m)
        r = self.system[ai].radius
        if self.system.object.finite:
            c = self.system.object.radius
        else:
            c = -tanarcsin(self.system.object.angle)
            y, u = u, y
        y[0], u[0] = (0, -c), (r/a, a*c/b)
        self.n[0] = self.system[0].refractive_index(self.l)

    def propagate(self, start=1, stop=None):
        super(ParaxialTrace, self).propagate()
        init = start - 1
        # FIXME not really round for gen astig...
        yu = np.vstack((self.y[init], self.y[init],
            self.u[init], self.u[init])).T
        n = self.n[init]
        els = self.system[start:stop or self.length]
        for i, el in enumerate(els):
            i += start
            yu, n = el.propagate_paraxial(yu, n, self.l)
            self.y[i], self.u[i] = np.vsplit(yu[:, self.axis::2].T, 2)
            self.n[i] = n

    def aberrations(self, start=1, stop=None):
        els = self.system[start:stop or self.length]
        self.c[start - 1] = self.v[start - 1] = 0
        for i, el in enumerate(els):
            i += start
            self.v[i] = el.dispersion(self.lmin, self.lmax)
            self.c[i] = el.aberration(self.y[i], self.u[i - 1],
                    self.n[i - 1], self.n[i], self.c.shape[-1])
        self.extrinsic_aberrations()

    def extrinsic_aberrations(self):
        # FIXME: wrong
        self.d[:] = 0
        st = self.system.stop
        t, s = 0, 1
        kmax = self.d.shape[-1]
        r = np.empty_like(self.d)
        for k in range(1, kmax - 1):
            for j in range(k + 1):
                for i in range(k - j + 1):
                    b = (self.c[:, :, :, k - j - i, j, i]
                       + self.d[:, :, :, k - j - i, j, i])
                    b[st-1::-1, s] = -np.cumsum(b[st:0:-1, s], axis=0)
                    b[st+1:, s] = np.cumsum(b[st:-1, s], axis=0)
                    b[st, s] = 0
                    b[1:, t] = np.cumsum(b[:-1, t], axis=0)
                    r[:, t, :, k - j - i, j, i] = b[:, t]
                    r[:, s, :, k - j - i, j, i] = b[:, s]
            for i in range(1, self.length):
                aberration_extrinsic(self.c[i], r[i], self.d[i], k + 1)

    @property
    def seidel3(self):
        c = self.c
        c = np.array([
                -2*c[:, 0, 1, 1, 0, 0], # SA3
                -c[:, 0, 1, 0, 1, 0], # CMA3
                -c[:, 0, 0, 0, 1, 0], # AST3
                c[:, 0, 0, 0, 1, 0] - 2*c[:, 0, 1, 0, 0, 1], # PTZ3
                -2*c[:, 0, 0, 0, 0, 1], # DIS3
                ])
        # transverse image seidel (like oslo)
        return c.T*self.height[1]/2/self.lagrange

    @property
    def seidel5(self):
        c = self.c + self.d
        c = np.array([
                -2*c[:, 0, 1, 2, 0, 0], # MU1
                -1*c[:, 0, 1, 1, 1, 0], # MU3
                -2*c[:, 0, 0, 1, 1, 0] - 2*c[:, 0, 1, 1, 0, 1] # MU5
                    + 2*c[:, 0, 1, 0, 2, 0], # MU6
                2*c[:, 0, 1, 1, 0, 1], # MU5
                # -2*c[:, 0, 0, 1, 0, 1]-c[:, 0, 0, 0, 2, 0]
                # -c[:, 0, 1, 0, 1, 1], # MU7
                # -c[:, 0, 1, 0, 1, 1]-c[:, 0, 0, 0, 2, 0], # MU8
                -2*c[:, 0, 0, 1, 0, 1] - 2*c[:, 0, 0, 0, 2, 0]
                    - 2*c[:, 0, 1, 0, 1, 1], # MU7+MU8
                -1*c[:, 0, 1, 0, 1, 1], # MU9
                -c[:, 0, 0, 0, 1, 1]/2, # (MU10-MU11)/4
                -2*c[:, 0, 1, 0, 0, 2] + c[:, 0, 0, 0, 1, 1]/2,
                # (5*MU11-MU10)/4
                # -2*c[:, 0, 0, 0, 1, 1]-2*c[:, 0, 1, 0, 0, 2], # MU10
                # -2*c[:, 0, 1, 0, 0, 2], # MU11
                -2*c[:, 0, 0, 0, 0, 2], # MU12
                ])
        # transverse image seidel (like oslo)
        return c.T*self.height[1]/2/self.lagrange

    @property
    def track(self):
        """distance from first to last surface"""
        return self.z[-2] - self.z[1]

    @property
    def height(self):
        """object and image ray height"""
        return self.y[(0, -1), 1]
        #self.lagrange/(self.n[-2]*self.u[-2,0])

    @property
    def pupil_distance(self):
        """pupil location relative to first/last surface"""
        return -self.y[(1, -2), 1]/self.u[(0, -2), 1]

    @property
    def pupil_height(self):
        p = self.pupil_distance
        return self.y[(1, -2), 0] + p*self.u[(0, -2), 0]

    @property
    def lagrange(self):
        return self.n[0]*(self.u[0, 0]*self.y[0, 1]
                - self.u[0, 1]*self.y[0, 0])

    @property
    def focal_length(self):
        """signed distance from principal planes to foci (infinite
        conjugates), Malacara1989 p27 2.41, 2.42: F-P"""
        f = self.lagrange/(
                self.u[0, 1]*self.u[-2, 0] -
                self.u[0, 0]*self.u[-2, 1])
        return f/self.n.take((0, -2))*[-1, 1]

    @property
    def focal_distance(self):
        """front/back focal distance relative to first/last surface
        Malacara1989 p27 2.43 2.44, F-V"""
        c = self.n.take((0, -2))*self.focal_length/self.lagrange
        fd = (self.y[(1, -2), 1]*self.u[(-2, 0), 0]
                - self.y[(1, -2), 0]*self.u[(-2, 0), 1])*c
        return fd

    @property
    def principal_distance(self):
        """distance from first/last surface to principal planes
        Malacara1989: P-V"""
        return self.focal_distance - self.focal_length

    @property
    def nodal_distance(self):
        """nodal points relative to first/last surfaces
        Malacara1989, N-V"""
        return self.focal_length[::-1] + self.focal_distance

    @property
    def numerical_aperture(self):
        # we plot u as a slope (tanU)
        # even though it is paraxial (sinu=tanu=u) we must convert here
        return np.fabs(self.n.take((0, -2))*sinarctan(self.u[(0, -2), 0]))

    @property
    def f_number(self):
        return np.fabs(self.focal_length/(2*self.pupil_height))

    @property
    def working_f_number(self):
        na = self.numerical_aperture
        return self.n.take((0, -2))/(2*na)

    @property
    def airy_radius(self):
        na = self.numerical_aperture
        return 1.22*self.l/(2*na)/self.system.scale

    @property
    def rayleigh_range(self):
        r = self.airy_radius
        return np.pi*r**2/self.l*self.system.scale

    @property
    def magnification(self):
        mt = (self.n[0]*self.u[0, 0])/(self.n[-2]*self.u[-2, 0])
        ma = self.u[-2, 1]/self.u[0, 1]
        return np.array([mt, ma])

    @property
    def number_of_points(self):
        """number of resolvable independent diffraction points
        (assuming no aberrations)"""
        return 4*self.lagrange**2/self.l**2

    @property
    def petzval_curvature(self):
        c = [getattr(el, "curvature", 0) for el in self.system]
        n = self.n
        p = c[1:-1]*(n[1:-1] - n[0:-2])/(n[1:-1]*n[0:-2])
        return p.sum()

    @property
    def eigenrays(self):
        e, v = np.linalg.eig(self.system.paraxial_matrix(self.l))
        return e, v

    def print_c3(self):
        return self.print_coeffs(self.seidel3,
                "SA3 CMA3 AST3 PTZ3 DIS3".split())

    def print_h3(self): # TODO
        c3a = self.aberration3*8 # chromatic
        return self.print_coeffs(c3a[(6, 12), :].T, 
                "PLC PTC".split())

    def print_c5(self):
        return self.print_coeffs(self.seidel5,
                "SA5 CMA5 TOBSA5 SOBSA5 TECMA5 SECMA5 AST5 PTZ5 DIS5".split())

    def print_params(self):
        yield "lagrange: %.5g" % self.lagrange
        yield "track length: %.5g" % self.track
        yield "object, image height: %s" % self.height
        yield "front, back focal length: %s" % self.focal_length
        yield "petzval radius: %.5g" % (1/self.petzval_curvature)
        yield "front, back focal distance: %s" % self.focal_distance
        yield "front, back principal distance: %s" % self.principal_distance
        yield "front, back nodal distance: %s" % self.nodal_distance
        yield "entry, exit pupil distance: %s" % self.pupil_distance
        yield "entry, exit pupil height: %s" % self.pupil_height
        yield "front, back numerical aperture: %s" % self.numerical_aperture
        yield "front, back f number: %s" % self.f_number
        yield "front, back working f number: %s" % self.working_f_number
        yield "front, back airy radius: %s" % self.airy_radius
        yield "transverse, angular magnification: %s" % self.magnification

    def print_trace(self):
        c = np.c_[self.z, self.y[:, 0], self.u[:, 0],
                self.y[:, 1], self.u[:, 1]]
        return self.print_coeffs(c,
                "track/axial y/axial u/chief y/chief u".split("/"),
                sum=False)

    def __str__(self):
        t = itertools.chain(
                self.print_params(), ("",),
                self.print_trace(), ("",),
                self.print_c3(), ("",),
                #self.print_h3(), ("",),
                self.print_c5(), ("",),
                )
        return "\n".join(t)

    def plot(self, ax, principals=False, pupils=False, focals=False,
            nodals=False, **kwargs):
        kwargs.setdefault("color", "black")
        # this assumes that the outgoing oa of an element
        # coincides with the incoming of the next, use align()
        y = self.y[:, :, None] * np.ones(3)
        y[:, :, 2] = self.z[:, None]
        y = self.from_axis(y, range(self.length))
        ax.plot(y[:, :, 2], y[:, :, self.axis], **kwargs)
        h = self.system.aperture.radius*1.5
        for p, flag in [
                (self.principal_distance, principals),
                (self.focal_distance, focals),
                (self.nodal_distance, nodals),
                ]:
            if flag:
                for i, pi, zi in zip((1, -1), p,
                        (0, self.system[-1].distance)):
                    y = self.origins[i] + self.system[i].from_axis(
                            np.array([(h, h, pi-zi), (-h, -h, pi-zi)]))
                    ax.plot(y[:, 2], y[:, self.axis], **kwargs)
        if pupils:
            p = self.pupil_distance
            h = self.pupil_height
            for i, hi, pi, zi in zip((1, -1), h, p,
                    (0, self.system[-1].distance)):
                y = np.empty((4, 3))
                y[:, 0] = y[:, 1] = -1.5, 1.5, -1, 1
                y *= hi
                y[:, 2] = pi - zi
                y = self.origins[i] + self.system[i].from_axis(y)
                y = y.reshape(2, 2, 3)
                ax.plot(y[:, :, 2], y[:, :, self.axis], **kwargs)

    def plot_yybar(self, ax, **kwargs):
        kwargs.setdefault("color", "black")
        ax.plot(self.y[:, 0], self.y[:, 1], **kwargs)

    # TODO
    # * introduce aperture at argmax(abs(y_axial)/radius)
    #   or at argmin(abs(u_axial))
    # * setting any of (obj na, entrance radius, ax slope, image na,
    # working fno), (field angle, obj height, image height), (obj dist,
    # img dist, obj pp, img pp, mag) works

    def resize(self):
        for e, y in zip(self.system[1:], self.y[1:]):
            e.radius = np.fabs(y).sum() # axial+chief

    def focal_length_solve(self, f, i=None):
        # TODO only works for last surface
        if i is None:
            i = self.length - 2
        y0, y = self.y[(i-1, i), 0]
        u0, u = self.u[i-1, 0], -self.y[0, 0]/f
        n0, n = self.n[(i-1, i), :]
        c = (n0*u0 - n*u)/(y*(n - n0))
        self.system[i].curvature = c
        self.propagate()

    def refocus(self):
        self.system[-1].distance -= self.y[-1, 0]/self.u[-1, 0]
        self.propagate()


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
        self.l = 1.

    def make_qi(self, l, n, waist, position=(0, 0.), angle=0.):
        z0 = np.pi*n*np.array(waist)**2*self.system.scale/l
        z = np.array(position)
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
        n = self.system[0].refractive_index(l)
        if qi is None:
            obj = self.system.object
            assert obj.finite # otherwise need pupil
            qi = self.make_qi(l, n, obj.radius)
        assert np.allclose(qi.T, qi), qi
        self.l = l
        self.n[0] = n
        self.qi[0] = qi

    def propagate(self, start=1, stop=None):
        super(GaussianTrace, self).propagate()
        init = start - 1
        qi, n = self.qi[init], self.n[init]
        els = self.system[start:stop or self.length]
        for i, el in enumerate(els):
            i += start
            qi, n = el.propagate_gaussian(qi, n, self.l)
            self.qi[i], self.n[i] = qi, n

    def qin_at(self, z=None):
        if z is None:
            return self.qi, self.n
        else:
            # qi[i] is valid after element i
            # element indices for z[i-1] <= z < z[i]
            # returns the qi right after element i if z == z[i]
            i = np.searchsorted(self.z, z) - 1
            i = np.where(i < 0, 0, i)
            dz = z - self.z[i, :]
            qi = self.qi[i, :]
            # have to do full freespace propagation here
            # simple astigmatic have just q = q0 + dz
            qixx, qixy, qiyy = qi[:, 0, 0], qi[:, 0, 1], qi[:, 1, 1]
            qixy2 = qixy**2
            n = 1/((1 + dz*qixx)*(1 + dz*qiyy) - dz**2*qixy2)
            qi1 = np.empty_like(qi)
            qi1[:, 0, 0] = n*(qixx*(1 + dz*qiyy) - dz*qixy2)
            qi1[:, 1, 0] = qi1[:, 0, 1] = n*qixy
            qi1[:, 1, 1] = n*(qiyy*(1 + dz*qixx) - dz*qixy2)
            n = self.n[i, :]
            return qi1, n

    def angle(self, qi):
        qixx, qixy, qiyy = qi[:, 0, 0], qi[:, 0, 1], qi[:, 1, 1]
        if np.iscomplexobj(qi):
            a = np.arctan(2*qixy/(qixx - qiyy))/2
            #a = np.where(np.isnan(a), 0, a)
        else:
            a = np.arctan2(2*qixy, qixx - qiyy)/2
        a = (a + np.pi/4) % (np.pi/2) - np.pi/4
        return a

    def normal(self, qi):
        a = self.angle(qi)
        ca, sa = np.cos(a), np.sin(a)
        o = np.array([[ca, -sa], [sa, ca]])
        #qi = np.where(np.isnan(qi), 0, qi)
        qi = np.einsum("jki,ikl,lmi->ijm", o, qi, o)
        assert np.allclose(qi[:, 0, 1], 0), qi
        assert np.allclose(qi[:, 1, 0], 0), qi
        return np.diagonal(qi, 0, 1, 2), a

    def spot_radius_at(self, z, normal=False):
        qi, n = self.qin_at(z)
        c = self.l/self.system.scale/np.pi/n[:, None]
        if normal:
            r, a = self.normal(-qi.imag)
            r = np.sqrt(c/r)
            return r, a
        else:
            r = np.diagonal(-qi.imag, 0, 1, 2)
            return np.sqrt(c/r)

    def curvature_radius_at(self, z, normal=False):
        qi, n = self.qin_at(z)
        if normal:
            r, a = self.normal(qi.real)
            return 1/r, a
        else:
            r = np.diagonal(qi.real, 0, 1, 2)
            return 1/r

    @property
    def curvature_radius(self): # on element
        return self.curvature_radius_at(z=None)

    @property
    def spot_radius(self): # on element
        return self.spot_radius_at(z=None)

    @property
    def waist_position(self): # after element relative to element
        w = -(1/np.diagonal(self.qi, 0, 1, 2)).real
        return w

    @property
    def rayleigh_range(self): # after element
        z = (1/np.diagonal(self.qi, 0, 1, 2)).imag
        return z

    @property
    def waist_radius(self): # after element
        n = self.n[:, None]
        r = self.rayleigh_range/np.pi/n*self.l/self.system.scale
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
        m = self.system.paraxial_matrix(self.l)
        # FIXME only know how to do this for simple astigmatic matrices
        # otherwise, solve qi*b*qi + qi*a - d*qi - c = 0
        assert self.is_simple_astigmatic(m)
        q = []
        for axis in (0, 1):
            a, b, c, d = m[axis::2, axis::2].flat
            q.append(np.roots((c, d - a, -b)))
        q = np.eye(2)[None, :]/np.array(q).T[:, :, None] # mode, axis
        return q

    def is_proper(self): # Nemes checks
        m = self.system.paraxial_matrix(self.l)
        a, b, c, d = m[:2, :2], m[:2, 2:], m[2:, :2], m[2:, 2:]
        for i, (v1, v2) in enumerate([
                (np.dot(a, d.T) - np.dot(b, c.T), np.eye(2)),
                (np.dot(a, b.T), np.dot(b, a.T)),
                (np.dot(c, d.T), np.dot(d, c.T)),
                ]):
            assert np.allclose(v1, v2), (i, v1, v2)

    @property
    def m(self):
        m = self.system.paraxial_matrix(self.l)
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
    def real(self): # 
        return (self.m**2).imag == 0

    @property
    def stable(self):
        return (self.m**2).real < 1

    # TODO: sagittal, meridional, angled, make_complete

    def print_trace(self):
        #c, rc = self.curvature_radius_at(z=None, normal=True)
        s, rs = self.spot_radius_at(z=None, normal=True)
        sa, sb = s.T
        wpx, wpy = self.waist_position.T # assumes simple astig
        wrx, wry = self.waist_radius.T # assumes simple astig
        c = np.c_[self.z, sa, sb, np.rad2deg(rs), wpx, wpy, wrx, wry]
        return self.print_coeffs(c,
                "track/spot a/spot b/spot ang/waistx dz/waisty dz/"
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
        self.system.image.distance += self.waist_position[-1, axis]
        self.propagate()

    def plot(self, ax, axis=1, npoints=5001, waist=True, scale=10, **kwargs):
        kwargs.setdefault("color", "black")
        z = np.linspace(self.z[0], self.z[-1], npoints)
        i = np.searchsorted(self.z, z) - 1
        m = self.mirrored[i, :]
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


@public
class GeometricTrace(Trace):
    # y[i]: intercept
    # i[i]: incoming/incidence direction
    # u[i]: outgoing/excidence direction
    # all in the normal coordinate system of the ith element (positions
    # relative to the element vertex)
    def allocate(self, nrays):
        super(GeometricTrace, self).allocate()
        self.nrays = nrays
        self.n = np.empty(self.length)
        self.y = np.empty((self.length, nrays, 3))
        self.u = np.empty_like(self.y)
        self.i = np.empty_like(self.y)
        self.l = 1.
        self.t = np.empty((self.length, nrays))

    def rays_given(self, y, u, l=None):
        y, u = np.atleast_2d(y, u)
        y, u = np.broadcast_arrays(y, u)
        n, m = y.shape
        if not hasattr(self, "y") or self.y.shape[0] != n:
            self.allocate(n)
        if l is None:
            l = self.system.wavelengths[0]
        self.l = l
        self.y[0] = 0
        self.y[0, :, :m] = y
        self.u[0] = 0
        self.u[0, :, :m] = u
        if m < 3: # assumes forward rays
            ux, uy = self.u[0, :, 0], self.u[0, :, 1]
            self.u[0, :, 2] = np.sqrt(1 - ux**2 - uy**2)
        self.i[0] = self.u[0]
        self.n[0] = self.system[0].refractive_index(l)
        self.t[0] = 0

    def propagate(self, start=1, stop=None, clip=False):
        super(GeometricTrace, self).propagate()
        init = start - 1
        stop = stop or self.length
        y, u, n, l = self.y[init], self.u[init], self.n[init], self.l
        y, u = self.system[init].from_normal(y, u)
        for i, e in enumerate(self.system[start:stop]):
            i += start
            y, u = e.to_normal(y - e.offset, u)
            self.i[i] = u
            y, u, n, t = e.propagate(y, u, n, l, clip)
            self.y[i], self.u[i], self.n[i], self.t[i] = y, u, n, t
            y, u = e.from_normal(y, u)

    def refocus(self):
        y = self.y[-1, :, :2]
        u = tanarcsin(self.i[-1])
        good = np.all(np.isfinite(u), axis=1)
        y, u = y[good], u[good]
        y, u = (y - y.mean(0)).ravel(), (u - u.mean(0)).ravel()
        # solution of sum((y+tu-sum(y+tu)/n)**2) == min
        t = -np.dot(y, u)/np.dot(u, u)
        self.system[-1].distance += t
        self.propagate()

    def opd(self, chief=0, radius=None, after=-2, image=-1, resample=4):
        t = self.t[:after + 1].sum(0)
        if not self.system.object.finite:
            # input reference sphere is a tilted plane
            # u0 * (y0 - y - t*u) == 0
            tj = np.dot(self.u[0, chief], (self.y[0, chief] - self.y[0]).T)
            t -= tj*self.n[0]
        if radius is None:
            radius = self.z[image] - self.z[after]
        # center sphere on chief image
        ea, ei = self.system[after], self.system[image]
        y = ea.from_normal(self.y[after]) + self.origins[after]
        y = ei.to_normal(y - self.origins[image]) - self.y[image, chief]
        u = ei.to_normal(ea.from_normal(self.u[after]))
        # http://www.sinopt.com/software1/usrguide54/evaluate/raytrace.htm
        # replace u with direction from y to chief image
        #u = -y/np.sqrt(np.square(y).sum(1))[:, None]
        y[:, 2] += radius
        ti = Spheroid(curvature=1./radius).intercept(y, u)
        t += ti*self.n[after]
        t = -(t - t[chief])/(self.l/self.system.scale)
        # positive t rays have a shorter path to ref sphere and 
        # are arriving before chief
        py = y + ti[:, None]*u
        py[:, 2] -= radius
        py -= py[chief]
        x, y, z = py.T
        if resample:
            pyt = np.vstack((x, y, t))
            x, y, t = pyt[:, np.all(np.isfinite(pyt), axis=0)]
            if not t.size:
                raise ValueError("no rays made it through")
            n = resample*self.y.shape[1]**.5
            h = np.fabs((x, y)).max()
            xs, ys = np.mgrid[-1:1:1j*n, -1:1:1j*n]*h
            ts = griddata((x, y), t, (xs, ys))
            x, y, t = xs, ys, ts
        return x, y, t

    def psf(self, chief=0, pad=4, resample=4, **kwargs):
        radius = self.system[-1].distance
        x, y, o = self.opd(chief, resample=resample, radius=radius,
                **kwargs)
        good = np.isfinite(o)
        n = np.count_nonzero(good)
        o = np.where(good, np.exp(-2j*np.pi*o), 0)/n**.5
        if resample:
            # NOTE: resample assumes constant amplitude in exit pupil
            nx, ny = (i*pad for i in o.shape)
            apsf = np.fft.fft2(o, (nx, ny))
            psf = (apsf*apsf.conj()).real/apsf.size
            dx = x[1, 0] - x[0, 0]
            k = 1/(self.l/self.system.scale)
            f = np.fft.fftfreq(nx, dx*k/radius)
            p, q = np.broadcast_arrays(f[:, None], f)
        else:
            raise NotImplementedError
            n = self.y.shape[1]**.5
            radius/2*np.pi*self.l/self.system.scale
            r = 3*x.max()
            p, q = np.mgrid[-1:1:1j*n, -1:1:1j*n]*r
            np.einsum("->", o, x, y)
        return p, q, psf

    def rays_paraxial(self, paraxial):
        y = np.zeros((2, 2))
        y[:, 1] = paraxial.y[0]
        u = np.zeros((2, 2))
        u[:, 1] = sinarctan(paraxial.u[0])
        self.rays_given(y, u)
        self.propagate(clip=False)

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

        y, u = self.system.object.aim(yo, yp, z, p)
        self.rays_given(y, u, l)

        if np.allclose(yp, 0):
            # aim chief and determine pupil distance
            def vary(a):
                z1 = z*a
                y, u = self.system.object.aim(yo, yp, z1, p)
                self.y[0, 0] = y
                self.u[0, 0] = u
                return z1
        else:
            # aim marginal and determine pupil aperture
            p1 = np.array(p) # copies
            def vary(a):
                p1[axis] = p[axis]*a
                y, u = self.system.object.aim(yo, yp, z, p1)
                self.y[0, 0] = y
                self.u[0, 0] = u
                return p1[axis]

        if stop is -1:
            # return clipping ray
            radii = np.array([e.radius for e in self.system[1:-1]])
            target = np.sign(yp[axis])
            @simple_cache
            def distance(a):
                vary(a)
                self.propagate(clip=False)
                res = self.y[1:-1, 0, axis]
                return max(res*target - radii)
        else:
            # return pupil ray
            if stop is None:
                stop = self.system.stop
            target = yp[axis]*self.system[stop].radius
            @simple_cache
            def distance(a):
                vary(a)
                self.propagate(stop=stop + 1, clip=False)
                res = self.y[stop, 0, axis]
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

    def pupil_distance(self, y, u, axis=1):
        # given real (non-axial) chief ray
        return -y[axis]/tanarcsin(u)[axis]

    def pupil_height(self, y, u, d, axis=1):
        # given real marginal ray
        h = y[axis] + tanarcsin(u)[axis]*d
        h = h*np.cos(np.arcsin(u[axis]))
        return h

    def aim_pupil(self, height, pupil_distance, pupil_height,
            l=None, axis=(0, 1), **kwargs):
        yo = (0, height)
        pd = pupil_distance
        ph = np.ones(2)*pupil_height
        if height:
            # can only determine pupil distance if chief is non-axial
            yp = (0, 0.)
            pd = self.aim(yo, yp, pd, ph, l, axis=1)
        for ax in axis:
            yp = [(1., 0), (0, 1.)][ax]
            ph[ax] = self.aim(yo, yp, pd, ph, l, axis=ax)
        return pd, ph

    @staticmethod
    def pupil_distribution(distribution, nrays):
        # TODO apodization
        """returns nrays in normalized aperture coordinates x/meridional
        and y/sagittal according to distribution, all rays are clipped
        to unit circle aperture.
        Returns center ray index, x, y
        
        meridional: equal spacing line
        sagittal: equal spacing line
        cross: meridional-sagittal cross
        tee: meridional (+-) and sagittal (+ only) tee
        random: random within aperture
        square: regular square grid
        triangular: regular triangular grid
        hexapolar: regular hexapolar grid
        """
        d = distribution
        n = nrays
        if n == 1:
            return 0, np.zeros((n, 2))
        elif d == "half-meridional":
            return 0, np.c_[np.zeros(n), np.linspace(0, 1, n)]
        elif d == "meridional":
            n -= n % 2
            return n/2, np.c_[np.zeros(n + 1), np.linspace(-1, 1, n + 1)]
        elif d == "sagittal":
            n -= n % 2
            return n/2, np.c_[np.linspace(-1, 1, n + 1), np.zeros(n + 1)]
        elif d == "cross":
            n -= n % 4
            return n/4, np.concatenate([
                np.c_[np.zeros(n/2 + 1), np.linspace(-1, 1, n/2 + 1)],
                np.c_[np.linspace(-1, 1, n/2 + 1), np.zeros(n/2 + 1)],
                ])
        elif d == "tee":
            n = (n - 2)/3
            return 2*n + 1, np.concatenate([
                np.c_[np.zeros(2*n + 1), np.linspace(-1, 1, 2*n + 1)],
                np.c_[np.linspace(0, 1, n + 1), np.zeros(n + 1)],
                ])
        elif d == "random":
            r, phi = np.random.rand(2, n)
            xy = np.exp(2j*np.pi*phi)*np.sqrt(r)
            xy = np.c_[xy.real, xy.imag]
            return 0, np.concatenate([[[0, 0]], xy])
        elif d == "square":
            n = int(np.sqrt(n*4/np.pi))
            xy = np.mgrid[-1:1:1j*n, -1:1:1j*n].reshape(2, -1)
            xy = xy[:, (xy**2).sum(0)<=1].T
            return 0, np.concatenate([[[0, 0]], xy])
        elif d == "triangular":
            n = int(np.sqrt(n*4/np.pi))
            xy = np.mgrid[-1:1:1j*n, -1:1:1j*n]
            xy[0] += (np.arange(n) % 2.)*(2./n)
            xy = xy.reshape(2, -1)
            xy = xy[:, (xy**2).sum(0)<=1].T
            return 0, np.concatenate([[[0, 0]], xy])
        elif d == "hexapolar":
            n = int(np.sqrt(n/3.-1/12.)-1/2.)
            l = [np.zeros((2, 1))]
            for i in np.arange(1, n + 1.):
                a = np.linspace(0, 2*np.pi, 6*i, endpoint=False)
                l.append([np.sin(a)*i/n, np.cos(a)*i/n])
            return 0, np.concatenate(l, axis=1).T

    def rays_clipping(self, height, pupil_distance, pupil_height,
            wavelength=None, axis=1, clip=False, **kwargs):
        yo = (0, height)
        pd = pupil_distance
        ph = np.ones(2)*pupil_height
        try:
            pd = self.aim(yo, (0, 0), pd, ph, wavelength, axis=1, **kwargs)
        except RuntimeError as e:
            print("chief aim failed", height, e)
        y, u = self.system.object.aim(yo, (0, 0), pd, ph)
        ys, us = [y], [u]
        for t in -1, 1:
            yp = [0, 0]
            yp[axis] = t
            try:
                ph[axis] = self.aim(yo, yp, pd, ph, wavelength, axis=axis, stop=-1)
            except RuntimeError as e:
                print("clipping aim failed", height, t, e)
            y, u = self.system.object.aim(yo, yp, pd, ph)
            ys.append(y)
            us.append(u)
        y, u = np.vstack(ys), np.vstack(us)
        self.rays_given(y, u, wavelength)
        self.propagate(clip=clip)

    def rays_point(self, height, pupil_distance, pupil_height,
            wavelength=None, nrays=11, distribution="meridional",
            clip=False, aim=(0, 1)):
        if aim:
            try:
                pupil_distance, pupil_height = self.aim_pupil(height,
                        pupil_distance, pupil_height, wavelength, axis=aim)
            except RuntimeError as e:
                print("pupil aim failed", height, e)
        icenter, yp = self.pupil_distribution(distribution, nrays)
        # NOTE: will not have same ray density in x and y if pupil is
        # distorted
        y, u = self.system.object.aim((0, height), yp,
                pupil_distance, pupil_height)
        self.rays_given(y, u, wavelength)
        self.propagate(clip=clip)
        return icenter

    def rays_line(self, height, pupil_distance, pupil_height,
            wavelength=None, nrays=21, aim=True, eps=1e-2, clip=False):
        yi = np.c_[np.zeros(nrays), np.linspace(0, height, nrays)]
        y = np.empty((nrays, 3))
        u = np.empty_like(y)
        if aim:
            for i in range(yi.shape[0]):
                try:
                    pdi = self.aim(yi[i], (0, 0), pupil_distance,
                            pupil_height, wavelength, axis=1)
                    y[i] = self.y[0, 0]
                    u[i] = self.u[0, 0]
                except RuntimeError:
                    print("chief aim failed", i)
        e = np.zeros((3, 1, 3)) # pupil
        e[(1, 2), :, (1, 0)] = eps # meridional, sagittal
        if self.system.object.finite:
            y = np.tile(y, (3, 1))
            ph = sinarctan(pupil_height/pupil_distance)
            u = (u + e*ph).reshape(-1, 3)
            u /= np.sqrt(np.square(u).sum(1))[:, None]
        else:
            y = (y + e*pupil_height).reshape(-1, 3)
            u = np.tile(u, (3, 1))
        self.rays_given(y, u, wavelength)
        self.propagate(clip=clip)

    def rays_paraxial_clipping(self, paraxial, height=1.,
            wavelength=None, **kwargs):
        # TODO: refactor rays_paraxial_*
        zp = paraxial.pupil_distance[0] + paraxial.z[1]
        rp = np.arctan2(paraxial.pupil_height[0], zp)
        return self.rays_clipping(height, zp, rp, wavelength, **kwargs)

    def rays_paraxial_point(self, paraxial, height=1.,
            wavelength=None, **kwargs):
        zp = paraxial.pupil_distance[0] + paraxial.z[1]
        rp = np.arctan2(paraxial.pupil_height[0], zp)
        return self.rays_point(height, zp, rp, wavelength, **kwargs)

    def rays_paraxial_line(self, paraxial, height=1.,
            wavelength=None, **kwargs):
        zp = paraxial.pupil_distance[0] + paraxial.z[1]
        rp = np.arctan2(paraxial.pupil_height[0], zp)
        return self.rays_line(height, zp, rp, wavelength, **kwargs)

    def resize(self, fn=lambda a, b: a):
        r = np.hypot(self.y[:, :, 0], self.y[:, :, 1])
        for e, ri in zip(self.system[1:], r[1:]):
            e.radius = fn(ri.max(), e.radius)

    def plot(self, ax, axis=1, **kwargs):
        kwargs.setdefault("color", "green")
        y = np.array([el.from_normal(yi) + oi for el, yi, oi
            in zip(self.system, self.y, self.origins)])
        ax.plot(y[:, :, 2], y[:, :, axis], **kwargs)

    def print_trace(self):
        t = np.cumsum(self.t, axis=0) - self.z[:, None]
        for i in range(self.nrays):
            yield "ray %i" % i
            c = np.concatenate((self.n[:, None], self.z[:, None],
                t[:, i, None], self.y[:, i, :], self.u[:, i, :]), axis=1)
            for _ in self.print_coeffs(c, "n/track z/rel path/"
                    "height x/height y/height z/angle x/angle y/angle z"
                    .split("/"), sum=False):
                yield _
            yield ""

    def __str__(self):
        t = itertools.chain(
                self.print_trace(), ("",),
                )
        return "\n".join(t)

# alias
@public
class FullTrace(GeometricTrace):
    pass
