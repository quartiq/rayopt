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
from .utils import sinarctan, tanarcsin


class Trace(object):
    def __init__(self, system):
        self.system = system

    def allocate(self):
        self.length = len(self.system)

    def print_coeffs(self, coeff, labels, sum=True):
        yield (u"%2s %1s" + u"% 10s" * len(labels)) % (
                (u"#", u"T") + tuple(labels))
        fmt = u"%2s %1s" + u"% 10.4g" * len(labels)
        for i, a in enumerate(coeff):
            yield fmt % ((i, self.system[i].typ) + tuple(a))
        if sum:
            yield fmt % ((u" ∑", u"") + tuple(coeff.sum(0)))


class ParaxialTrace(Trace):
    def __init__(self, system, aberration_orders=3):
        super(ParaxialTrace, self).__init__(system)
        self.allocate(aberration_orders)
        self.rays()
        self.propagate()
        self.aberrations()

    def allocate(self, k):
        super(ParaxialTrace, self).allocate()
        l = self.system.object.wavelengths
        self.l = l[0]
        self.lmin = min(l)
        self.lmax = max(l)
        n = self.length
        self.y = np.empty((n, 2))
        self.u = np.empty((n, 2))
        self.v = np.empty(n)
        self.n = np.empty(n)
        self.c = np.empty((n, 2, 2, k, k, k))
        self.d = np.empty_like(self.c)

    def rays(self):
        y, u = self.y, self.u
        l = self.system.object.wavelengths[0]
        ai = self.system.aperture_index
        m = self.system.paraxial_matrix(l, stop=ai + 1)
        mi = np.linalg.inv(m)
        r = self.system[ai].radius
        c = self.system.object.radius
        if not self.system.object.infinite:
            y, u, mi, c, r = u, y, -mi[::-1], -c, -r
        y[0, 0], u[0, 0] = r*mi[0, 0] - r*mi[0, 1]*mi[1, 0]/mi[1, 1], 0
        y[0, 1], u[0, 1] = c*mi[0, 1]/mi[1, 1], c

    def propagate(self, start=0, stop=None):
        self.z = np.cumsum([e.thickness for e in self.system])
        init = start - 1 if start else 0
        yu, n = np.array((self.y[init], self.u[init])).T, self.n[init]
        els = self.system[start:stop or self.length]
        for i, el in enumerate(els):
            i += start
            yu, n = el.propagate_paraxial(yu, n, self.l)
            (self.y[i], self.u[i]), self.n[i] = yu.T, n

    def aberrations(self, start=0, stop=None):
        els = self.system[start:stop or self.length]
        for i, el in enumerate(els):
            self.v[i] = el.dispersion(self.lmin, self.lmax)
            # ignore i == 0 case, object handles it
            self.c[i] = el.aberration(self.y[i], self.u[i - 1],
                    self.n[i - 1], self.n[i], self.c.shape[-1])
        self.extrinsic_aberrations()

    def extrinsic_aberrations(self): # FIXME: wrong
        self.d[:] = 0
        st = self.system.aperture_index
        t, s = 0, 1
        kmax = self.d.shape[-1]
        r = np.zeros_like(self.d)
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
            for i in range(self.length):
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
        return f/self.n[(0, -2), :]*[-1, 1]

    @property
    def focal_distance(self):
        """front/back focal distance relative to first/last surface
        Malacara1989 p27 2.43 2.44, F-V"""
        c = self.n[(0, -2), :]*self.focal_length/self.lagrange
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
        return np.fabs(self.n[(0, -2), :]*sinarctan(self.u[(0, -2), 0]))

    @property
    def f_number(self):
        na = self.numerical_aperture
        return self.n[(0, -2), :]/(2*na)

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
        yield "front, back focal distance: %s" % self.focal_distance
        yield "front, back principal distance: %s" % self.principal_distance
        yield "front, back nodal distance: %s" % self.nodal_distance
        yield "entry, exit pupil distance: %s" % self.pupil_distance
        yield "entry, exit pupil height: %s" % self.pupil_height
        yield "front, back numerical aperture: %s" % self.numerical_aperture
        yield "front, back working f number: %s" % self.f_number
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
        ax.plot(self.z, self.y, **kwargs)
        for p, flag in [
                (self.principal_distance, principals),
                (self.focal_distance, focals),
                (self.nodal_distance, nodals),
                ]:
            if flag:
                p = p + self.z[(1, -2), :]
                h = self.system.aperture.radius
                x = np.array([-1, 1])[:, None]
                ax.plot(p*np.ones((2, 1)), 1.5*h*x, **kwargs)
        if pupils:
            p = self.pupil_distance + self.z[(1, -2), :]
            h = self.pupil_height
            x = np.array([-1.5, -1, np.nan, 1, 1.5])[:, None]
            ax.plot(p*np.ones((5, 1)), h*x, **kwargs)

    # TODO introduce aperture at argmax(abs(y_axial)/radius)
    # or at argmin(abs(u_axial))

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
        self.system.image.thickness -= self.y[-1, 0]/self.u[-1, 0]
        self.propagate()
       

class GaussianTrace(Trace):
    def __init__(self, system):
        super(GaussianTrace, self).__init__(system)
        self.allocate()
        self.rays()
        self.propagate()

    def allocate(self):
        super(GaussianTrace, self).allocate()
        self.q = np.empty((self.length,), dtype=np.complex_)
        self.l = 1.
        self.z = np.empty(self.length)
        self.n = np.empty_like(self.z)

    def rays(self, q=None):
        # 1/q = 1/R - i*lambda/(pi*n*w**2)
        # q = z + i*z0
        # z0 = pi*n*w0**2/lambda
        if q is None:
            # assert not self.system.object.infinite # let it slip
            obj = self.system.object
            self.l = obj.wavelengths[0]
            n = obj.material.refractive_index(self.l)
            q = 1j*np.pi*n*obj.radius**2/self.l*self.system.scale
        self.q[0] = q

    def propagate(self, start=0, stop=None):
        self.z = np.cumsum([e.thickness for e in self.system])
        init = start - 1 if start else 0
        q, n = self.q[init], self.n[init]
        els = self.system[start:stop or self.length]
        for i, el in enumerate(els):
            i += start
            q, n = el.propagate_gaussian(q, n, self.l)
            self.q[i], self.n[i] = q, n

    def qn_at(self, z=None):
        if z is None:
            return self.q, self.n
        else:
            # q[i] is valid after element i
            # element indices for self.z[i-1] < z < self.z[i]
            i = np.searchsorted(self.z, z) - 1
            dz = z - self.z[i, :]
            q = self.q[i, :] + dz
            n = self.n[i, :]
            return q, n

    def spot_radius_at(self, z):
        q, n = self.qn_at(z)
        r = (-1/(1/q).imag*self.l/self.system.scale/np.pi/n)**.5
        return r

    def curvature_radius_at(self, z):
        q, n = self.qn_at(z)
        r = 1/(1/q).real
        return r

    @property
    def curvature_radius(self): # on element
        return self.curvature_radius_at(z=None)

    @property
    def spot_radius(self): # on element
        return self.spot_radius_at(z=None)

    @property
    def waist_position(self): # after element relative to element
        return -self.q.real

    @property
    def rayleigh_range(self): # after element
        return self.q.imag

    @property
    def waist_radius(self): # after element
        return (self.q.imag/np.pi/self.n*self.l/self.system.scale)**.5

    @property
    def diverging(self):
        return self.curvature_radius > 0

    @property
    def confined(self):
        return self.rayleigh_range > 0

    @property
    def intensity_max(self, lambd):
        return (2/np.pi)**.5/self.waist_radius

    @property
    def eigenmodes(self):
        a, b, c, d = self.system.paraxial_matrix(self.l).flat
        q = np.roots((c, d - a, -b))
        return q

    @property
    def m(self):
        a, b, c, d = self.system.paraxial_matrix(self.l).flat
        m = (a + d)/2
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
        c = np.c_[self.z, self.curvature_radius, self.spot_radius,
                self.waist_position, self.waist_radius]
        return self.print_coeffs(c,
                "track/curv rad/spot rad/waist pos/waist rad".split("/"),
                sum=False)

    def __str__(self):
        t = itertools.chain(
                self.print_trace(), ("",),
                )
        return "\n".join(t)

    def resize(self, waists=3):
        for e, y in zip(self.system[1:], self.spot_radius[1:]):
            e.radius = y*waists

    def refocus(self):
        self.system.image.thickness += self.waist_position[-1]
        self.propagate()

    def plot(self, ax, npoints=101, waist_position=False,
            rayleigh_range=False, **kwargs):
        kwargs.setdefault("color", "black")
        z = np.linspace(min(self.z), max(self.z), npoints)
        w = self.spot_radius_at(z)
        ax.plot(z, w, **kwargs)
        ax.plot(z, -w, **kwargs)
        if waist_position:
            p = self.waist_position + self.z
            h = self.waist_radius*([-1], [1])
            p, h = np.broadcast_arrays(p, h)
            ax.plot(p, h, **kwargs)
        if rayleigh_range:
            # [lr, tb, i]
            r = self.rayleigh_range
            p = (self.waist_position + self.z + (-r, r))
            h = (self.waist_radius*2**.5*([-1], [1]))
            p, h = np.broadcast_arrays(p[:, None], h)
            p = p.reshape(2, -1)
            h = h.reshape(2, -1)
            ax.plot(p, h, **kwargs)


class FullTrace(Trace):
    def allocate(self, nrays):
        super(FullTrace, self).allocate()
        self.nrays = nrays
        self.y = np.empty((self.length, nrays, 3))
        self.u = np.empty_like(self.y)
        self.l = 1.
        self.z = np.empty(self.length)
        self.n = np.empty_like(self.z)
        self.t = np.empty((self.length, nrays))

    def rays_given(self, y, u, l=None):
        y, u = np.atleast_2d(y, u)
        y, u = np.broadcast_arrays(y, u)
        if l is None:
            l = self.system.object.wavelengths[0]
        self.allocate(max(y.shape[0], u.shape[0]))
        self.l = l
        self.y[0, :, :] = 0
        self.y[0, :, :y.shape[1]] = y
        self.u[0, :, :] = 0
        self.u[0, :, :u.shape[1]] = u
        self.u[0, :, 2] = np.sqrt(1 - np.square(self.u[0, :, :2]).sum(1))

    def propagate(self, start=0, stop=None, clip=False):
        self.z = np.cumsum([e.thickness for e in self.system])
        init = start - 1 if start else 0
        y, u, n, l = self.y[init], self.u[init], self.n[init], self.l
        for i, e in enumerate(self.system[start:stop or self.length]):
            i += start
            y, u, n, t = e.transformed_yu(e.propagate, y, u, n, l, clip)
            self.y[i], self.u[i], self.n[i], self.t[i] = y, u, n, t

    def refocus(self):
        y = self.y[-1, :, :2]
        u = tanarcsin(self.u[-2])
        good = np.all(np.isfinite(u), axis=1)
        y, u = y[good], u[good]
        y, u = (y - y.mean(0)).ravel(), (u - u.mean(0)).ravel()
        # solution of sum((y+tu-sum(y+tu)/n)**2) == min
        t = -np.dot(y, u)/np.dot(u, u)
        self.system.image.thickness += t
        self.propagate()

    def opd(self, chief=0, radius=None, after=-2, resample=4):
        t = self.t[:after + 1].sum(0)
        if self.system.object.infinite:
            # input reference sphere is a tilted plane
            # u0 * (y0 - y - t*u) == 0
            tj = np.dot(self.u[0, chief], (self.y[0, chief] - self.y[0]).T)
            t -= tj*self.n[0]
        if radius is None:
            radius = self.z[-1] - self.z[after]
            #axis=1
            #if np.fabs(self.u[-2, chief, axis]) > 1e-6:
            #    radius = self.y[-1, chief, axis]/self.u[-2, chief, axis]
        # center sphere on chief image
        y = self.y[after] - self.y[-1, chief]
        y[:, 2] -= self.z[-1] - self.z[after]
        u = self.u[after]
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
        else:
            x, y = py.T
        return x, y, t

    def psf(self, chief=0, pad=4, resample=4, **kwargs):
        radius = self.system.image.thickness
        x, y, o = self.opd(chief, resample=resample, radius=radius,
                **kwargs)
        good = np.isfinite(o)
        n = np.count_nonzero(good)
        o = np.where(good, np.exp(-2j*np.pi*o), 0)/n**.5
        if resample:
            # FIXME resample assumes constant amplitude in exit pupil
            nx, ny = (i*pad for i in o.shape)
            apsf = np.fft.fft2(o, (nx, ny))
            psf = (apsf*apsf.conj()).real/apsf.size
            dx = x[1, 0] - x[0, 0]
            k = 1/(self.l/self.system.scale)
            f = np.fft.fftfreq(nx, dx)*radius/k
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

    def aim(self, y, u, l=None, axis=1, target=0., stop=None,
            tol=1e-3, maxiter=100):
        """aims ray at aperture center (or target)
        changing angle (in case of finite object) or
        position in case of infinite object"""
        self.rays_given(y, u, l)
        var = self.y if self.system.object.infinite else self.u
        assert var.shape[1] == 1
        v0 = var[0, 0, axis].copy()

        if stop is -1: # return clipping ray
            radii = np.array([e.radius for e in self.system[1:-1]])
            def distance(a):
                var[0, 0, axis] = a*v0
                self.propagate(stop=-1, clip=False)
                res = self.y[1:-1, 0, axis]*target
                return max(res - radii)
        else: # return pupil ray
            if stop is None:
                stop = self.system.aperture_index
            target *= self.system[stop].radius
            def distance(a):
                var[0, 0, axis] = a*v0
                self.propagate(stop=stop + 1, clip=False)
                res = self.y[stop, 0, axis]
                return res - target

        def find_start(fun, a0):
            f0 = fun(a0)
            if not np.isnan(f0):
                return a0, f0
            for scale in np.logspace(.01, .3, 5):
                for ai in a0*scale, a0/scale:
                    fi = fun(ai)
                    if not np.isnan(fi):
                        return ai, fi
            raise RuntimeError("no starting ray found")

        a0, f0 = find_start(distance, 1.)
        if abs(f0 - target) > tol:
            a0 = newton(distance, a0, tol=tol, maxiter=maxiter)
        var[0, 0, axis] = a0*v0
        return self.y[0, 0, :2], self.u[0, 0, :2]

    def aim_pupil(self, height, pupil_distance, pupil_height,
            l=None, axis=(0, 1), **kwargs):
        yo = (0, height)
        pd = pupil_distance
        ph = np.ones(2)*pupil_height
        if height:
            # can only determine pupil distance if chief is non-axial
            yp = (0, 0)
            y, u = self.system.object.to_pupil(yo, yp, pd, ph[1])
            y, u = self.aim(y, u, l, axis=1, target=0)
            pd = self.system.object.pupil_distance(y, u)
            # rescale apparent pupil height (improve guess)
            ph *= pd/pupil_distance
        for ax in axis:
            yp = (0, 1) if ax else (1, 0)
            y, u = self.system.object.to_pupil(yo, yp, pd, ph[ax])
            y, u = self.aim(y, u, l, axis=ax, target=1)
            ph[ax] = self.system.object.pupil_height(y, u, pd, axis=ax)
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
            pd, ph = self.aim_pupil(height, pd, ph[axis], wavelength,
                    axis=(), **kwargs)
        except RuntimeError:
            print("chief failed", height)
            pass
        y, u = self.system.object.to_pupil(yo, (0, 0), pd, ph[axis])
        ys, us = [y], [u]
        for t in -1, 1:
            yp = [0, 0]
            yp[axis] = t
            y, u = self.system.object.to_pupil(yo, yp, pd, ph[axis])
            try:
                y, u = self.aim(y, u, wavelength, axis=axis, target=t, stop=-1)
            except RuntimeError:
                print("clipping aim failed", height, t)
                pass
            ys.append(y)
            us.append(u)
        y, u = np.vstack(ys), np.vstack(us)
        self.rays_given(y, u, wavelength)
        self.propagate(clip=clip)

    def rays_paraxial_clipping(self, paraxial, height=1.,
            wavelength=None, **kwargs):
        # TODO: refactor rays_paraxial_*
        zp = paraxial.pupil_distance[0] + paraxial.z[1]
        rp = paraxial.pupil_height[0]
        return self.rays_clipping(height, zp, rp, wavelength, **kwargs)

    def rays_point(self, height, pupil_distance, pupil_height,
            wavelength=None, nrays=11, distribution="meridional",
            clip=False, aim=(0, 1)):
        if aim:
            try:
                pupil_distance, pupil_height = self.aim_pupil(height,
                        pupil_distance, pupil_height, wavelength, axis=aim)
            except RuntimeError:
                print("pupil aim failed", height)
                pass
        icenter, yp = self.pupil_distribution(distribution, nrays)
        y, u = self.system.object.to_pupil((0, height), yp,
                pupil_distance, pupil_height)
        self.rays_given(y, u, wavelength)
        self.propagate(clip=clip)
        return icenter

    def rays_paraxial_point(self, paraxial, height=1.,
            wavelength=None, **kwargs):
        zp = paraxial.pupil_distance[0] + paraxial.z[1]
        rp = paraxial.pupil_height[0]
        return self.rays_point(height, zp, rp, wavelength, **kwargs)

    def rays_line(self, height, pupil_distance, pupil_height,
            wavelength=None, nrays=21, aim=True, eps=1e-3, clip=False):
        yi = np.c_[np.zeros(nrays), np.linspace(0, height, nrays)]
        y, u = self.system.object.to_pupil(yi, (0, 0.), pupil_distance,
                pupil_height)
        if aim:
            for i in range(y.shape[0]):
                try:
                    y[i], u[i] = self.aim(y[i], u[i], wavelength, axis=1)
                except RuntimeError:
                    print("chief aim failed", i)
                    pass
        e = np.zeros((3, 1, 2)) # pupil
        e[(1, 2), :, (1, 0)] = eps*pupil_height # meridional, sagittal
        if self.system.object.infinite:
            y = (y + e).reshape(-1, 2)
            u = np.tile(u, (3, 1))
        else:
            y = np.tile(y, (3, 1))
            u = (u + e/pupil_distance).reshape(-1, 2)
        self.rays_given(y, u, wavelength)
        self.propagate(clip=clip)

    def rays_paraxial_line(self, paraxial, height=1.,
            wavelength=None, **kwargs):
        zp = paraxial.pupil_distance[0] + paraxial.z[1]
        rp = paraxial.pupil_height[0]
        return self.rays_line(height, zp, rp, wavelength, **kwargs)

    def resize(self, fn=lambda a, b: a):
        x, y = self.y[:, :, 0], self.y[:, :, 1]
        r = np.hypot(x, y)
        for e, ri in zip(self.system[1:], r[1:]):
            e.radius = fn(ri, e.radius)

    def plot(self, ax, axis=1, **kwargs):
        kwargs.setdefault("color", "green")
        y = self.y[:, :, axis]
        z = self.y[:, :, 2] + self.z[:, None]
        ax.plot(z, y, **kwargs)

    def print_trace(self):
        for i in range(self.nrays):
            yield "ray %i" % i
            c = np.concatenate((self.n[:, None], self.z[:, None],
                np.cumsum(self.t[:, i, None], axis=0)-self.z[:, None],
                self.y[:, i, :], self.u[:, i, :]), axis=1)
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

