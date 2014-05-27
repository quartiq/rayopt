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

import itertools

import numpy as np
from scipy.interpolate import griddata

from .elements import Spheroid
from .utils import sinarctan, tanarcsin, public, pupil_distribution
from .raytrace import Trace


@public
class GeometricTrace(Trace):
    """
    y[i]: intercept at surface
    i[i]: incoming/incidence direction before surface
    u[i]: outgoing/excidence direction after surface
    all in i-surface normal coordinates relative to vertex
    """
    def allocate(self, nrays):
        super(GeometricTrace, self).allocate()
        self.nrays = nrays
        self.n = np.empty(self.length)
        self.y = np.empty((self.length, nrays, 3))
        self.u = np.empty_like(self.y)
        self.i = np.empty_like(self.y)
        self.w = None
        self.ref = None
        self.l = 1.
        self.t = np.empty((self.length, nrays))

    def rays_given(self, y, u, l=None, w=None, ref=None):
        y, u = np.atleast_2d(y, u)
        y, u = np.broadcast_arrays(y, u)
        n, m = y.shape
        if not hasattr(self, "y") or self.y.shape[1] != n:
            self.allocate(n)
        if l is None:
            l = self.system.wavelengths[0]
        if w is None:
            w = np.ones(n)/n
        self.w = w
        self.ref = ref
        self.l = l
        self.y[0, :, :m] = y
        self.y[0, :, m:] = 0
        self.u[0, :, :m] = u
        if m < 3: # assumes forward rays
            u2 = np.square(self.u[0, :, :2]).sum(-1)
            self.u[0, :, 2] = np.sqrt(1 - u2)
        self.i[0] = self.u[0]
        self.n[0] = self.system[0].refractive_index(l)
        self.t[0] = 0

    def propagate(self, start=1, stop=None, clip=False):
        super(GeometricTrace, self).propagate()
        init = start - 1
        y, u, n, l = self.y[init], self.u[init], self.n[init], self.l
        y, u = self.system[init].from_normal(y, u)
        for j, yunit in enumerate(self.system.propagate(
                y, u, n, l, start, stop, clip)):
            j += start
            self.y[j], self.u[j], self.n[j], self.i[j], self.t[j] = yunit

    def refocus(self, at=-1):
        y = self.y[at, :, :2]
        u = tanarcsin(self.i[at])
        good = np.all(np.isfinite(u), axis=1)
        y, u = y[good], u[good]
        if self.w is not None:
            w = self.w[good]
        else:
            w = np.ones(y.shape[0])
        y = y - y.mean(0)
        u = u - u.mean(0)
        wy = (w[:, None]*y).ravel()
        wu = (w[:, None]*u).ravel()
        u = u.ravel()
        # solution of sum(w*(y+tu-sum(y+tu)/n)**2) == min
        t = -np.dot(wy, u)/np.dot(wu, u)
        self.system[at].distance += t
        self.propagate()

    def opd(self, radius=None, after=-2, image=-1, resample=4):
        chief = self.ref
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

    def psf(self, pad=4, resample=4, **kwargs):
        radius = self.system[-1].distance
        x, y, o = self.opd(resample=resample, radius=radius,
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

    def rms(self, i=-1, ref=None):
        y = self.y[i, :, :2]
        if ref is None:
            y0 = y.mean(0)
        else:
            y0 = y[ref]
        r = np.square(y - y0).sum(1)
        if self.w is not None:
            w = self.w
        else:
            w = np.ones_like(r)/r.shape[0]
        r = (r*w).sum()
        return np.sqrt(r)

    def rays_paraxial(self, paraxial):
        y = np.zeros((2, 2))
        y[:, paraxial.axis] = paraxial.y[0]
        u = np.zeros((2, 2))
        u[:, paraxial.axis] = sinarctan(paraxial.u[0])
        self.rays_given(y, u)
        self.propagate()

    def rays(self, yo, yp, wavelength, stop=None, filter=None,
            clip=False, weight=None, ref=0):
        if filter is None:
            filter = not clip
        z, p = self.system.pupil(yo, l=wavelength, stop=stop)
        y, u = self.system.aim(yo, yp, z, p, filter=filter)
        self.rays_given(y, u, wavelength, weight, ref)
        self.propagate(clip=clip)

    def rays_point(self, yo, wavelength=None, nrays=11,
            distribution="meridional", filter=None, stop=None,
            clip=False):
        ref, yp, weight = pupil_distribution(distribution, nrays)
        self.rays(yo, yp, wavelength, filter=filter, stop=stop,
                clip=clip, weight=weight, ref=ref)

    def rays_clipping(self, yo, wavelength=None, axis=1):
        z, p = self.system.pupil(yo, l=wavelength, stop=-1)
        yp = np.zeros((3, 2))
        yp[1:, axis] = p[:, axis]/np.fabs(p).max()
        self.rays(yo, yp, wavelength, stop=-1, filter=False)

    def rays_line(self, yo, wavelength=None, nrays=21, eps=1e-2):
        yi = np.linspace(0, 1, nrays)[:, None]*np.atleast_2d(yo)
        y = np.empty((3, nrays, 3))
        u = np.empty_like(y)
        e = np.zeros((3, 2)) # chief, meridional, sagittal
        e[(1, 2), (1, 0)] = eps
        z, p = self.system.pupil((0, 0), l=wavelength)
        for i in range(yi.shape[0]):
            z = self.system.aim_chief(yi[i], z, np.fabs(p).max(),
                    l=wavelength)
            y[:, i], u[:, i] = self.system.aim(yi[i], e, z, p)
        self.rays_given(y.reshape(-1, 3), u.reshape(-1, 3), wavelength)
        self.propagate()

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
