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

import itertools

import numpy as np
from scipy.optimize import (newton, fsolve)
import matplotlib.pyplot as plt

from traits.api import (HasTraits, Float, Array, Property,
    cached_property, Instance, Int, Enum)

from .material import lambda_d
from .system import System
from .special_sums import polar_sum
from .aberration_orders import aberration_trace


def dir_to_angles(x,y,z):
    r = np.array([x,y,z], dtype=np.float64)
    return r/np.linalg.norm(r)

def tanarcsin(u):
    return u/np.sqrt(1-u**2)

def sinarctan(u):
    return u/np.sqrt(1+u**2)


class Trace(HasTraits):
    length = Int()
    nrays = Int()
    l = Array(dtype=np.float64, shape=(None)) # wavelength
    n = Array(dtype=np.float64, shape=(None, None)) # refractive index
    y = Array(dtype=np.float64, shape=(3, None, None)) # height
    u = Array(dtype=np.float64, shape=(3, None, None)) # angle

    def allocate(self):
        self.l = np.zeros((self.nrays,), dtype=np.float64)
        self.n = np.zeros((self.length, self.nrays), dtype=np.float64)
        self.y = np.zeros((3, self.length, self.nrays), dtype=np.float64)
        self.u = np.zeros((3, self.length, self.nrays), dtype=np.float64)

    def __init__(self, **kw):
        super(Trace, self).__init__(**kw)
        self.length = len(self.system.elements)


class ParaxialTrace(Trace):
    system = Instance(System)
    
    # marginal/axial, principal/chief
    nrays = 2

    v = Array(dtype=np.float64, shape=(None, None)) # dispersion
    l1 = Array(dtype=np.float64, shape=(None)) # min l
    l2 = Array(dtype=np.float64, shape=(None)) # max l
    aberration_c = Array(dtype=np.float64, shape=(None, 2, 2, 4, 4, 4))
    aberration_d = Array(dtype=np.float64, shape=(None, 2, 2, 4, 4, 4))

    def allocate(self):
        super(ParaxialTrace, self).allocate()
        self.v = np.zeros((self.length, self.nrays), dtype=np.float64)
        self.l1 = np.zeros((self.nrays,), dtype=np.float64)
        self.l2 = np.zeros((self.nrays,), dtype=np.float64)
        self.aberration_c = np.zeros((self.length, 2, 2, 4, 4, 4), dtype=np.float64)
        self.aberration_d = np.zeros((self.length, 2, 2, 4, 4, 4), dtype=np.float64)

    lagrange = Property
    height = Property
    focal_length = Property
    focal_distance = Property
    pupil_height = Property
    pupil_position = Property
    f_number = Property
    numerical_aperture = Property
    airy_radius = Property
    magnification = Property

    def __init__(self, **k):
        super(ParaxialTrace, self).__init__(**k)
        self.allocate()
        self.l[:] = self.system.object.wavelengths[0]
        self.l1[:] = min(self.system.object.wavelengths)
        self.l2[:] = max(self.system.object.wavelengths)
        self.n[0] = map(self.system.object.material.refractive_index,
                self.l)

    def __str__(self):
        t = itertools.chain(
                self.print_params(), ("",),
                self.print_trace(), ("",),
                self.print_c3(), ("",),
                #self.print_h3(), ("",),
                self.print_c5(),
                )
        return "\n".join(t)

    def find_rays(self):
        c = self.system.object.radius
        a, b = self.u, self.y
        if self.system.object.infinity:
            a, b = b, a
        eps = 1e-2
        a[0, 0], b[0, 0] = (eps, 0), (0, eps)
        r, h, k = self.to_aperture()
        a[0, 0], b[0, 0] = (r*eps/h[0], -c*h[1]/h[0]), (0, c)

    # TODO introduce aperture at argmax(abs(y_marginal)/radius)
    # or at argmin(abs(u_marginal))

    def size_elements(self):
        for e, y in zip(self.system.elements[1:], self.y[0, 1:]):
            e.radius = np.fabs(y).sum() # marginal+chief

    def propagate(self):
        self.find_rays()
        for i, e in enumerate(self.system.elements):
            e.propagate_paraxial(self, i)
        self.aberration_c, self.aberration_d = aberration_trace(self,
                kmax=4)

    def to_aperture(self):
        for i, e in enumerate(self.system.elements):
            e.propagate_paraxial(self, i)
            if e.typestr == "A":
                return e.radius, self.y[0, i], self.u[0, i]

    def focal_length_solve(self, f, i=None):
        # TODO only works for last surface
        if i is None:
            i = len(self.system.elements)-2
        y0, y = self.y[0, (i-1, i), 0]
        u0, u = self.u[0, i-1, 0], -self.y[0, 0, 0]/f
        n0, n = self.n[(i-1, i), 0]
        c = (n0*u0-n*u)/(y*(n-n0))
        self.system.elements[i].curvature = c

    def focal_plane_solve(self):
        self.system.image.origin[2] -= self.y[0, -1, 0]/self.u[0, -1, 0]

    def print_coeffs(self, coeff, labels):
        yield ("%2s %1s" + "% 10s" * len(labels)) % (
                ("#", "T") + tuple(labels))
        fmt = "%2s %1s" + "% 10.4g" * len(labels)
        for i, a in enumerate(coeff):
            yield fmt % ((i, self.system.elements[i].typestr) + tuple(a))
        yield fmt % ((" ∑", "") + tuple(coeff.sum(0)))

    def print_c3(self):
        c = self.aberration_c
        c = np.array([
                -2*c[:, 0, 1, 1, 0, 0],
                -c[:, 0, 1, 0, 1, 0],
                -c[:, 0, 0, 0, 1, 0],
                c[:, 0, 0, 0, 1, 0] - 2*c[:, 0, 1, 0, 0, 1],
                -2*c[:, 0, 0, 0, 0, 1],
                ])
        # transverse image seidel (like oslo)
        return self.print_coeffs(c.T*self.height[1]/2/self.lagrange,
                "SA3 CMA3 AST3 PTZ3 DIS3".split())

    def print_h3(self):
        c3a = self.aberration3*8 # chromatic
        return self.print_coeffs(c3a[(6, 12), :].T, 
                "PLC PTC".split())

    def print_c5(self):
        c = self.aberration_c + self.aberration_d
        c = np.array([
                -2*c[:, 0, 1, 2, 0, 0], # SA5
                -1*c[:, 0, 1, 1, 1, 0], # CMA5
                -2*c[:, 0, 0, 1, 1, 0]-2*c[:, 0, 1, 1, 0, 1]+2*c[:, 0, 1, 0, 2, 0], # TOBSA5
                2*c[:, 0, 1, 1, 0, 1], # SOBSA5
                -2*c[:, 0, 0, 1, 0, 1]-2*c[:, 0, 0, 0, 2, 0]-2*c[:, 0, 1, 0, 1, 1], # TECMA5
                -1*c[:, 0, 1, 0, 1, 1], # SECMA5
                -c[:, 0, 0, 0, 1, 1]/2, # AST5
                -2*c[:, 0, 1, 0, 0, 2]+c[:, 0, 0, 0, 1, 1]/2, # PTZ5
                -2*c[:, 0, 0, 0, 0, 2], # DIS5
                ])
        # transverse image seidel (like oslo)
        return self.print_coeffs(c.T*self.height[1]/2/self.lagrange,
                "SA5 CMA5 TOBSA5 SOBSA5 TECMA5 SECMA5 AST5 PTZ5 DIS5".split())

    def print_params(self):
        yield "lagrange: %.5g" % self.lagrange
        yield "focal length: %.5g" % self.focal_length
        yield "object, image height: %.5g, %.5g" % self.height
        yield "front, back focal distance: %.5g, %.5g" % self.focal_distance
        yield "entry, exit pupil position: %.5g, %.5g" % self.pupil_position
        yield "entry, exit pupil height: %.5g, %.5g" % self.pupil_height
        yield "front, back numerical aperture: %.5g, %.5g" % self.numerical_aperture
        yield "front, back working f number: %.5g, %.5g" % self.f_number
        yield "front, back airy radius: %.5g, %.5g" % self.airy_radius
        yield "transverse, angular magnification: %.5g, %.5g" % self.magnification

    def print_trace(self):
        # TODO: would like surface incidence angles
        yield "%2s %1s% 10s% 10s% 10s% 10s" % (
                "#", "T", "marg h", "marg a", "chief h", "chief a")
        for i, ((hm, hc), (am, ac)) in enumerate(zip(self.y[0], self.u[0])):
            yield "%2s %1s% 10.4g% 10.4g% 10.4g% 10.4g" % (
                    i, self.system.elements[i].typestr, hm, am, hc, ac)

    def _get_lagrange(self):
        return self.n[0,0]*(
                self.u[0,0,0]*self.y[0,0,1]-
                self.u[0,0,1]*self.y[0,0,0])

    def _get_focal_length(self):
        return -self.lagrange/self.n[0,0]/(
                self.u[0,0,0]*self.u[0,-2,1]-
                self.u[0,0,1]*self.u[0,-2,0])

    def _get_height(self):
        return self.y[0, 0, 1], self.lagrange/(self.n[-2,0]*self.u[0,-2,0])
 
    def _get_focal_distance(self):
        return (-self.y[0,1,0]/self.u[0,0,0],
                -self.y[0,-2,0]/self.u[0,-2,0])
       
    def _get_numerical_aperture(self):
        return (abs(self.n[0,0]*self.u[0,0,0]),
                abs(self.n[-2,0]*self.u[0,-2,0]))

    def _get_pupil_position(self):
        return (self.y[2,0,1]-self.y[0,0,1]/self.u[0,0,1],
                self.y[2,-1,1]-self.y[0,-1,1]/self.u[0,-1,1])

    def _get_pupil_height(self):
        return (self.y[0,0,0]+(self.pupil_position[0]-self.y[2,0,0])*self.u[0,0,0],
                self.y[0,-1,0]+(self.pupil_position[1]-self.y[2,-1,0])*self.u[0,-1,0])

    def _get_f_number(self):
        return (1/(2*self.numerical_aperture[0]),
                1/(2*self.numerical_aperture[1]))

    def _get_airy_radius(self):
        return (1.22*self.l[0]/(2*self.numerical_aperture[0]),
                1.22*self.l[0]/(2*self.numerical_aperture[1]))

    def _get_magnification(self):
        return ((self.n[0,0]*self.u[0,0,0])/(
                self.n[-2,0]*self.u[0,-2,0]),
                (self.n[-2,1]*self.u[0,-2,1])/(
                self.n[0,1]*self.u[0,0,1]))


class FullTrace(Trace):
    o = Array(dtype=np.float64, shape=(None)) # intensity
    p = Array(dtype=np.float64, shape=(None, None)) # lengths

    def allocate(self):
        super(FullTrace, self).allocate()
        self.o = np.zeros((self.nrays,), dtype=np.float64)
        self.p = np.zeros((self.length, self.nrays), dtype=np.float64)

    def rays_like_paraxial(self, paraxial):
        self.nrays = 2
        self.allocate()
        self.l = paraxial.l
        self.n[0] = paraxial.n[0]
        self.y[0, 0] = paraxial.y[0, 0]
        self.y[1, 0] = 0.
        self.y[2, 0] = 0.
        self.u[0, 0] = sinarctan(paraxial.u[0, 0])
        self.u[1, 0] = 0.
        self.u[2, 0] = np.sqrt(1-self.u[0, 0]**2)

    def rays_for_point(self, paraxial, height, wavelength, nrays,
            distribution):
        # TODO apodization
        xp, yp = self.get_rays(distribution, nrays)
        hp, rp = paraxial.pupil_position[0], paraxial.pupil_height[0]
        r = self.system.object.radius
        if self.system.object.infinity:
            r = sinarctan(r)
            p, q = height[0]*r, height[1]*r
            a, b = xp*rp-hp*tanarcsin(p), yp*rp-hp*tanarcsin(q)
        else:
            a, b = height[0]*r, height[1]*r
            p, q = sinarctan((xp*rp-a)/hp), sinarctan((yp*rp-b)/hp)
        self.nrays = xp.shape[0]
        self.allocate()
        self.l[:] = wavelength
        self.n[0] = self.system.object.material.refractive_index(
                wavelength)
        self.y[0, 0] = a
        self.y[1, 0] = b
        self.y[2, 0] = 0
        self.u[0, 0] = p
        self.u[1, 0] = q
        self.u[2, 0] = np.sqrt(1-p**2-q**2)

    def rays_for_object(self, paraxial, wavelength, nrays, eps=1e-6):
        hp, rp = paraxial.pupil_position[0], paraxial.pupil_height[0]
        r = self.system.object.radius
        if self.system.object.infinity:
            r = sinarctan(r)
        xi, yi = np.tile([np.linspace(0, r, nrays), np.zeros((nrays,),
            dtype=np.float64)], 3)
        xp, yp = np.zeros_like(xi), np.zeros_like(yi)
        xp[nrays:2*nrays] = eps*rp
        yp[2*nrays:] = eps*rp
        if self.system.object.infinity:
            p, q = xi, yi
            a, b = xp-hp*tanarcsin(p), yp-hp*tanarcsin(q)
        else:
            a, b = xi, yi
            p, q = sinarctan((xp-a)/hp), sinarctan((yp-b)/hp)
        self.nrays = nrays*3
        self.allocate()
        self.l[:] = wavelength
        self.n[0] = self.system.object.material.refractive_index(
                wavelength)
        self.y[0, 0] = a
        self.y[1, 0] = b
        self.y[2, 0] = 0
        self.u[0, 0] = p
        self.u[1, 0] = q
        self.u[2, 0] = np.sqrt(1-p**2-q**2)

    def plot_transverse(self, heights, wavelengths, fig=None, paraxial=None,
            npoints_spot=100, npoints_line=30):
        if fig is None:
            fig = plt.figure(figsize=(10, 8))
            fig.subplotpars.left = .05
            fig.subplotpars.bottom = .05
            fig.subplotpars.right = .95
            fig.subplotpars.top = .95
            fig.subplotpars.hspace = .2
            fig.subplotpars.wspace = .2
        if paraxial is None:
            paraxial = ParaxialTrace(system=self.system)
            paraxial.propagate()
        nh = len(heights)
        ia = self.system.aperture_index
        n = npoints_line
        gs = plt.GridSpec(nh, 6)
        axm0, axs0, axl0, axc0 = None, None, None, None
        for i, hi in enumerate(heights):
            axm = fig.add_subplot(gs.new_subplotspec((i, 0), 1, 2),
                    sharex=axm0, sharey=axm0)
            if axm0 is None: axm0 = axm
            #axm.set_title("meridional h=%s, %s" % hi)
            #axm.set_xlabel("Y")
            #axm.set_ylabel("tanU")
            axs = fig.add_subplot(gs.new_subplotspec((i, 2), 1, 1),
                    sharex=axs0, sharey=axs0)
            if axs0 is None: axs0 = axs
            #axs.set_title("sagittal h=%s, %s" % hi)
            #axs.set_xlabel("X")
            #axs.set_ylabel("tanV")
            axl = fig.add_subplot(gs.new_subplotspec((i, 3), 1, 1),
                    sharex=axl0, sharey=axl0)
            if axl0 is None: axl0 = axl
            #axl.set_title("longitudinal h=%s, %s" % hi)
            #axl.set_xlabel("Z")
            #axl.set_ylabel("H")
            axp = fig.add_subplot(gs.new_subplotspec((i, 4), 1, 1),
                    aspect="equal", sharex=axs0, sharey=axm0)
            #axp.set_title("rays h=%s, %s" % hi)
            #axp.set_ylabel("X")
            #axp.set_ylabel("Y")
            axc = fig.add_subplot(gs.new_subplotspec((i, 5), 1, 1),
                    sharex=axc0, sharey=axc0)
            if axc0 is None: axc0 = axc
            #axc.set_title("encircled h=%s, %s" % hi)
            #axc.set_ylabel("R")
            #axc.set_ylabel("E")
            for j, wi in enumerate(wavelengths):
                self.rays_for_point(paraxial, hi, wi, npoints_line, "tee")
                self.propagate()
                # top rays (small tanU) are right/top
                axm.plot(-tanarcsin(self.u[0, -1, :2*n/3])
                        +tanarcsin(paraxial.u[0, -1, 1])*hi[0],
                        self.y[0, -1, :2*n/3]-paraxial.y[0, -1, 1]*hi[0],
                        "-", label="%s" % wi)
                axs.plot(self.y[1, -1, 2*n/3:],
                        -tanarcsin(self.u[1, -1, 2*n/3:]),
                        "-", label="%s" % wi)
                axl.plot(-(self.y[0, -1, :2*n/3]-paraxial.y[0, -1, 1]*hi[0])*
                        self.u[2, -1, :2*n/3]/self.u[0, -1, :2*n/3],
                        self.y[0, ia, :2*n/3],
                        "-", label="%s" % wi)
                self.rays_for_point(paraxial, hi, wi, npoints_spot,
                        "random")
                self.propagate()
                axp.plot(self.y[1, -1]-paraxial.y[0, -1, 1]*hi[1],
                        self.y[0, -1]-paraxial.y[0, -1, 1]*hi[0],
                        ".", markersize=3, markeredgewidth=0,
                        label="%s" % wi)
                xy = self.y[(0, 1), -1]
                xy = xy[:, np.all(np.isfinite(xy), 0)]
                xym = xy.mean(axis=1)
                r = ((xy-xym[:, None])**2).sum(axis=0)**.5
                rb = np.bincount(
                        (r*(npoints_line/r.max())).astype(np.int),
                        minlength=npoints_line+1).cumsum()
                axc.plot(np.linspace(0, r.max(), npoints_line+1),
                        rb.astype(np.float)/self.y.shape[2])
            for ax in axs0, axm0, axc0:
                ax.relim()
                ax.autoscale_view(True, True, True)
        return fig

    def plot_longitudinal(self, wavelengths, fig=None, paraxial=None,
            npoints=20):
        if fig is None:
            fig = plt.figure(figsize=(6, 4))
            fig.subplotpars.left = .05
            fig.subplotpars.bottom = .05
            fig.subplotpars.right = .95
            fig.subplotpars.top = .95
            fig.subplotpars.hspace = .2
            fig.subplotpars.wspace = .2
        if paraxial is None:
            paraxial = ParaxialTrace(system=self.system)
            paraxial.propagate()
        n = npoints
        gs = plt.GridSpec(1, 2)
        axl = fig.add_subplot(gs.new_subplotspec((0, 0), 1, 1))
        #axl.set_title("distortion")
        #axl.set_xlabel("D")
        #axl.set_ylabel("Y")
        axc = fig.add_subplot(gs.new_subplotspec((0, 1), 1, 1))
        #axl.set_title("field curvature")
        #axl.set_xlabel("Z")
        #axl.set_ylabel("Y")
        for i, (wi, ci) in enumerate(zip(wavelengths, "bgrcmyk")):
            self.rays_for_object(paraxial, wi, npoints)
            self.propagate()
            axl.plot(self.y[0, -1, :npoints]-np.linspace(0, paraxial.height[1], npoints),
                self.y[0, -1, :npoints], ci+"-", label="d")
            # tangential field curvature
            # -(real_y-parax_y)/(tanarcsin(real_u)-tanarcsin(parax_u))
            xt = -(self.y[0, -1, npoints:2*npoints]-self.y[0, -1, :npoints])/(
                  tanarcsin(self.u[0, -1, npoints:2*npoints])-tanarcsin(self.u[0, -1, :npoints]))
            # sagittal field curvature
            # -(real_x-parax_x)/(tanarcsin(real_v)-tanarcsin(parax_v))
            xs = -(self.y[1, -1, 2*npoints:]-self.y[1, -1, :npoints])/(
                  tanarcsin(self.u[1, -1, 2*npoints:])-tanarcsin(self.u[1, -1, :npoints]))
            axc.plot(xt, self.y[0, -1, :npoints], ci+"--", label="zt")
            axc.plot(xs, self.y[0, -1, :npoints], ci+"-", label="zs")
        return fig

    def get_rays(self, distribution, nrays):
        d = distribution
        n = nrays
        if d == "random":
            xy = 2*np.random.rand(2, n*4/np.pi)-1
            return xy[:, (xy**2).sum(0)<=1]
        elif d == "meridional":
            return np.linspace(-1, 1, n), np.zeros((n,))
        elif d == "sagittal":
            return np.zeros((n,)), np.linspace(-1, 1, n)
        elif d == "square":
            r = np.around(np.sqrt(n*4/np.pi))
            x, y = np.mgrid[-1:1:1j*r, -1:1:1j*r]
            xy = np.array([x.ravel(), y.ravel()])
            return xy[:, (xy**2).sum(0)<=1]
        elif d == "triangular":
            r = np.around(np.sqrt(n*4/np.pi))
            x, y = np.mgrid[-1:1:1j*r, -1:1:1j*r]
            xy = np.array([x.ravel(), y.ravel()])
            return xy[:, (xy**2).sum(0)<=1]
        elif d == "hexapolar":
            r = int(np.around(np.sqrt(n/3.-1/12.)-1/2.))
            l = [[np.array([0]), np.array([0])]]
            for i in range(1, r+1):
                a = np.arange(0, 2*np.pi, 2*np.pi/(6*i))
                l.append([i*np.sin(a)/r, i*np.cos(a)/r])
            return np.concatenate(l, axis=1)
        elif d == "cross":
            return np.concatenate([
                [np.linspace(-1, 1, n/2), np.zeros((n/2,))],
                [np.zeros((n/2,)), np.linspace(-1, 1, n/2)]], axis=1)
        elif d == "tee":
            return np.concatenate([
                [np.linspace(-1, 1, 2*n/3), np.zeros((2*n/3,))],
                [np.zeros((n/3,)), np.linspace(0, 1, n/3)]], axis=1)

    def propagate(self, clip=True):
        for i, e in enumerate(self.system.elements):
            e.propagate(self, i, clip)

    def __str__(self):
        t = itertools.chain(
                #self.print_params(),
                self.print_trace(),
                #self.print_c3(),
                )
        return "\n".join(t)

    def print_trace(self):
        yield "%2s %1s% 10s% 10s% 10s% 10s% 10s% 10s% 10s" % (
                "#", "T", "height x", "height y", "height z",
                "angle x", "angle y", "angle z", "length")
        for i in range(self.nrays):
            yield ""
            yield "ray %i" % i
            for j in range(self.length):
                yield "%2s %1s% 10.4g% 10.4g% 10.4g% 10.4g% 10.4g% 10.4g% 10.4g" % (
                        j, self.system.elements[j].typestr, 
                        self.y[0, j, i], self.y[1, j, i], self.y[2, j, i],
                        self.u[0, j, i], self.u[1, j, i], self.u[2, j, i],
                        self.p[j, i])
