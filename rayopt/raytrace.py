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
from scipy.optimize import newton

# from .special_sums import polar_sum
# from .aberration_orders import aberration_trace


def dir_to_angles(r):
    return r/np.linalg.norm(r)

def tanarcsin(u):
    return u/np.sqrt(1 - np.square(u))

def sinarctan(u):
    return u/np.sqrt(1 + np.square(u))


class Trace(object):
    def __init__(self, system):
        self.system = system

    def allocate(self):
        self.length = len(self.system)

    def print_coeffs(self, coeff, labels, sum=True):
        yield ("%2s %1s" + "% 10s" * len(labels)) % (
                ("#", "T") + tuple(labels))
        fmt = "%2s %1s" + "% 10.4g" * len(labels)
        for i, a in enumerate(coeff):
            yield fmt % ((i, self.system[i].typ) + tuple(a))
        if sum:
            yield fmt % ((" ∑", "") + tuple(coeff.sum(0)))


class ParaxialTrace(Trace):
    def __init__(self, system, aberration_orders=4):
        super(ParaxialTrace, self).__init__(system)
        self.allocate(aberration_orders)
        self.find_rays()
        self.propagate()

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

    def propagate(self, start=0, stop=None):
        self.z = np.cumsum([e.thickness for e in self.system])
        init = start - 1 if start else 0
        yu0, n0 = np.array((self.y[init], self.u[init])).T, self.n[init]
        for i, el in enumerate(self.system[start:stop or self.length]):
            yu, n = el.propagate_paraxial(yu0, n0, self.l)
            (self.y[i], self.u[i]), self.n[i] = yu.T, n
            self.c[i] = el.aberration(yu[:, 0], yu0[:, 1],
                    n0, n, self.c.shape[-1])
            self.v[i] = el.dispersion(self.lmin, self.lmax)
            yu0, n0 = yu, n
        self.d[:] = 0 # TODO

    def find_rays(self):
        y, u = self.y, self.u
        l = self.system.object.wavelengths[0]
        ai = self.system.aperture_index
        m = self.system.paraxial_matrix(l, stop=ai + 1)
        mi = np.linalg.inv(m)
        r = self.system[ai].radius
        c = self.system.object.radius
        if not self.system.object.infinite:
            y, u, m = u, y, mi[::-1]
        y[0, 0], u[0, 0] = r*mi[0, 0] - r*mi[0, 1]*mi[1, 0]/mi[1, 1], 0
        y[0, 1], u[0, 1] = c*mi[0, 1]/mi[1, 1], c

    def __str__(self):
        t = itertools.chain(
                self.print_params(), ("",),
                self.print_trace(), ("",),
                self.print_c3(), ("",),
                #self.print_h3(), ("",),
                self.print_c5(),
                )
        return "\n".join(t)

    # TODO introduce aperture at argmax(abs(y_axial)/radius)
    # or at argmin(abs(u_axial))

    def size_elements(self):
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

    def focal_plane_solve(self):
        self.system.image.thickness -= self.y[-1, 0]/self.u[-1, 0]
        self.propagate()

    def plot(self, ax, principals=False, pupils=False, focals=False,
            nodals=False, **kwargs):
        kwargs.setdefault("linestyle", "-")
        kwargs.setdefault("color", "green")
        ax.plot(self.z, self.y[:, 0], **kwargs)
        ax.plot(self.z, self.y[:, 1], **kwargs)
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

    def print_c3(self):
        c = self.c
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
        return self.print_coeffs(c.T*self.height[1]/2/self.lagrange,
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
        c = np.c_[self.y[:, 0], self.u[:, 0], self.y[:, 1], self.u[:, 1]]
        return self.print_coeffs(c,
                "axial y/axial u/chief y/chief u".split("/"), sum=False)
        
    @property
    def track(self):
        return self.z[-2] - self.z[1]

    @property
    def height(self):
        "object and image ray height"
        return self.y[(0, -1), 1]
        #self.lagrange/(self.n[-2]*self.u[-2,0])

    @property
    def pupil_distance(self):
        "pupil location relative to first/last surface"
        return -self.y[(1, -2), 1]/self.u[(0, -2), 1]

    @property
    def pupil_height(self):
        p = self.pupil_distance
        return self.y[(1, -2), 0] + p*self.u[(0, -2), 0]

    @property
    def lagrange(self):
        return self.n[0]*(self.u[0,0]*self.y[0,1] - self.u[0,1]*self.y[0,0])

    @property
    def focal_length(self):
        """signed distance from principal planes to foci
        Malacara1989 p27 2.41, 2.42: F-P"""
        f = self.lagrange/(
                self.u[0, 1]*self.u[-2, 0] -
                self.u[0, 0]*self.u[-2, 1])
        return f/self.n[(0, -2), :]*[-1, 1]

    @property
    def focal_distance(self):
        """ffd bfd relative to first/last surfaces
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
        return np.fabs(self.n[(0, -2), :]*sinarctan(self.u[(0, -2), 0]))

    @property
    def f_number(self):
        na = self.numerical_aperture
        return self.n[(0, -2), :]/(2*na)

    @property
    def airy_radius(self):
        na = self.numerical_aperture
        return 1.22*self.l/(2*na)

    @property
    def magnification(self):
        return (self.n[(0, -2), :]*self.u[(0, -2), (0,
            1)])/(self.n[(-2, 0), :]*self.u[(-2, 0), (0, 1)])

    @property
    def number_of_points(self):
        """number of resolvable independent diffraction points
        (assuming no aberrations)"""
        return 4*self.lagrange**2/self.l**2



class FullTrace(Trace):
    def allocate(self, nrays):
        super(FullTrace, self).allocate()
        self.nrays = nrays
        self.y = np.empty((self.length, nrays, 3))
        self.u = np.empty_like(self.y)
        self.l = 1.
        self.z = np.empty(self.length)
        self.n = np.empty((self.length, nrays))
        self.t = np.empty_like(self.n)

    def propagate(self, start=0, stop=None, clip=True):
        self.z = np.cumsum([e.thickness for e in self.system])
        init = start - 1 if start else 0
        y, u, n, l = self.y[init], self.u[init], self.n[init], self.l
        for i, e in enumerate(self.system[start:stop or self.length]):
            y, u, n, t = e.transformed_yu(e.propagate, y, u, n, l, clip)
            self.y[i], self.u[i], self.n[i], self.t[i] = y, u, n, t

    def size_elements(self, fn=lambda a, b: a, axis=0):
        for e, y in zip(self.system[1:], self.y[1:, :, axis]):
            e.radius = fn(np.fabs(y).max(), e.radius)

    def plot(self, ax, axis=0, **kwargs):
        kwargs.setdefault("linestyle", "-")
        kwargs.setdefault("color", "green")
        kwargs.setdefault("alpha", .3)
        y = self.y[:, :, 0]
        z = self.y[:, :, 2] + self.z[:, None]
        ax.plot(z, y, **kwargs)

    def rays_given(self, y, u, l=None):
        y, u = np.atleast_2d(y, u)
        if l is None:
            l = self.system.object.wavelengths[0]
        self.allocate(y.shape[0])
        self.l = l
        self.y[0, :, :] = 0
        self.y[0, :, :y.shape[1]] = y
        self.u[0, :, :] = 0
        self.u[0, :, :u.shape[1]] = u
        self.u[0, :, 2] = np.sqrt(1 - np.square(self.u[0, :, :2]).sum(1))

    def rays_like_paraxial(self, paraxial):
        y = paraxial.y[0, :, None]
        u = paraxial.u[0, :, None]
        self.rays_given(y, sinarctan(u))
        self.propagate(clip=False)

    def aim(self, index=0, axis=0, tol=1e-2, maxiter=10):
        """aims ray by index at aperture center
        angle (in case of finite object) or
        position in case of infinite object"""
        var = self.y if self.system.object.infinite else self.u
        stop = self.system.aperture_index
        v0 = var[0, index, axis].copy()

        def distance(a):
            var[0, index, axis] = a*v0
            self.propagate(stop=stop + 1, clip=False)
            res = self.y[stop, index, axis]
            return res

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

        try:
            a0, f0 = find_start(distance, 1.)
            if abs(f0) > tol:
                a0 = newton(distance, a0, tol=tol, maxiter=maxiter)
            return (a0 - 1)*v0
        except RuntimeError:
            var[0, index, axis] = v0
            raise

    def aim_given(self, y, u, l=None, aim=None, **kwargs):
        if aim is not None:
            self.allocate(1)
            ya = y[None, aim] if y.shape[0] > 1 else y
            ua = u[None, aim] if u.shape[0] > 1 else u
            self.rays_given(ya, ua, l)
            try:
                corr = self.aim(0, **kwargs)
            except RuntimeError:
                corr = 0.
            if self.system.object.infinite:
                y += corr
            else:
                u += corr
        self.rays_given(y, u, l)
       
    def rays_paraxial_point(self, paraxial, height=(1., 0), wavelength=None,
            **kwargs):
        zp = paraxial.pupil_distance[0] + paraxial.z[1]
        rp = paraxial.pupil_height[0]
        self.rays_point(height, zp, rp, wavelength, **kwargs)

    def rays_point(self, object_height, pupil_distance,
            pupil_radius, wavelength, nrays=11,
            distribution="meridional", aim=True):
        # TODO apodization
        icenter, (xp, yp) = self.pupil_distribution(distribution, nrays)
        self.allocate(xp.shape[0])
        r = self.system.object.radius
        if self.system.object.infinite:
            r = sinarctan(r)
            p, q = object_height[0]*r, object_height[1]*r
            a = xp*pupil_radius-pupil_distance*tanarcsin(p)
            b = yp*pupil_radius-pupil_distance*tanarcsin(q)
            pq, ab = np.array([[p, q]]), np.c_[a, b]
        else:
            a, b = object_height[0]*r, object_height[1]*r
            p = sinarctan((xp*pupil_radius-a)/pupil_distance)
            q = sinarctan((yp*pupil_radius-b)/pupil_distance)
            ab, pq = np.array([[a, b]]), np.c_[p, q]
        self.aim_given(ab, pq, wavelength, aim=icenter if aim else None)
        self.propagate()

    def rays_for_object(self, paraxial, wavelength, nrays, eps=1e-6):
        hp, rp = paraxial.pupil_distance[0], paraxial.pupil_height[0]
        r = self.system.object.radius
        if self.system.object.infinity:
            r = sinarctan(r)
        xi, yi = np.tile([np.linspace(0, r, nrays), np.zeros(nrays)], 3)
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
        self.l = wavelength
        self.n[0] = self.system.object.material.refractive_index(
                wavelength)
        self.y[0, 0] = a
        self.y[1, 0] = b
        self.y[2, 0] = 0
        self.u[0, 0] = p
        self.u[1, 0] = q
        self.u[2, 0] = np.sqrt(1-p**2-q**2)

    @classmethod
    def transverse_plot(cls, fig, heights=[(0, 0), (.707, .707), (1., 1.)],
            wavelengths=None, paraxial=None,
            npoints_spot=100, npoints_line=30):
        pass

    def plot_transverse(self, fig, heights=[(0, 0), (.707, .707), (1., 1.)],
            wavelengths=None, paraxial=None,
            npoints_spot=100, npoints_line=30):
        fig.subplotpars.left = .05
        fig.subplotpars.bottom = .05
        fig.subplotpars.right = .95
        fig.subplotpars.top = .95
        fig.subplotpars.hspace = .2
        fig.subplotpars.wspace = .2
        if paraxial is None:
            paraxial = ParaxialTrace(self.system)
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

    def pupil_distribution(self, distribution, nrays):
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
            return 0, np.zeros(2, n)
        elif d == "meridional":
            return n/2, (np.linspace(-1, 1, n), np.zeros(n))
        elif d == "sagittal":
            return n/2, (np.zeros(n), np.linspace(-1, 1, n))
        elif d == "cross":
            return n/2, np.concatenate([
                [np.linspace(-1, 1, n/2), np.zeros(n/2)],
                [np.zeros(n/2), np.linspace(-1, 1, n/2)]], axis=1)
        elif d == "tee":
            return 0, np.concatenate([
                [np.zeros(n/3), np.linspace(0, 1, n/3)],
                [np.linspace(-1, 1, 2*n/3), np.zeros(2*n/3)]], axis=1)
        elif d == "random":
            r, phi = np.random.rand(2, n)
            xy = np.exp(2j*np.pi*phi)**2
            return 0, np.concatenate([
                np.zeros((2, 1)), np.array((xy.real, xy.imag))], axis=1)
        elif d == "square":
            r = int(np.sqrt(n*4/np.pi))
            x, y = np.mgrid[-1:1:1j*r, -1:1:1j*r]
            xy = np.array([x.ravel(), y.ravel()])
            return n/2, xy[:, (xy**2).sum(0)<=1]
        elif d == "triangular":
            r = int(np.sqrt(n*4/np.pi))
            x, y = np.mgrid[-1:1:1j*r, -1:1:1j*r]
            x += (np.arange(r) % 2.)/r
            xy = np.array([x.ravel(), y.ravel()])
            return n/2, xy[:, (xy**2).sum(0)<=1]
        elif d == "hexapolar":
            r = int(np.sqrt(n/3.-1/12.)-1/2.)
            l = [np.zeros((2, 1))]
            for i in np.arange(1, r+1)/r:
                a = np.arange(0, 2*np.pi, 2*np.pi/(6*i))
                l.append([np.sin(a)*i/r, i*np.cos(a)/r])
            return 0, np.concatenate(l, axis=1)

    def __str__(self):
        t = itertools.chain(
                #self.print_params(),
                self.print_trace(),
                #self.print_c3(),
                )
        return "\n".join(t)

    def print_trace(self):
        for i in range(self.nrays):
            yield "ray %i" % i
            c = np.concatenate((self.n[:, i, None], self.z[:, None],
                np.cumsum(self.t[:, i, None], axis=0)-self.z[:, None],
                self.y[:, i, :], self.u[:, i, :]), axis=1)
            for _ in self.print_coeffs(c, "n/track z/rel path/"
                    "height x/height y/height z/angle x/angle y/angle z"
                    .split("/"), sum=False):
                yield _
            yield ""

