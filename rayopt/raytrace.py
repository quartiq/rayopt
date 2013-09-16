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
from .aberration_orders import aberration_extrinsic


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
    def __init__(self, system, aberration_orders=3):
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

    def propagate(self, start=0, stop=None, aberration=True):
        self.z = np.cumsum([e.thickness for e in self.system])
        init = start - 1 if start else 0
        yu, n = np.array((self.y[init], self.u[init])).T, self.n[init]
        els = self.system[start:stop or self.length]
        for i, el in enumerate(els):
            yu, n = el.propagate_paraxial(yu, n, self.l)
            (self.y[i], self.u[i]), self.n[i] = yu.T, n
            self.v[i] = el.dispersion(self.lmin, self.lmax)
            if aberration and i > 0:
                self.c[i] = el.aberration(self.y[i], self.u[i - 1],
                        self.n[i - 1], self.n[i], self.c.shape[-1])
        if aberration:
            self.extrinsic_aberrations()

    def extrinsic_aberrations(self): # FIXME: wrong
        self.d[:] = 0
        st = self.system.aperture_index
        t, s = 0, 1
        kmax = self.d.shape[-1]
        r = np.zeros((self.length, 2, 2, kmax, kmax, kmax))
        for ki in range(2, kmax):
            k = ki - 1
            for j in range(k + 1):
                for i in range(k - j + 1):
                    b = (self.c[:, :, :, k - j - i, j, i]
                       + self.d[:, :, :, k - j - i, j, i])
                    b = np.cumsum(b, axis=0)
                    r[:, t, :, k - j - i, j, i] = b[:, t]
                    r[:, s, :, k - j - i, j, i] = b[:, s] - b[(st,), s]
            for i in range(self.length):
                aberration_extrinsic(self.c[i], r[i], self.d[i], ki)

    def find_rays(self):
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
        self.system.image.radius = abs(self.height[1])
        self.system.size_convex()

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
        return 1.22*self.l/(2*na)/self.system.scale

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
        self.system.size_convex()

    def plot(self, ax, axis=0, **kwargs):
        kwargs.setdefault("linestyle", "-")
        kwargs.setdefault("color", "green")
        kwargs.setdefault("alpha", .3)
        y = self.y[:, :, axis]
        z = self.y[:, :, 2] + self.z[:, None]
        ax.plot(z, y, **kwargs)

    def rays_given(self, y, u, l=None):
        y, u = np.atleast_2d(y, u)
        if l is None:
            l = self.system.object.wavelengths[0]
        self.allocate(max(y.shape[0], u.shape[0]))
        self.l = l
        self.y[0, :, :] = 0
        self.y[0, :, :y.shape[1]] = y
        self.u[0, :, :] = 0
        self.u[0, :, :u.shape[1]] = u
        self.u[0, :, 2] = np.sqrt(1 - np.square(self.u[0, :, :2]).sum(1))

    def rays_paraxial(self, paraxial):
        y = paraxial.y[0, :, None]
        u = paraxial.u[0, :, None]
        self.rays_given(y, sinarctan(u))
        self.propagate(clip=False)

    def aim(self, index=0, axis=0, tol=1e-2, maxiter=10):
        """aims ray by index at aperture center
        changing angle (in case of finite object) or
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

    def aim_given(self, y, u, l=None, aim=None, axis=0, **kwargs):
        if aim is not None:
            self.allocate(1)
            ya = y[(aim,), :] if y.shape[0] > 1 else y
            ua = u[(aim,), :] if u.shape[0] > 1 else u
            self.rays_given(ya, ua, l)
            try:
                corr = self.aim(index=0, axis=axis, **kwargs)
            except RuntimeError:
                corr = 0.
            if self.system.object.infinite:
                y[:, axis] += corr
            else:
                u[:, axis] += corr
        self.rays_given(y, u, l)
       
    def rays_paraxial_point(self, paraxial, height=(1., 0.),
            wavelength=None, **kwargs):
        zp = paraxial.pupil_distance[0] + paraxial.z[1]
        rp = paraxial.pupil_height[0]
        return self.rays_point(height, zp, rp, wavelength, **kwargs)

    def rays_point(self, object_height, pupil_distance, pupil_radius,
            wavelength, nrays=11, distribution="meridional",
            clip=True, aim=True):
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
            a, b = -object_height[0]*r, -object_height[1]*r
            p = sinarctan((xp*pupil_radius-a)/pupil_distance)
            q = sinarctan((yp*pupil_radius-b)/pupil_distance)
            ab, pq = np.array([[a, b]]), np.c_[p, q]
        self.aim_given(ab, pq, wavelength, aim=icenter if aim else None)
        self.propagate(clip=clip)
        return icenter

    def rays_line(self, paraxial, wavelength, nrays, eps=1e-6):
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

    def setup_fanplot(self, fig, n, adjust=True):
        from matplotlib import gridspec
        if adjust:
            fig.subplotpars.left = .05
            fig.subplotpars.bottom = .05
            fig.subplotpars.right = .95
            fig.subplotpars.top = .95
            fig.subplotpars.hspace = .2
            fig.subplotpars.wspace = .2
        gs = gridspec.GridSpec(n, 5)
        axpx0, axpy0, axex0, axey0, axpx1 = None, None, None, None, None
        ax = []
        for i in range(n):
            axp = fig.add_subplot(gs.new_subplotspec((i, 0), 1, 1),
                    aspect="equal", sharex=axex0, sharey=axey0)
            axex0, axey0 = axex0 or axp, axey0 or axp
            axm = fig.add_subplot(gs.new_subplotspec((i, 1), 1, 2),
                    sharex=axpy0, sharey=axey0)
            axpy0 = axpy0 or axm
            axsm = fig.add_subplot(gs.new_subplotspec((i, 3), 1, 1),
                    sharex=axpx0, sharey=axey0)
            axpx0 = axpx0 or axsm
            axss = fig.add_subplot(gs.new_subplotspec((i, 4), 1, 1),
                    sharex=axex0, sharey=axpx1)
            axpx1 = axpx1 or axss
            ax.append((axp, axm, axsm, axss))
            kw = dict(rotation="horizontal",
                    horizontalalignment="left",
                    verticalalignment="bottom")
            for axi, xl, yl in [
                    (axp, "EX", "EY"),
                    (axm, "PY", "EY"),
                    (axsm, "PX", "EY"),
                    (axss, "EX", "PX"),
                    ]:
                axi.spines["right"].set_visible(False)
                axi.spines["top"].set_visible(False)
                axi.spines["left"].set_position("zero")
                axi.spines["bottom"].set_position("zero")
                axi.tick_params(bottom=True, top=False,
                        left=True, right=False,
                        labeltop=False, labelright=False,
                        labelleft=True, labelbottom=True,
                        direction="out", axis="both")
                axi.xaxis.set_smart_bounds(True)
                axi.yaxis.set_smart_bounds(True)
                #axi.set_title("h=%s, %s" % hi)
                axi.set_xlabel(xl, **kw)
                axi.set_ylabel(yl, **kw)
            axp.spines["left"].set_visible(False)
            axp.spines["bottom"].set_visible(False)
            axp.tick_params(bottom=False, left=False,
                    labelbottom=False, labelleft=False)
            axp.set_aspect("equal")
        return ax

    def tweak_fanplots(self, ax):
        for axp, axm, axsm, axss in ax:
            for axi in axm, axsm, axss, axp:
                axi.relim()
                #axi.autoscale_view(True, True, True)
                t = axi.get_xticks()
                axi.set_xticks((t[0], t[-1]))
                t = axi.get_yticks()
                axi.set_yticks((t[0], t[-1]))
                xl, xu = axi.get_xlim()
                yl, yu = axi.get_ylim()
                axi.xaxis.set_label_coords(xu, .02*(yu - yl),
                        transform=axi.transData)
                axi.yaxis.set_label_coords(.02*(xu - xl), yu,
                        transform=axi.transData)

    def plot_transverse(self, fig, paraxial=None,
            heights=[(1., 0.), (.707, .0), (0., 0.)],
            wavelengths=None, nrays_spot=100, nrays_line=23,
            adjust=True, colors="gbrcmyk"):
        import matplotlib.patches as patches
        if paraxial is None:
            paraxial = ParaxialTrace(self.system)
        if wavelengths is None:
            wavelengths = self.system.object.wavelengths
        nh = len(heights)
        ia = self.system.aperture_index
        ax = self.setup_fanplot(fig, nh, adjust)
        p = paraxial.pupil_distance[0]
        r = paraxial.airy_radius[1]
        for hi, axi in zip(heights, ax):
            axp, axm, axsm, axss = axi
            axp.add_patch(patches.Circle((0, 0), r, edgecolor="black",
                facecolor="none"))
            axp.set_title("OY,OX=%s,%s" % hi)
            for wi, ci in zip(wavelengths, colors):
                ref = self.rays_paraxial_point(paraxial, hi, wi,
                        nrays=nrays_spot, distribution="triangular")
                exy = self.y[-1, :, :2]
                exy = exy - exy[ref]
                axp.plot(exy[:, 1], exy[:, 0],
                        ".%s" % ci, markersize=3, markeredgewidth=0, label="%s" % wi)
                ref = self.rays_paraxial_point(paraxial, hi, wi,
                        nrays=nrays_line, distribution="tee")
                y, u = self.y, self.u
                exy = y[-1, :, :2]
                exy = exy - exy[ref]
                pxy = y[1, :, :2] + p*u[0, :, :2]/u[0, :, 2:]
                pxy = pxy - pxy[ref]
                axm.plot(pxy[:ref, 0], exy[:ref, 0], "-%s" % ci, label="%s" % wi)
                axsm.plot(pxy[ref:, 1], exy[ref:, 0], "-%s" % ci, label="%s" % wi)
                axss.plot(exy[ref:, 1], pxy[ref:, 1], "-%s" % ci, label="%s" % wi)
        self.tweak_fanplots(ax)

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
            n -= n % 2
            return n/2, (np.linspace(-1, 1, n + 1), np.zeros(n + 1))
        elif d == "sagittal":
            n -= n % 2
            return n/2, (np.zeros(n + 1), np.linspace(-1, 1, n + 1))
        elif d == "cross":
            n -= n % 4
            return n/4, np.concatenate([
                [np.linspace(-1, 1, n/2 + 1), np.zeros(n/2 + 1)],
                [np.zeros(n/2 + 1), np.linspace(-1, 1, n/2 + 1)]], axis=1)
        elif d == "tee":
            n = (n - 2)/3
            return 2*n + 1, np.concatenate([
                [np.linspace(-1, 1, 2*n + 1), np.zeros(2*n + 1)],
                [np.zeros(n + 1), np.linspace(0, 1, n + 1)],
                ], axis=1)
        elif d == "random":
            r, phi = np.random.rand(2, n)
            xy = np.exp(2j*np.pi*phi)**2
            xy = [xy.real, xy.imag]
            return 0, np.concatenate([np.zeros((2, 1)), xy], axis=1)
        elif d == "square":
            r = int(np.sqrt(n*4/np.pi))
            xy = np.mgrid[-1:1:1j*r, -1:1:1j*r].reshape(2, -1)
            xy = xy[:, (xy**2).sum(0)<=1]
            return 0, np.concatenate([np.zeros((2, 1)), xy], axis=1)
        elif d == "triangular":
            r = int(np.sqrt(n*4/np.pi))
            xy = np.mgrid[-1:1:1j*r, -1:1:1j*r]
            xy[0] += (np.arange(r) % 2.)*(2./r)
            xy = xy.reshape(2, -1)
            xy = xy[:, (xy**2).sum(0)<=1]
            return 0, np.concatenate([np.zeros((2, 1)), xy], axis=1)
        elif d == "hexapolar":
            r = int(np.sqrt(n/3.-1/12.)-1/2.)
            l = [np.zeros((2, 1))]
            for i in np.arange(1, r + 1)/r:
                a = np.arange(0, 2*np.pi, 2*np.pi/(6*i))
                l.append([np.sin(a)*i/r, np.cos(a)*i/r])
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

