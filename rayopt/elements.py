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
from scipy.optimize import newton

from .utils import public
from .transformations import (euler_matrix, euler_from_matrix, 
        rotation_matrix)
from .name_mixin import NameMixin
from .aberration_orders import aberration_intrinsic
from .material import Material


@public
class TransformMixin(object):
    def __init__(self, distance=0., direction=(0, 0, 1.), angles=(0, 0, 0)):
        self.update(distance, direction, angles)
        # offset = distance*direction: in lab system, relative to last
        # element (thus cumulative in lab system)
        # angles: relative to unit offset (-incidence angle)
        # excidence: excidence angles for axial ray (snell)

    def dict(self):
        dat = {}
        if self.distance:
            dat["distance"] = float(self.distance)
        if not self.straight:
            dat["direction"] = list(map(float, self.direction))
        if not self.normal:
            dat["angles"] = list(map(float, self.angles))
        return dat

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, offset):
        d = np.linalg.norm(offset)
        self.update(d, offset/d, self._angles)

    @property
    def angles(self):
        return self._angles

    @angles.setter
    def angles(self, angles):
        self.update(self._distance, self._direction, angles)

    @property
    def distance(self):
        return self._distance

    @distance.setter
    def distance(self, distance):
        self.update(distance, self._direction, self._angles)

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, direction):
        self.update(self._distance, direction, self._angles)

    @property
    def incidence(self):
        # of the optical axis onto the surface
        return self.to_normal(self._direction)

    def excidence(self, mu):
        i = self.incidence
        if mu == 1:
            return i
        r = 0, 0, 1
        a = abs(mu)*i[2]
        g = -a + np.sign(mu)*np.sqrt(a**2 - mu**2 + 1)
        e = abs(mu)*i + g[None]*r
        return e

    def align(self, direction, mu):
        """orient such that direction is excidence"""
        i = self.direction
        r = mu*i - direction # snell
        if mu < 1:
            r *= -1
        if np.allclose(r, 0):
            r = 0, 0, 1.
        r /= np.linalg.norm(r)
        rdir = np.cross(i, r)
        rang = np.arcsin(np.linalg.norm(rdir))
        if np.allclose(rdir, 0):
            rdir = 1., 0, 0
        rot = rotation_matrix(rang, rdir).T
        angles = euler_from_matrix(rot, "rxyz")
        self.update(self.distance, self.direction, angles)

    def update(self, distance, direction, angles):
        dlen = np.linalg.norm(direction)
        if not dlen:
            direction = 0, 0, 1.
        self._direction = u = np.array(direction)/dlen
        if distance < 0:
            distance *= -1
            u *= -1
        self.straight = np.allclose(u, (0, 0, 1.))
        self._distance = d = distance
        self._offset = o = d*u
        self._angles = a = np.array(angles)
        self.normal = np.allclose(a, 0.)
        self.rotated = not (self.normal and self.straight)
        if not self.rotated:
            self.rot_axis = self.rot_normal = None
            return
        r = np.eye(3)
        if not self.straight:
            rdir = np.cross(u, (0, 0, 1.))
            rang = np.arcsin(np.linalg.norm(rdir))
            if u[2] < 0: # == np.dot((0, 0, 1), u)
                rang = np.pi - rang
            if np.allclose(rdir, 0):
                rdir = 1., 0, 0
            self.rot_axis = r1 = rotation_matrix(rang, rdir)[:3, :3]
            r = np.dot(r, r1)
        if not self.normal:
            r1 = euler_matrix(axes="rxyz", *tuple(a))[:3, :3]
            r = np.dot(r, r1)
        self.rot_normal = r

    def _do_rotate(self, rotation, inverse, flag, y):
        if flag:
            if inverse:
                rotation = rotation.T
            y = tuple(np.dot(yi, rotation) for yi in y)
        if len(y) == 1:
            y = y[0]
        return y

    def from_axis(self, *y):
        return self._do_rotate(self.rot_axis, False, not self.straight, y)

    def to_axis(self, *y):
        return self._do_rotate(self.rot_axis, True, not self.straight, y)

    def from_normal(self, *y):
        return self._do_rotate(self.rot_normal, False, self.rotated, y)

    def to_normal(self, *y):
        return self._do_rotate(self.rot_normal, True, self.rotated, y)


@public
class Element(NameMixin, TransformMixin):
    _default_type = "spheroid"

    def __init__(self, radius=np.inf, diameter=None, **kwargs):
        super(Element, self).__init__(**kwargs)
        if diameter is not None:
            radius = diameter/2
        self.radius = radius

    def dict(self):
        dat = NameMixin.dict(self)
        dat.update(TransformMixin.dict(self))
        if np.isfinite(self.radius):
            dat["radius"] = float(self.radius)
        return dat

    def intercept(self, y, u):
        # ray length to intersection with element
        # only reference plane, overridden in subclasses
        # solution for z=0
        s = -y[:, 2]/u[:, 2]
        # given angles is set correctly, mask s
        return s

    def clip(self, y, u):
        good = np.square(y[:, :2]).sum(1) <= self.radius**2
        u = np.where(good[:, None], u, np.nan)
        return u

    def propagate_paraxial(self, yu0, n0, l):
        n, m = self.paraxial_matrix(n0, l)
        yu = np.dot(yu0, m.T)
        return yu, n

    def propagate_gaussian(self, q0i, n0, l):
        n, m = self.paraxial_matrix(n0, l)
        a, b, c, d = m[:2, :2], m[:2, 2:], m[2:, :2], m[2:, 2:]
        qi = np.dot(c + np.dot(d, q0i), np.linalg.inv(a + np.dot(b, q0i)))
        return qi, n

    def paraxial_matrix(self, n0, l):
        m = np.eye(4)
        d = self.distance
        m[0, 2] = m[1, 3] = d
        return n0, m

    def propagate(self, y0, u0, n0, l):
        t = self.intercept(y0, u0)
        y = y0 + t[:, None]*u0
        if clip:
            u0 = self.clip(y, u0)
        n = n0
        return y, u0, n, t*n0

    def reverse(self):
        pass

    def rescale(self, scale):
        self.distance *= scale
        self.radius *= scale

    def surface_cut(self, axis, points):
        rad = self.radius
        xyz = np.zeros((2, 3))
        xyz[:, axis] = -rad, rad
        return xyz

    def aberration(self, *args):
        return 0

    def dispersion(self, *args):
        return 0


@public
class Interface(Element):
    def __init__(self, material=None, **kwargs):
        super(Interface, self).__init__(**kwargs)
        if material:
            material = Material.make(material)
        self.material = material

    def dict(self):
        dat = super(Interface, self).dict()
        if self.material is not None:
            dat["material"] = str(self.material)
        return dat

    def refractive_index(self, wavelength):
        return abs(self.material.refractive_index(wavelength))

    def paraxial_matrix(self, n0, l):
        n, m = super(Interface, self).paraxial_matrix(n0, l)
        if self.material is not None:
            n = self.refractive_index(l)
        return n, m

    def refract(self, y, u0, mu):
        return u0

    def propagate(self, y0, u0, n0, l, clip=True):
        t = self.intercept(y0, u0)
        y = y0 + t[:, None]*u0
        if clip:
            u0 = self.clip(y, u0)
        u, n = u0, n0
        if self.material is not None:
            if self.material.mirror:
                mu = -1.
            else:
                n = self.refractive_index(l)
                mu = n0/n
            u = self.refract(y, u0, mu)
        return y, u, n, t*n0

    def dispersion(self, lmin, lmax):
        if self.material is not None:
            return self.material.delta_n(lmin, lmax)
        return 0.

    def surface_sag(self, p):
        raise NotImplementedError

    def surface_normal(self, p):
        raise NotImplementedError

    def edge_sag(self, axis=1):
        r = np.zeros(3)
        r[axis] = self.radius
        return self.surface_sag(r)

    def intercept(self, y, u):
        s = super(Interface, self).intercept(y, u)
        for i in range(y.shape[0]):
            yi, ui = y[None, i], u[None, i]
            def func(si):
                return self.surface_sag(yi + si*ui)[0]
            def fprime(si):
                return np.dot(self.surface_normal(yi + si*ui), ui.T)[0]
            try:
                s[i] = newton(func=func, fprime=fprime, x0=s[i],
                        tol=1e-7, maxiter=5)
            except RuntimeError:
                s[i] = np.nan
        return s

    def refract(self, y, u0, mu):
        # G. H. Spencer and M. V. R. K. Murty
        # General Ray-Tracing Procedure
        # JOSA, Vol. 52, Issue 6, pp. 672-676 (1962)
        # doi:10.1364/JOSA.52.000672
        if mu == 1:
            return u0
        r = self.surface_normal(y)
        r2 = np.square(r).sum(1)
        muf = abs(mu)
        a = muf*(u0*r).sum(1)/r2
        # solve g**2 + 2*a*g + b=0
        if mu == -1:
            u = u0 - 2*a[:, None]*r # reflection
        else:
            b = (mu**2 - 1)/r2
            g = -a + np.sign(mu)*np.sqrt(np.square(a) - b)
            u = muf*u0 + g[:, None]*r # refraction
        return u

    def surface_cut(self, axis, points):
        if self.material is None:
            return super(Interface, self).surface_cut(axis, points)
        rad = self.radius
        xyz = np.zeros((points, 3))
        xyz[:, axis] = np.linspace(-rad, rad, points)
        xyz[:, 2] = -self.surface_sag(xyz)
        return xyz


@public
@Element.register
class Spheroid(Interface):
    def __init__(self, curvature=0., conic=1., aspherics=None, roc=None,
            **kwargs):
        super(Spheroid, self).__init__(**kwargs)
        if roc is not None:
            curvature = 1./roc
        self.curvature = curvature
        self.conic = conic
        if aspherics is not None:
            aspherics = list(aspherics)
        self.aspherics = aspherics
        if self.curvature and np.isfinite(self.radius):
            assert self.radius**2 <= 1/(self.conic*self.curvature**2)

    def dict(self):
        dat = super(Spheroid, self).dict()
        if self.curvature:
            dat["curvature"] = float(self.curvature)
        if self.conic != 1.:
            dat["conic"] = float(self.conic)
        if self.aspherics is not None:
            dat["aspherics"] = list(map(float, self.aspherics))
        return dat

    def surface_sag(self, xyz):
        x, y, z = xyz.T
        if not self.curvature:
            return z
        r2 = x**2 + y**2
        c, k = self.curvature, self.conic
        e = c*r2/(1 + np.sqrt(1 - k*c**2*r2))
        if self.aspherics is not None:
            for i, ai in enumerate(self.aspherics):
                e += ai*r2**(i + 2)
        return z - e

    def surface_normal(self, xyz):
        x, y, z = xyz.T
        q = np.ones_like(xyz)
        if not self.curvature:
            q[:, :2] = 0
            return q
        r2 = x**2 + y**2
        c, k = self.curvature, self.conic
        e = c/np.sqrt(1 - k*c**2*r2)
        if self.aspherics is not None:
            for i, ai in enumerate(self.aspherics):
                e += 2*ai*(i + 2)*r2**(i + 1)
        q[:, 0] = -x*e
        q[:, 1] = -y*e
        return q

    def intercept(self, y, u):
        if self.aspherics is not None:
            return Interface.intercept(self, y, u) # expensive iterative
        # replace the newton-raphson with the analytic solution
        c, k = self.curvature, self.conic
        if c == 0:
            return -y[:, 2]/u[:, 2] # flat
        ky, ku = y, u
        if k != 1:
            ky, ku = ky.copy(), ku.copy()
            ky[:, 2] *= k
            ku[:, 2] *= k
        d = c*(u*ky).sum(1) - u[:, 2]
        e = c*(u*ku).sum(1)
        f = c*(y*ky).sum(1) - 2*y[:, 2]
        s = -(d + np.sign(u[:, 2])*np.sqrt(d**2 - e*f))/e
        return s #np.where(s*np.sign(self.distance)>=0, s, np.nan)

    def paraxial_matrix(self, n0, l):
        # [y', u'] = M * [y, u]
        c = self.curvature
        if self.aspherics is not None:
            c += 2*self.aspherics[0]
        d = self.distance
        md = np.eye(4)
        md[0, 2] = md[1, 3] = d

        # FIXME angles is incomplete:
        # rotate to meridional/sagittal then compute total incidence
        # angle, matrix, then rotate back
        theta = self.angles[0] if self.angles is not None else 0.
        costheta = np.cos(theta)
        n = n0
        m = np.eye(4)
        if self.material is not None:
            if self.material.mirror:
                m[2, 0] = 2*c*costheta
                m[3, 1] = 2*c/costheta
                m = -m
            else:
                n = self.refractive_index(l)
                mu = n/n0
                p = np.sqrt(mu**2 + costheta**2 - 1)
                m[1, 1] = p/(mu*costheta)
                m[2, 0] = c*(costheta - p)/mu
                m[3, 1] = c*(costheta - p)/(costheta*p)
                m[2, 2] = 1./mu
                m[3, 3] = costheta/p
        m = np.dot(m, md)

        if self.angles is not None:
            phi = self.angles[2]
            cphi, sphi = np.cos(phi), np.sin(phi)
            r1 = np.array([[cphi, -sphi], [sphi, -cphi]])
            r = np.eye(4)
            r[:2, :2] = r[2:, 2:] = r1
            m = np.dot(r, np.dot(m, r.T))

        return n, m
   
    def reverse(self):
        super(Spheroid, self).reverse()
        self.curvature *= -1
        if self.aspherics is not None:
            self.aspherics = [-ai for ai in self.aspherics]

    def rescale(self, scale):
        super(Spheroid, self).rescale(scale)
        self.curvature /= scale
        if self.aspherics is not None:
            self.aspherics = [ai/scale**(2*i + 3) for i, ai in
                    enumerate(self.aspherics)]

    def aberration(self, y, u, n0, n, kmax):
        y, yb = y
        u, ub = u
        c = self.curvature
        f, g = (c*y + u)*n0, (c*yb + ub)*n0
        if self.material is not None and self.material.mirror:
            n = -n # FIXME appear incorrect
        a = np.zeros((2, 2, kmax, kmax, kmax))
        aberration_intrinsic(c, f, g, y, yb, 1/n0, 1/n, a, kmax - 1)
        return a


try:
    # the numba version is three times faster for nrays=3 but ten times
    # slower for nrays=1000...
    raise ImportError
    from .numba_elements import fast_propagate
    _Spheroid = Spheroid
    @public
    @Element.register
    class Spheroid(_Spheroid):
        def propagate(self, y0, u0, n0, l, clip=True):
            m = y0.shape[0]
            y = np.empty((m, 3))
            u = np.empty((m, 3))
            t = np.empty((m,))
            n = fast_propagate(self, y0, u0, n0, l, clip, y, u, t)
            return y, u, n, t
except ImportError:
    pass
