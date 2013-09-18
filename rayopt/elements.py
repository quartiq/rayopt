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

import numpy as np
from scipy.optimize import newton

from .transformations import (euler_matrix, translation_matrix,
        concatenate_transforms)
from .name_mixin import NameMixin
from .aberration_orders import aberration_intrinsic
from .utils import sinarctan, tanarcsin


class TransformMixin(object):
    def __init__(self, offset=None, angles=None):
        self.offset = offset or np.zeros(3)
        self.angles = angles or np.zeros(3)
        self.update()

    def update(self):
        self.rotation = euler_matrix(axes="rxyz", *self.angles)
        self.inverse_rotation = np.linalg.inv(self.rotation)
        translation = translation_matrix(self.offset)
        self.transformation = concatenate_transforms(translation,
                self.rotation)
        self.inverse_transformation = np.linalg.inv(self.transformation)
        self.at_origin = (np.allclose(self.angles, 0) and
                np.allclose(self.offset, 0))

    def transform_to(self, y, angle=False):
        y = np.atleast_2d(y)
        if self.at_origin:
            return y
        if y.shape[1] == 4:
            y = np.dot(y, self.inverse_transformation.T)
        else:
            if not angle:
                y = y - self.offset
            y = np.dot(y, self.inverse_rotation.T[:3, :3])
        return y

    def transform_from(self, y, angle=False):
        y = np.atleast_2d(y)
        if self.at_origin:
            return y
        if y.shape[1] == 4:
            y = np.dot(y, self.transformation.T)
        else:
            y = np.dot(y, self.rotation.T[:3, :3])
            if not angle:
                y += self.offset
        return y

    def transformed_yu(self, fun, y0, u0, *args, **kwargs):
        y0 = self.transform_to(y0)
        u0 = self.transform_to(u0, angle=True)
        ret = fun(y0, u0, *args, **kwargs)
        y, u = ret[:2]
        y = self.transform_from(y)
        u = self.transform_from(u, angle=True)
        return (y, u) + ret[2:]


class Primitive(NameMixin, TransformMixin):
    typ = "P"

    def __init__(self, thickness=0., radius=np.inf, **kwargs):
        super(Primitive, self).__init__(**kwargs)
        self.radius = radius
        self.thickness = thickness

    def intercept(self, y, u):
        # ray length to intersection with element
        # only reference plane, overridden in subclasses
        # solution for z=0
        s = -y[:, 2]/u[:, 2]
        # given angles is set correctly, mask s
        return s

    def clip(self, y, u):
        r2 = y[:, 0]**2 + y[:, 1]**2
        u[:, 2] = np.where(r2 > self.radius**2, np.nan, u[:, 2])

    def propagate_paraxial(self, yu0, n0, l):
        n, m = self.paraxial_matrix(n0, l)
        yu0 = np.atleast_2d(yu0)
        yu = np.dot(m, yu0.T).T
        return yu, n

    def paraxial_matrix(self, n0, l):
        d = self.thickness
        return n0, np.array([[1., d], [0, 1]])

    def propagate(self, y0, u0, n0, l, clip=True):
        # length up to surface
        y = y0 - [0, 0, self.thickness]
        t = self.intercept(y, u0)
        # new transverse position
        y += t[:, None]*u0
        u = u0
        if clip:
            u = u.copy()
            self.clip(y, u)
        return y, u, n0, t*n0

    def reverse(self):
        self.offset[1:] *= -1
        self.update()

    def rescale(self, scale):
        self.offset *= scale
        self.update()
        self.thickness *= scale
        self.radius *= scale

    def surface_cut(self, axis, points):
        rad = self.radius
        if not np.isfinite(rad):
            rad = 1.
        xyz = np.zeros((points, 3))
        xyz[:, axis] = np.linspace(-rad, rad, points)
        return xyz

    def aberration(self, *args):
        return 0

    def dispersion(self, *args):
        return 0


class Element(Primitive):
    typ = "E"

    def __init__(self, material=None, **kwargs):
        super(Element, self).__init__(**kwargs)
        self.material = material

    def paraxial_matrix(self, n0, l):
        n, m = super(Element, self).paraxial_matrix(n0, l)
        if self.material is not None:
            n = self.material.refractive_index(l)
        return n, m

    def refract(self, y, u0, mu):
        return u0

    def propagate(self, y0, u0, n0, l, clip=True):
        y, u, n, t = super(Element, self).propagate(y0, u0, n0, l,
                clip)
        if self.material is not None:
            n = self.material.refractive_index(l)
            u = self.refract(y, u, n0/n)
        return y, u, n, t*n0

    def dispersion(self, lmin, lmax):
        v = 0
        if self.material is not None:
            v = self.material.delta_n(lmin, lmax)
        return v


class Interface(Element):
    typ = "F"

    def shape_func(self, p):
        raise NotImplementedError

    def shape_func_deriv(self, p):
        raise NotImplementedError

    def intercept(self, y, u):
        s = np.zeros(y.shape[0])
        s0 = super(Interface, self).intercept(y, u)
        for i in range(y.shape[0]):
            yi, ui, s0i = y[i], u[i], s0[i]
            def func(si): return self.shape_func(yi + si*ui)
            def fprime(si): return np.dot(
                    self.shape_func_deriv(yi + si*ui).T, ui)
            try:
                s[i] = newton(func=func, fprime=fprime, x0=s0i,
                        tol=1e-7, maxiter=5)
            except RuntimeError:
                s[i] = np.nan
        return s # np.where(s>=0, s, np.nan) # TODO mask

    def refract(self, y, u0, mu):
        # General Ray-Tracing Procedure
        # G. H. Spencer and M. V. R. K. Murty
        # JOSA, Vol. 52, Issue 6, pp. 672-676 (1962)
        # doi:10.1364/JOSA.52.000672
        if np.all(mu == 1): # all/any?
            return u0
        r = self.shape_func_deriv(y)
        r2 = np.square(r).sum(axis=1)
        muf = np.fabs(mu)
        a = muf*(u0*r).sum(axis=1)/r2
        # solve g**2 + 2*a*g + b=0
        if np.all(mu == -1): # all/any?
            u = u0 - 2*a[:, None]*r # reflection
        else:
            b = (mu**2 - 1)/r2
            g = -a + np.sign(mu)*np.sqrt(a**2 - b)
            u = muf*u0 + g[:, None]*r # refraction
        return u

    def surface_cut(self, axis, points):
        xyz = super(Interface, self).surface_cut(axis, points)
        xyz[:, 2] = -self.shape_func(xyz)
        return xyz


class Spheroid(Interface):
    typ = "S"

    def __init__(self, curvature=0., conic=1., aspherics=[], **kwargs):
        super(Spheroid, self).__init__(**kwargs)
        self.curvature = curvature
        self.conic = conic
        self.aspherics = np.array(aspherics)
        #if self.curvature and self.radius:
        #    assert self.radius**2 < 1/(self.conic*self.curvature**2)

    def shape_func(self, p):
        x, y, z = p.T
        r2 = x**2 + y**2
        o = np.sum([ai*r2**(i + 2) for i, ai in 
            enumerate(self.aspherics)], axis=0)
        c, k = self.curvature, self.conic
        return z - c*r2/(1 + np.sqrt(1 - k*c**2*r2)) - o

    def shape_func_deriv(self, p):
        x, y, z = p.T
        r2 = x**2 + y**2
        o = 2*np.sum([ai*(i + 2)*r2**(i + 1) for i, ai in 
            enumerate(self.aspherics)], axis=0)
        c, k = self.curvature, self.conic
        e = c/np.sqrt(1 - k*c**2*r2) + o
        return np.array([-x*e, -y*e, np.ones_like(e)]).T

    def intercept(self, y, u):
        if not self.aspherics.size:
            # replace the newton-raphson with the analytic solution
            if self.curvature == 0:
                s = Element.intercept(self, y, u) # flat
            else:
                k = np.array([1., 1., self.conic])[None, :]
                ky = k*y
                ku = k*u
                c = self.curvature
                d = c*(u*ky).sum(axis=1) - u[:, 2]
                e = c*(u*ku).sum(axis=1)
                f = c*(y*ky).sum(axis=1) - 2*y[:, 2]
                s = (-d - np.sign(u[:, 2])*np.sqrt(d**2 - e*f))/e
        else:
            s = super(Spheroid, self).intercept(y, u)
        return s #np.where(s*np.sign(self.thickness)>=0, s, np.nan)

    def paraxial_matrix(self, n0, l):
        # [y', u'] = M * [y, u]
        c = self.curvature
        if self.aspherics.size:
            c += 2*self.aspherics[0]
        n = self.material.refractive_index(l)
        mu = n0/n
        d = self.thickness
        p = c*(mu - 1)
        return n, np.array([[1, d], [p, d*p + mu]])
   
    def reverse(self):
        super(Spheroid, self).reverse()
        self.curvature *= -1
        self.aspherics *= -1

    def rescale(self, scale):
        super(Spheroid, self).rescale(scale)
        self.curvature /= scale
        self.aspherics /= scale**(2*np.arange(self.aspherics.size) + 1)

    def aberration(self, y, u, n0, n, kmax):
        y, yb = y
        u, ub = u
        f, g = (self.curvature*y + u)*n0, (self.curvature*yb + ub)*n0
        c = np.zeros((2, 2, kmax, kmax, kmax))
        aberration_intrinsic(self.curvature, f, g, y, yb, 1/n0, 1/n,
                c, kmax - 1)
        return c


class Object(Element):
    typ = "O"

    def __init__(self, wavelengths=[588e-9], infinite=None, **kwargs):
        super(Object, self).__init__(**kwargs)
        self.wavelengths = wavelengths
        self.infinite = infinite

    def rescale(self, scale):
        super(Object, self).rescale(scale)
        if self.infinite: # undo
            self.radius /= scale

    def paraxial_matrix(self, n0, l):
        n = self.material.refractive_index(l)
        return n, np.eye(2)
    
    def propagate(self, y0, u0, n0, l, clip=True):
        n = self.material.refractive_index(l)
        t = np.zeros_like(l)
        return y0, u0, n, t

    def to_pupil(self, height, pupil, distance, radius):
        yo, yp = np.broadcast_arrays(height, pupil)
        r = self.radius
        if self.infinite:
            u = sinarctan(yo*r)
            y = yp*radius - distance*tanarcsin(u)
        else:
            y = -yo*r
            u = sinarctan((yp*radius - y)/distance)
        return y, u


class Aperture(Primitive):
    typ = "A"

    def surface_cut(self, axis, points):
        r = self.radius
        if not np.isfinite(r):
            r = 1.
        xyz = np.zeros((5, 3))
        xyz[:, axis] = np.array([-r*1.5, -r, np.nan, r, r*1.5])
        return xyz


class Image(Primitive):
    typ = "I"
