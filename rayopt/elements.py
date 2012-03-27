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
from scipy.optimize import (newton, fsolve)

from traits.api import (HasTraits, Float, Array,
        Trait, cached_property, Property, Bool)

from .transformations import euler_matrix, translation_matrix
from .material import Material, air


class Element(HasTraits):
    typestr = "E"
    origin = Array(dtype=np.float64, shape=(3,))
    angles = Array(dtype=np.float64, shape=(3,))
    rotation = Property(depends_on="angles")
    transform = Property(depends_on="origin, angles")
    inverse_transform = Property(depends_on="origin, angles")
    material = Trait(air, Material)
    radius = Float

    @cached_property
    def _get_rotation(self):
        return euler_matrix(axes="rxyz", *-self.angles)

    @cached_property
    def _get_transform(self):
        t = translation_matrix(-self.origin)
        return np.dot(self.rotation,t)

    @cached_property
    def _get_inverse_transform(self):
        return np.linalg.inv(self.transform)

    def intercept(self, y, u):
        # ray length to intersection with element
        # only reference plane, overridden in subclasses
        # solution for z=0
        s = np.where(y[2]==0, 0., -y[2]/u[2])
        return s # TODO mask, np.where(s>=0, s, np.nan)

    def propagate(self, r, j):
        y0, u0, n0 = r.y[:, j-1], r.u[:, j-1], r.n[j-1]
        y = np.dot(self.rotation[:3, :3], y0-self.origin[:, None])
        u = np.dot(self.rotation[:3, :3], u0)
        # length up to surface
        r.p[j-1] = self.intercept(y, u)
        # new transverse position
        r.y[:, j] = y + r.p[None, j-1]*u
        if self.material is None:
            r.n[j] = r.n[j-1]
            r.u[:, j] = r.u[:, j-1]
        else:
            r.n[j] = map(self.material.refractive_index, r.l)
            r.u[:, j] = self.refract(r.y[:, j], u, r.n[j-1]/r.n[j])
        # fix origin

    def refract(self, y, u, mu):
        return u
  
    def propagate_paraxial(self, r, j):
        y0, u0, n0 = r.y[0, j-1], r.u[0, j-1], r.n[j-1]
        t = self.origin[2]
        n = map(self.material.refractive_index, r.l)
        y = y0+t*u0 # propagate
        i = u0 # incident
        u = u0
        r.y[0, j] = y
        r.y[2, j] = r.y[2, j-1]+t
        r.i[0, j] = i
        r.u[0, j] = u
        r.n[j] = n
        r.v[j] = [self.material.delta_n(l1, l2) for l1, l2 in zip(r.l1, r.l2)]
 
    def aberration3(self, r, j):
        r.c3[:, j] = 0

    def aberration5(self, r, j):
        r.c3[:, j] = 0

    def reverse(self):
        pass

    def surface(self, axis, points=20):
        t = np.array([-self.radius, self.radius])
        xyz = np.zeros((3, 2))
        xyz[axis] = t
        return xyz[axis]+self.origin[axis], xyz[2]+self.origin[2]


class Interface(Element):
    typestr = "F"

    def shape_func(self, p):
        raise NotImplementedError

    def shape_func_deriv(self, p):
        raise NotImplementedError

    def intercept(self, y, u):
        s = np.zeros((y.shape[1],))
        for i in range(y.shape[1]):
            try:
                yi, ui = y[:, i], u[:, i]
                s[i] = newton(
                        func=lambda si: self.shape_func(yi+si*ui),
                        fprime=lambda si: np.dot(
                            self.shape_func_deriv(yi+si*ui), ui),
                        x0=-yi[2]/ui[2], tol=1e-7, maxiter=15)
            except RuntimeError:
                s[i] = np.nan
        return s # np.where(s>=0, s, np.nan) # TODO mask

    def refract(self, y, u, mu):
        # General Ray-Tracing Procedure
        # G. H. Spencer and M. V. R. K. Murty
        # JOSA, Vol. 52, Issue 6, pp. 672-676 (1962)
        # doi:10.1364/JOSA.52.000672
        r = self.shape_func_deriv(y)
        r2 = (r*r).sum(axis=0)
        a = mu*(u*r).sum(axis=0)/r2
        # solve g**2+2*a*g+b=0
        #if mu**2 == 1: # FIXME one
        #    g = -2*a
        b = (mu**2-1)/r2
        # sign(mu) for reflection
        g = -a+np.sign(mu)*np.sqrt(a**2-b)
        u1 = mu*u+g*r
        #print "refract", self, u, mu, u1
        return u1

    def reverse(self):
        raise NotImplementedError

    def surface(self, axis, points=20): # TODO 2d
        t = np.linspace(-self.radius, self.radius, points)
        xyz = np.zeros((3, points))
        xyz[axis] = t
        xyz[2] = -self.shape_func(xyz)
        return xyz[axis]+self.origin[axis], xyz[2]+self.origin[2]


class Spheroid(Interface):
    typestr = "S"
    curvature = Float(0)
    conic = Float(1)
    # TODO: assert self.radius**2 < 1/(self.conic*self.curvature**2)
    aspherics = Array(dtype=np.float64)

    def shape_func(self, p):
        x, y, z = p
        r2 = x**2+y**2
        o = np.sum([ai*r2**(i+2) for i, ai in 
            enumerate(self.aspherics)], axis=0)
        c, k = self.curvature, self.conic
        return z-c*r2/(1+np.sqrt(1-k*c**2*r2))-o

    def shape_func_deriv(self, p):
        x, y, z = p
        r2 = x**2+y**2
        o = 2*np.sum([ai*(i+2)*r2**(i+1) for i, ai in 
            enumerate(self.aspherics)], axis=0)
        c, k = self.curvature, self.conic
        e = c/np.sqrt(1-k*c**2*r2)+o
        return np.array([-x*e, -y*e, np.ones_like(e)])

    def intercept(self, y, u):
        if len(self.aspherics) == 0:
            # replace the newton-raphson with the analytic solution
            c = self.curvature
            if c == 0:
                return Element.intercept(self, y, u)
            else:
                k = np.array([1., 1., self.conic])[:, None]
                ky = k*y
                ku = k*u
                d = c*(u*ky).sum(axis=0)-u[2]
                e = c*(u*ku).sum(axis=0)
                f = c*(y*ky).sum(axis=0)-2*y[2]
                s = (-d-np.sqrt(d**2-e*f))/e
                #print "intercept", self, u, y, d, e, f, s
        else:
            return Interface.intercept(self, y, u)
        return s #np.where(s*np.sign(self.origin[2])>=0, s, np.nan)

    def propagate_paraxial(self, r, j):
        y0, u0, n0 = r.y[0, j-1], r.u[0, j-1], r.n[j-1]
        c = self.curvature
        if len(self.aspherics) > 0:
            c += 2*self.aspherics[0]
        t = self.origin[2]
        n = map(self.material.refractive_index, r.l)
        mu = n0/n
        y = y0+t*u0 # propagate
        i = c*y+u0 # incidence
        u = mu*u0+c*(mu-1.)*y # refract
        r.y[0, j] = y
        r.y[2, j] = r.y[2, j-1]+t
        r.i[0, j] = i
        r.u[0, j] = u
        r.n[j] = n
        r.v[j] = [self.material.delta_n(l1, l2) for l1, l2 in zip(r.l1, r.l2)]
    
    def aberration3(self, r, j):
        # need to multiply by h=image height = inv/(n[-2] u[-2])
        y0, u0, n0, v0 = r.y[0, j-1], r.u[0, j-1], r.n[j-1], r.v[j-1]
        y, u, i, n, v = r.y[0, j], r.u[0, j], r.i[0, j], r.n[j], r.v[j]
        c = self.curvature
        mu = n0/n
        l = n*(u[0]*y[1]-u[1]*y[0])
        s = .5*n0*(1-mu)*y*(u+i)/l
        # transverse third-order spherical
        tsc = s[0]*i[0]**2
        # sagittal third-order coma
        cc = s[0]*i[0]*i[1]
        # tangential third-order com
        # 3*cc
        # transverse third-order astigmatism
        tac = s[0]*i[1]**2
        # transverse third-order Petzval
        tpc = ((1-mu)*c*l/n0/2)[0]
        # third-order distortion
        dc = s[1]*i[0]*i[1]+.5*(u[1]**2-u0[1]**2)
        # paraxial transverse axial, lateral chromatic
        tachc, tchc = -y[0]*i/l*(v0-mu*v)

        if len(self.aspherics) > 0:
           k = (4*self.aspherics[0]+(self.conic-1)*c**3/2)*(n-n0)/l
           tsc += k*y[0]**4
           cc += k*y[0]**3*y[1]
           tac += k*y[0]**2*y[1]**2
           dc += k*y[0]*y[1]**3
        r.c3[:, j] = [tsc, cc, tac, tpc, dc, tachc, tchc]

    def reverse(self):
        self.curvature *= -1
        self.aspherics *= -1


class Object(Element):
    typestr = "O"
    infinity = Bool(True)
    wavelengths = Array(dtype=np.float64, shape=(None,))


class Aperture(Element):
    typestr = "A"

    def propagate(self, r, j, stop=False):
        super(Aperture, self).propagate(r, j)
        if stop:
            r2 = (r.y[(0, 1), j]**2).sum(axis=0)
            np.putmask(r.y[:, j], r2>self.radius**2, np.nan)


class Image(Element):
    typestr = "I"
