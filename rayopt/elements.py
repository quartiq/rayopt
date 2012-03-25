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

from traits.api import (HasTraits, Float, Array,
        Trait, cached_property, Property, Enum)

from .transformations import euler_matrix, translation_matrix
from .material import Material, air


def dotprod(a,b):
    return (a*b).sum(-1)

class Element(HasTraits):
    typestr = "E"
    origin = Array(dtype=np.float64, shape=(3,))
    angles = Array(dtype=np.float64, shape=(3,))
    transform = Property(depends_on="origin, angles")
    inverse_transform = Property(depends_on="transform")
    material = Trait(air, Material)
    radius = Float

    @cached_property
    def _get_transform(self):
        r = euler_matrix(axes="rxyz", *self.angles)
        t = translation_matrix(-self.origin)
        return np.dot(r,t)

    @cached_property
    def _get_inverse_transform(self):
        return np.linalg.inv(self.transform)

    def transform_to(self, rays):
        return rays.transform(self.transform)

    def transform_from(self, rays):
        return rays.transform(self.inverse_transform)

    def intercept(self, positions, angles):
        # ray length to intersection with element
        # only reference plane, overridden in subclasses
        # solution for z=0
        s = -positions[..., 2]/angles[..., 2]
        return s # TODO mask, np.where(s>=0, s, np.nan)

    def propagate(self, in_rays):
        out_rays = self.transform_to(in_rays)
        # length up to surface
        in_rays.lengths = self.intercept(
                out_rays.positions, out_rays.angles)
        out_rays.optical_path_lengths = in_rays.optical_path_lengths+\
                in_rays.lengths*in_rays.refractive_index
        # new transverse position
        out_rays.positions = out_rays.positions + \
                (in_rays.lengths*out_rays.angles.T).T
        out_rays.wavelength = in_rays.wavelength
        if self.material is None:
            out_rays.refractive_index = in_rays.refractive_index
        else:
            out_rays.refractive_index = self.material.refractive_index(
                    out_rays.wavelength)
            m = in_rays.refractive_index/out_rays.refractive_index
            out_rays.angles = self.refract(
                    out_rays.positions, out_rays.angles, m)
        return in_rays, out_rays
   
    def propagate_paraxial(self, r, j):
        y0, u0, n0 = r.y[0, j-1], r.u[0, j-1], r.n[j-1]
        t = self.origin[2]
        n = self.material.refractive_index(r.l)
        y = y0+t*u0 # propagate
        i = u0 # incident
        u = u0
        r.y[0, j] = y
        r.y[2, j] = r.y[2, j-1]+t
        r.i[0, j] = i
        r.u[0, j] = u
        r.n[j] = n
        r.v[j] = self.material.delta_n(r.l1, r.l2)
 
    def aberration3(self, r, j):
        r.c3[j] = 0

    def aberration5(self, r, j):
        r.c3[j] = 0

    def revert(self):
        pass

    def surface(self, axis, points=20):
        t = np.linspace(-self.radius, self.radius, 2)
        xyz = [np.zeros_like(t)]*3
        xyz[axis] = t
        return xyz[axis]+self.origin[axis], xyz[2]+self.origin[2]


class Interface(Element):
    typestr = "F"

    def shape_func(self, p):
        raise NotImplementedError

    def shape_func_deriv(self, p):
        raise NotImplementedError

    def intercept(self, p, a):
        s = np.zeros_like(p[:,0])
        for i in range(p.shape[0]):
            try:
                s[i] = newton(func=lambda s: self.shape_func(p[i]+s*a[i]),
                    fprime=lambda s: np.dot(self.shape_func_deriv(p[i]+s*a[i]),
                        a[i]), x0=-p[i,2]/a[i,2], tol=1e-7, maxiter=15)
            except RuntimeError:
                s[i] = nan
        return where(s>=0, s, nan) # TODO mask

    def refract(self, f, a, m):
        # General Ray-Tracing Procedure
        # G. H. SPENCER and M. V. R. K. MURTY
        # JOSA, Vol. 52, Issue 6, pp. 672-676 (1962)
        # doi:10.1364/JOSA.52.000672
        # sign(m) for reflection
        fp = self.shape_func_deriv(f)
        fp2 = dotprod(fp, fp)
        o = m*dotprod(a, fp)/fp2
        if m**2 == 1:
            g = -2*o
        else:
            p = (m**2-1)/fp2
            g = sign(m)*np.sqrt(o**2-p)-o
        r = m*a+(g*fp.T).T
        #print "rfr", self, f, a, g, r
        return r

    def revert(self):
        raise NotImplementedError

    def surface(self, axis, points=20):
        t = np.linspace(-self.radius, self.radius, points)
        xyz = [np.zeros_like(t)]*3
        xyz[axis] = t
        xyz[2] = -self.shape_func(np.array(xyz).T)
        return xyz[axis]+self.origin[axis], xyz[2]+self.origin[2]


class Spheroid(Interface):
    typestr = "S"
    curvature = Float(0)
    conic = Float(1) # assert self.radius**2 < 1/(self.conic*self.curvature**2)
    aspherics = Array(dtype=np.float64)

    def shape_func(self, p):
        x, y, z = p.T
        r2 = x**2+y**2
        j = range(len(self.aspherics))
        o = dotprod(self.aspherics,
                np.array([r2**(i+2) for i in j]).T)
        return z-self.curvature*r2/(1+
                    np.sqrt(1-self.conic*self.curvature**2*r2))-o

    def shape_func_deriv(self, p):
        x, y, z = p.T
        r2 = x**2+y**2
        j = range(len(self.aspherics))
        o = dotprod(2*self.aspherics,
                np.nan_to_num(np.array([(i+2)*r2**(i+1) for i in j])).T)
        e = self.curvature/np.sqrt(1-self.conic*self.curvature**2*r2)+o
        return np.array([-x*e, -y*e, np.ones_like(e)]).T

    def intercept(self, p, a):
        if len(self.aspherics) == 0:
            # replace the newton-raphson with the analytic solution
            c = self.curvature
            if c == 0:
                return Element.intercept(self, p, a)
            else:
                k = np.array([1,1,self.conic])
                d = c*dotprod(a,k*p)-a[...,2]
                e = c*dotprod(a,k*a)
                f = c*dotprod(p,k*p)-2*p[...,2]
                s = (-np.sqrt(d**2-e*f)-d)/e
        else:
            return Interface.intercept(self, p, a)
        return where(s*sign(self.origin[2])>=0, s, nan)

    def propagate_paraxial(self, r, j):
        y0, u0, n0 = r.y[0, j-1], r.u[0, j-1], r.n[j-1]
        c = self.curvature
        if len(self.aspherics) > 0:
            c += 2*self.aspherics[0]
        t = self.origin[2]
        n = self.material.refractive_index(r.l)
        mu = n0/n
        y = y0+t*u0 # propagate
        i = c*y+u0 # incidence
        u = mu*u0+c*(mu-1.)*y # refract
        r.y[0, j] = y
        r.y[2, j] = r.y[2, j-1]+t
        r.i[0, j] = i
        r.u[0, j] = u
        r.n[j] = n
        r.v[j] = self.material.delta_n(r.l1, r.l2)
    
    def aberration3(self, r, j):
        y0, u0, n0, v0 = r.y[0, j-1], r.u[0, j-1], r.n[j-1], r.v[j-1]
        y, u, i, n, v = r.y[0, j], r.u[0, j], r.i[0, j], r.n[j], r.v[j]
        c = self.curvature
        mu = n0/n
        l = n*(u[0]*y[1]-u[1]*y[0])
        s = .5*n0*(1-mu)*y*(u+i)/l
        tsc = s[0]*i[0]**2
        cc = s[0]*i[0]*i[1]
        tac = s[0]*i[1]**2
        tpc = ((1-mu)*c*l/n0/2)[0]
        dc = s[1]*i[0]*i[1]+.5*(u[1]**2-u0[1]**2)
        tachc, tchc = -y[0]*i/l*(v0-mu*v)

        if len(self.aspherics) > 0:
           k = (4*self.aspherics[0]+(self.conic-1)*c**3/2)*(n-n0)/l
           tsc += k*y[0]**4
           cc += k*y[0]**3*y[1]
           tac += k*y[0]**2*y[1]**2
           dc += k*y[0]*y[1]**3
        r.c3[:, j] = [tsc, cc, tac, tpc, dc, tachc, tchc]

    def revert(self):
        self.curvature *= -1
        self.aspherics *= -1


class Object(Element):
    typestr = "O"
    radius = Float(np.inf)
    field_angle = Float(.1)
    apodization = Enum(("constant", "gaussian", "cos3"))

    def rays_to_height(self, xy, height):
        if self.radius == np.inf:
            p = np.array([(xy[0], xy[1], np.zeros_like(xy[0]))])
            a = np.array([(height[0]*self.field_angle,
                        height[1]*self.field_angle,
                        np.sqrt(1-(height[0]*self.field_angle)**2
                              -(height[1]*self.field_angle)**2))])
        else:
            p = np.array([(height[0]*self.radius,
                        height[1]*self.radius,
                        np.zeros_like(height[0]))])
            a = np.array([(xy[0], xy[1], np.sqrt(1-xy[0]**2-xy[1]**2))])
        return p, a

    def rays_for_point(self, height, chief, marg, num):
        chief_x, chief_y = chief
        marg_px, marg_nx, marg_py, marg_ny = marg
        mmarg_x, mmarg_y = marg_px+marg_nx, marg_py+marg_ny
        dmarg_x, dmarg_y = marg_px-marg_nx, marg_py-marg_ny

        x, y = mgrid[marg_nx:marg_px:num*1j, marg_ny:marg_py:num*1j]
        x, y = x.flatten(), y.flatten()
        r2 = (((x-mmarg_x)/dmarg_x)**2+((y-mmarg_y)/dmarg_y)**2)<.25
        x, y = np.extract(r2, x), np.extract(r2, y)
        x = np.concatenate(([chief_x], np.linspace(marg_nx, marg_px, num),
            np.ones((num,))*chief_x, x))
        y = np.concatenate(([chief_y], np.ones((num,))*chief_y,
            np.linspace(marg_nx, marg_px, num), y))
        p, a = self.rays_to_height((x,y),
                (height[0]*np.ones_like(x), height[1]*np.ones_like(y)))
        return p[0].T, a[0].T


class Aperture(Element):
    typestr = "A"
    radius = Float

    def propagate(self, in_rays, stop=False):
        in_rays, out_rays = super(Aperture, self).propagate(in_rays)
        if stop:
            r = (out_rays.positions[...,(0,1)]**2).sum(axis=-1)
            putmask(out_rays.positions[...,2], r>self.radius**2, nan)
        return in_rays, out_rays


class Image(Element):
    typestr = "I"
    radius = Float
