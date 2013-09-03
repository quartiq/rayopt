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
        s = -y[2]/u[2]
        #s = np.where(y[2]==0, 0., s)
        return s # TODO mask, np.where(s>=0, s, np.nan)

    def clip(self, u, y):
        r2 = (y[(0, 1), :]**2).sum(axis=0)
        np.putmask(u[2], r2>self.radius**2, np.nan)

    def propagate(self, r, j, clip):
        y0, u0, n0 = r.y[:, j-1], r.u[:, j-1], r.n[j-1]
        y = np.dot(self.rotation[:3, :3], y0-self.origin[:, None])
        u = np.dot(self.rotation[:3, :3], u0)
        n = n0
        # length up to surface
        p = self.intercept(y, u)
        # new transverse position
        y = y + p[None, :]*u
        if clip:
            self.clip(u, y)
        if self.material is not None:
            n = map(self.material.refractive_index, r.l)
            u = self.refract(y, u, n0/n)
        r.n[j] = n
        r.p[j-1] = p
        r.y[:, j] = np.dot(self.rotation[:3, :3].T, y)#+self.origin[:, None]
        r.u[:, j] = np.dot(self.rotation[:3, :3].T, u)

    def refract(self, y, u, mu):
        return u
  
    def propagate_paraxial(self, r, j):
        y0, u0 = r.y[0, j-1], r.u[0, j-1]
        t = self.origin[2]
        n = map(self.material.refractive_index, r.l)
        y = y0+t*u0 # propagate
        u = u0
        r.n[j] = n
        r.y[0, j] = y
        r.y[2, j] = r.y[2, j-1]+t
        r.u[0, j] = u
        r.v[j] = [self.material.delta_n(l1, l2) for l1, l2 in zip(r.l1, r.l2)]

    def paraxial_matrix(self, l, n0):
        # [y', u'] = M * [y, u]
        n = self.material.refractive_index(l)
        d = self.origin[2]
        return n, np.matrix([[1, d], [0, 1]])

    def set_aberration3(self, r, j):
        r.c3[:, j] = 0
    
    def aberration(self, r, j):
        r.aberration3[:, j] = 0
        r.aberration5_intrinsic[:, j] = 0
        r.aberration5[:, j] = 0

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
                        x0=-yi[2]/ui[2], tol=1e-7, maxiter=10)
            except RuntimeError:
                s[i] = np.nan
        return s # np.where(s>=0, s, np.nan) # TODO mask

    def refract(self, y, u, mu):
        # General Ray-Tracing Procedure
        # G. H. Spencer and M. V. R. K. Murty
        # JOSA, Vol. 52, Issue 6, pp. 672-676 (1962)
        # doi:10.1364/JOSA.52.000672
        if np.all(mu == 1): return u # no change
        r = self.shape_func_deriv(y)
        r2 = (r*r).sum(axis=0)
        a = np.fabs(mu)*(u*r).sum(axis=0)/r2
        # solve g**2+2*a*g+b=0
        if np.all(mu == -1): return u-2*a*r # reflection
        b = (mu**2-1)/r2
        g = -a+np.sign(mu)*np.sqrt(a**2-b)
        return np.fabs(mu)*u+g*r # refraction

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
                s = (-d-np.sign(u[2])*np.sqrt(d**2-e*f))/e
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
        u = mu*u0+c*(mu-1.)*y # refract
        r.y[0, j] = y
        r.y[2, j] = r.y[2, j-1]+t
        r.u[0, j] = u
        r.n[j] = n
        r.v[j] = [self.material.delta_n(l1, l2) for l1, l2 in zip(r.l1, r.l2)]

    def paraxial_matrix(self, l, n0):
        # [y', u'] = M * [y, u]
        n = self.material.refractive_index(l)
        d = self.origin[2]
        p = (n-n0)*self.curvature/n
        return n, np.matrix([[1, d], [-p, n0/n-d*p]])
   
    def set_aberration3(self, r, j):
        # need to multiply by h=image height = inv/(n[-2] u[-2])
        y0, u0, n0, v0 = r.y[0, j-1], r.u[0, j-1], r.n[j-1], r.v[j-1]
        y, u, n, v = r.y[0, j], r.u[0, j], r.n[j], r.v[j]
        c = self.curvature
        mu = n0/n
        i = c*y+u0 # incidence
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
            # FIXME check
            k = (4*self.aspherics[0]+(self.conic-1)*c**3/2)*(n-n0)/l
            k = k[0]
            tsc += k*y[0]**4
            cc += k*y[0]**3*y[1]
            tac += k*y[0]**2*y[1]**2
            dc += k*y[0]*y[1]**3
        r.c3[:, j] = [tsc, cc, tac, tpc, dc, tachc, tchc]

    def aberration(self, r, j):
        # kdp5/parax2.for, Smith/ModernOpticalEngineering
        y, yb = r.y[0, j]
        (u0, ub0), (u, ub) = -r.u[0, j-1:j+1]
        n0, n = r.n[j-1:j+1, 0]
        v0, v = r.v[j-1:j+1, 0]
        c = self.curvature

        # 3rd order helpers
        mu = n0/n
        i, ib = c*y-u0, c*yb-ub0 # incidence
        e, eb = mu*i, mu*ib # excidence
        l = n*(yb*u-y*ub) # lagrange
        s, sb = n0*y*(u-i)*(1-mu)/(2*l), n0*yb*(ub-ib)*(1-mu)/(2*l)

        # 3rd order coefficients
        p3 = (mu-1)*c/n0 # 3 petzval
        b3, b3b = s*i**2, -sb*ib**2 # 3 spherical
        f3, f3b = s*i*ib, -sb*i*ib # 3 coma
        c3, c3b = s*ib**2, -sb*i**2 # 3 astigmatism
        d3, d3b = c3+p3*l/2., c3b-p3*l/2. # 3 curvature
        e3, e3b = -f3b+(ub**2-ub0**2)/2, -f3+(u**2-u0**2)/2. # 3 distortion
        g3, g3b = y*(v0-mu*v)*i, y*(v0-mu*v)*ib # long, lat color

        check = 2*(c3+c3b)+(u0*ub0-u*ub)
        #assert check == 0, check
        check = ib*d3-i*e3
        #assert check == 0, check
        check = i*d3b-i*e3b
        #assert check == 0, check

        if len(self.aspherics) > 0:
            w = ((self.conic-1)/8.+self.aspherics[1])
            k = 4*(n0-n)*k/l
            b3 += k*y**4
            b3b += -k*yb**4
            f3 += k*y**3*yb
            f3b += -k*yb**3*y
            c3 += k*y**2*yb**2
            c3b += -k*yb**2*y**2
            d3 += k*y**2*yb**2
            d3b += -k*yb**2*y**2
            e3 += k*y*yb**3
            e3b += -k*yb*y**3
        
        r.aberration3[:, j] = (p3, b3, f3, c3, p3*l/2, e3, -g3/l/2, 
                b3b, f3b, c3b, d3b, e3b, -g3b/l/2)
        (p3s, b3s, f3s, c3s, d3s, e3s, g3s,
                b3bs, f3bs, c3bs, d3bs, e3bs, g3bs
                ) = r.aberration3[:, :j].sum(axis=1)

        #g3 *= -l*2 # not used
        #g3b *= -l*2 # not used

        # 5th order helpers
        x73 = 3*i*e+2*u**2-3*u0**2
        x74 = 3*i*eb+2*u*ub-3*u0*ub0
        x75 = 3*ib*eb+2*ub**2-3*ub0**2
        x76 = -i*(3*u0-u)
        x77 = -ib*(2*u0-u)-i*ub0
        x78 = -ib*(3*ub0-ub)
        x42 = yb**2*c*i-y*ib*(ub+ub0) # yb*i*(ib-u0b)+...
        x82 = -yb**2*c*u0+y*eb*(ub+ub0) # yb*u0*(ib-u0b)-...
        xb42 = y**2*c*ib+yb*i*(u+u0) # y*ib*(i-u0)+...
        xb82 = -y**2*c*ub0+yb*e*(u+u0) # y*u0b*(i-u0)+...
        w = (i**2+e**2+u**2-3*u0**2)/8.

        s1p = 3*w*s*i/2.
        s2p = s*(ib*x73+i*x74-ub*x76-u*x77)/4.
        s3p = n0*(mu-1)*(x42*x73+x76*x82+y*(i+u)*(i*x75-u*x78))/4.
        s4p = s*(ib*x74-ub*x77)/2.
        s5p = n0*(mu-1)*(x42*x74+x77*x82+y*(i+u)*(ib*x75-ub*x78))/4.
        s6p = n0*(mu-1)*(x42*x75+x78*x82)/8.
        s1q = n0*(mu-1)*(xb42*x73+x76*xb82)/8.

        # intrinsic 5th order
        b5 = 2*i*s2p # spherical coma
        # coma
        f5a, f5b = 2*ib*s1p+i*s2p, i*s2p
        # oblique spherical
        m5a, m5b, m5c = 2*ib*s2p, 2*i*s3p, 2*i*s4p
        # elliptical coma
        n5a, n5b, n5c = 2*ib*s3p, 2*ib*s4p+2*i*s5p, 2*i*s5p
        c5 = ib*s5p/2. # astigmatism
        p5 = 2*i*s6p-ib*s5p/2. # petzval
        e5 = 2*ib*s6p # image distortion
        e5b = 2*i*s1q # pupil distortion

        r.aberration5_intrinsic[:, j] = (b5, f5a, f5b, m5a, m5b, m5c,
                n5a, n5b, n5c, c5, p5, e5, e5b)

        # extrinsic 5th order
        b5e = 3*(f3*b3s-b3*f3s)/(2*l) # spherical
        # coma
        f5ae = ((p3+4*c)*b3s+(5*f3s-4*e3s)*f3-(2*p3s+5*c3bs)*b3)/(2*l)
        f5be = ((p3+2*c)*b3s+2*(2*f3s-e3bs)*f3-(p3s+4*c3bs)*b3)/(2*l)
        # oblique spherical
        m5ae = (e3*b3s+(4*f3s-e3bs)*c3+(c3s-4*c3bs-2*p3s)*f3-b3*f3bs)/l
        m5be = (e3*b3s+(p3+c3)*(2*f3s-e3bs)+(p3s+3*c3s-2*c3bs)*f3
                -3*b3*f3bs)/(2*l)
        m5ce = 2*((2*c3+p3)*f3s+(c3s-2*c3bs)*f3b-b3*f3bs)/l
        # elliptical coma
        n5ae = (3*e3*f3s-(p3+c3)*(p3s+c3bs)+2*c3*(p3s-c3bs)
                +f3*(e3s-2*f3bs)-b3*b3s)/(2*l)
        n5be = (3*e3*f3s+(p3+3*c3)*(3*c3s-c3bs+p3s)-c3*(p3s+c3s)
                +f3*(e3s-8*f3bs)-b3*b3bs)/l
        n5ce = (e3*f3s+(p3+c3)*(3*c3s-c3bs+p3s)+c3*(c3s+p3s)
                +f3*(e3s-4*f3bs)-b3*b3bs)/(2*l)
        # astigmatism
        c5e = (e3*(4*c3s+p3s)-p3*f3bs+2*c3*(e3s-2*f3bs)-2*f3*b3bs)/(4*l)
        # petzval
        p5e = (e3*(p3s-2*c3s)+p3*(4*e3s-f3bs)+2*c3*(e3s+f3bs)
                -2*f3*b3bs)/(4*l)
        # image distortion
        e5e = (5*e3*e3s-(p3+3*c3)*b3bs)/(2*l)
        # pupil distortion
        e5be = (-3*e3*e3bs+(p3+3*c3b)*b3s)/(2*l)
        
        r.aberration5[:, j] = r.aberration5_intrinsic[:, j] + (
                b5e, f5ae, f5be, m5ae, m5be, m5ce, n5ae,
                n5be, n5ce, c5e, p5e, e5e, e5be)

    def reverse(self):
        self.curvature *= -1
        self.aspherics *= -1


class Object(Element):
    typestr = "O"
    infinity = Bool(True)
    wavelengths = Array(dtype=np.float64, shape=(None,))

    def propagate_paraxial(self, r, j):
        pass
    
    def propagate(self, r, j, clip):
        pass


class Aperture(Element):
    typestr = "A"

    def surface(self, axis, points=20):
        t = np.array([-self.radius*1.5, -self.radius, np.nan,
            self.radius, self.radius*1.5])
        xyz = np.zeros((3, 5))
        xyz[axis] = t
        return xyz[axis]+self.origin[axis], xyz[2]+self.origin[2]


class Image(Element):
    typestr = "I"
