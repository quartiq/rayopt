# -*- coding: utf8 -*-
#
#   rayopt - raytracing for optical imaging systems
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

from __future__ import (absolute_import, print_function,
                        unicode_literals, division)

import numpy as np
from scipy.optimize import newton

from .utils import public
from .transformations import (euler_matrix, euler_from_matrix,
                              rotation_matrix)
from .name_mixin import NameMixin
from .material import Material


@public
class TransformMixin(object):
    def __init__(self, distance=0., direction=(0, 0, 1.), angles=(0, 0, 0),
                 offset=None):
        self.update(distance, direction, angles)
        if offset is not None:
            self.offset = offset
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
        if d:
            direction = offset/d
        else:
            direction = 0, 0, 1.
        self.update(d, direction, self._angles)

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
        r = mu*i - direction  # snell
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
        angles = euler_from_matrix(rot, str("rxyz"))
        self.update(self.distance, self.direction, angles)

    def update(self, distance, direction, angles):
        dlen = np.linalg.norm(direction)
        if not dlen:
            direction = 0, 0, 1.
            dlen = 1.
        u = np.array(direction)/dlen
        if distance < 0:
            distance *= -1
            u *= -1
        self._distance = d = distance
        self._direction = u
        self._offset = d*u
        self._angles = a = np.array(angles)
        self.straight = np.allclose(u, (0, 0, 1.))
        self.normal = np.allclose(a, 0.)
        self.rotated = not (self.normal and self.straight)
        if not self.rotated:
            self.rot_axis = self.rot_normal = None
            return
        r = np.eye(3)
        if not self.straight:
            rdir = np.cross(u, (0, 0, 1.))
            rang = np.arcsin(np.linalg.norm(rdir))
            if u[2] < 0:  # == np.dot((0, 0, 1), u)
                rang = np.pi - rang
            if np.allclose(rdir, 0):
                rdir = 1., 0, 0
            self.rot_axis = r1 = rotation_matrix(rang, rdir)[:3, :3]
            r = np.dot(r, r1)
        if not self.normal:
            r1 = euler_matrix(axes=str("rxyz"), *tuple(a))[:3, :3]
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

    def refract(self, y, u0, mu):
        return u0

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

    def propagate(self, y0, u0, n0, l, clip=True):
        t = self.intercept(y0, u0)
        y = y0 + t[:, None]*u0
        if clip:
            u0 = self.clip(y, u0)
        n = n0
        return y, u0, n, t*n0

    def transfer_poly(self, state):
        fd = (-state.f).shift(self.offset[2])
        fdp = fd*state.p
        r = state.r + fd*(2*state.k + fdp)
        k = state.k + fdp
        return fd, r, k

    def intercept_poly(self, r, p, k):
        S = r.__class__
        f = S()
        fr = S()
        g = S().shift(1)
        return r, f, fr, g

    def propagate_poly(self, state, l):
        raise NotImplementedError

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

    def get_n_mu(self, n0, l):
        if self.material is None:
            return n0, 1.
        if self.material.mirror:
            return n0, -1.
        n = self.material.refractive_index(l)
        return n, n0/n

    def dict(self):
        dat = super(Interface, self).dict()
        if self.material is not None:
            dat["material"] = str(self.material)
        return dat

    def refractive_index(self, wavelength):
        return self.material.refractive_index(wavelength)

    def paraxial_matrix(self, n0, l):
        n, m = super(Interface, self).paraxial_matrix(n0, l)
        if self.material is not None:
            n = self.refractive_index(l)
        return n, m

    def propagate(self, y0, u0, n0, l, clip=True):
        t = self.intercept(y0, u0)
        y = y0 + t[:, None]*u0
        if clip:
            u0 = self.clip(y, u0)
        u = u0
        n, mu = self.get_n_mu(n0, l)
        if mu:
            u = self.refract(y, u0, mu)
        return y, u, n, t*n0

    def dispersion(self, lmin, lmax):
        if self.material is None:
            return 0.
        return self.material.delta_n(lmin, lmax)

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
            u = u0 - 2*a[:, None]*r  # reflection
        else:
            b = (mu**2 - 1)/r2
            g = -a + np.sign(mu)*np.sqrt(np.square(a) - b)
            u = muf*u0 + g[:, None]*r  # refraction
        return u

    def surface_cut(self, axis, points):
        if self.material is None:
            return super(Interface, self).surface_cut(axis, points)
        rad = self.radius
        xyz = np.zeros((points, 3))
        xyz[:, axis] = np.linspace(-rad, rad, points)
        xyz[:, 2] = -self.surface_sag(xyz)
        return xyz

    def intercept_poly(self, r, p, k):
        raise NotImplementedError

    def propagate_poly(self, state, l):
        fd, rt, kt = self.transfer_poly(state)
        r, f, fr, g = self.intercept_poly(rt, state.p, kt)
        n, mu = self.get_n_mu(state.n, l)

        p1 = state.p.copy().shift(1)
        mun = mu*p1**-.5  # (30)
        ct = g*mun*(-2*(kt + f*state.p)*fr).shift(1)  # (31)
        gdct = g*((ct*ct).shift(1 - mu**2)**.5 - ct)  # (32)
        n1i = (mun + gdct)**-1.  # (33)
        a = f + fd
        b = -2*n1i*gdct*fr  # (34)
        c = mun*n1i
        ap = a*state.p

        r = state.r + a*(2*state.k + ap)  # (~35)
        p = (n1i*n1i).shift(-1)  # (40.2)
        k = b*r + c*(state.k + ap)

        s = state.s + a*state.v  # (39)
        t = state.t + a*state.w
        v = b*s + c*state.v
        w = b*t + c*state.w
        o = state.o + state.n*a*p1**.5  # (57)
        PolyState = state.__class__
        return PolyState(f=f, n=n, r=r, k=k, p=p, s=s, t=t, v=v, w=w, o=o)


@public
@Element.register
class Spheroid(Interface):
    def __init__(self, curvature=0., conic=0., aspherics=None, roc=None,
                 alternate_intersection=False, **kwargs):
        super(Spheroid, self).__init__(**kwargs)
        if roc is not None:
            curvature = 1./roc
        self.alternate_intersection = alternate_intersection
        self.curvature = curvature
        self.conic = conic
        if aspherics is not None:
            aspherics = list(aspherics)
        self.aspherics = aspherics
        if self.curvature and np.isfinite(self.radius) and self.conic > -1:
            assert self.radius**2 <= 1/((1 + self.conic)*self.curvature**2)

    def dict(self):
        dat = super(Spheroid, self).dict()
        if self.curvature:
            dat["curvature"] = float(self.curvature)
        if self.conic:
            dat["conic"] = float(self.conic)
        if self.aspherics is not None:
            dat["aspherics"] = list(map(float, self.aspherics))
        if self.alternate_intersection:
            dat["alternate_intersection"] = True
        return dat

    def surface_sag(self, xyz):
        e = xyz[..., 2].copy()
        if not self.curvature and self.aspherics is None:
            return e
        xy = xyz[..., :2]
        r2 = np.einsum("...i,...i", xy, xy)
        if self.curvature:
            c, k = self.curvature, self.conic
            e -= c*r2/(1 + np.sqrt(1 - (1 + k)*c**2*r2))
        if self.aspherics is not None:
            d = 0.
            for ai in reversed(self.aspherics):
                d += ai
                d *= r2
            e -= d
        return e

    def surface_normal(self, xyz):
        q = np.zeros_like(xyz)
        q[..., 2] = 1
        if not self.curvature and self.aspherics is None:
            return q
        xy = xyz[..., :2]
        r2 = np.einsum("...i,...i", xy, xy)
        e = 0.
        if self.curvature:
            c, k = self.curvature, self.conic
            e -= c/np.sqrt(1 - (1 + k)*c**2*r2)
        if self.aspherics is not None:
            d = 0.
            for i in reversed(range(len(self.aspherics))):
                d *= r2
                d += 2*(i + 1)*self.aspherics[i]
            e -= d
        q[..., :2] = xy*e[..., None]
        return q

    def intercept(self, y, u):
        if self.aspherics is not None:
            return Interface.intercept(self, y, u)  # expensive iterative
        # replace the newton-raphson with the analytic solution
        c, k = self.curvature, self.conic
        if c == 0:
            return -y[:, 2]/u[:, 2]  # flat
        if not k:
            uy = (u*y).sum(1)
            uu = 1.
            yy = np.square(y).sum(1)
        else:
            k = np.array([(1, 1, 1 + k)])
            uy = (u*y*k).sum(1)
            uu = (np.square(u)*k).sum(1)
            yy = (np.square(y)*k).sum(1)
        d = c*uy - u[:, 2]
        e = c*uu
        f = c*yy - 2*y[:, 2]
        g = np.sqrt(np.square(d) - e*f)
        if self.alternate_intersection:
            g *= -1
        # g *= np.sign(u[:, 2])
        s = -(d + g)/e
        return s

    def paraxial_matrix(self, n0, l):
        # [y', u'] = M * [y, u]
        c = self.curvature
        if self.aspherics is not None:
            c = c + 2*self.aspherics[0]
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
            self.aspherics = [ai/scale**(2*i + 1) for i, ai in
                              enumerate(self.aspherics)]

    def aberration(self, y, u0, u, n0, n, v0, v):
        c = self.curvature
        if self.material is not None and self.material.mirror:
            n = -n  # FIXME check, cleanup
        mu = n0/n
        # incidence
        i = c*y+u0
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
        tpc = (1-mu)*c*l/n0/2
        # third-order distortion
        dc = s[1]*i[0]*i[1]+.5*(u[1]**2-u0[1]**2)
        # paraxial transverse axial, lateral chromatic
        tachc, tchc = -y[0]*i/l*(v0-mu*v)

        if self.aspherics:
            # FIXME check
            k = (4*self.aspherics[0]+(self.conic-1)*c**3/2)*(n-n0)/l
            k = k[0]
            tsc += k*y[0]**4
            cc += k*y[0]**3*y[1]
            tac += k*y[0]**2*y[1]**2
            dc += k*y[0]*y[1]**3
        return tsc, cc, tac, tpc, dc, tachc, tchc

    def intercept_poly(self, r, p, k):
        S = r.__class__
        u = self.curvature*np.sign(self.offset[2])
        if u == 0.:
            r, f, fr, g = Element.intercept_poly(self, r, p, k)
        else:
            p1 = p.copy().shift(1)
            a = (-u*k).shift(1)
            a -= (a*a - p1*r*u**2)**.5
            a = a*p1**-1  # (44)
            f = a/u
            r = a*(-a).shift(2)  # (45)
            g = (-a).shift(1)  # (47)
            fr = .5*u*g**-1.  # (46)
        if self.aspherics:
            # FIXME: not curve/conic
            u = self.aspherics
            r0 = r
            for i in range(len(u)):  # (28)
                df = S()
                for uj in reversed(u):
                    df = df.shift(uj*np.sign(self.offset[2]))*r
                # FIXME: real Newton Raphson
                r = r0 + df*(2*k + df*p)
            dfr = S()
            for i in reversed(range(len(u))):
                dfr = (dfr*r).shift((i + 1)*u[i]*np.sign(self.offset[2]))
            # FIXME
            f += df
            fr += dfr
            g = (4*r*dfr*dfr).shift(1)**-.5
        return r, f, fr, g
