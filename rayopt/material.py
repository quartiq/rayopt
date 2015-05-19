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

import warnings

import numpy as np

from .utils import simple_cache
from .name_mixin import NameMixin
from .utils import public


__all__ = "vacuum mirror air fraunhofer".split()


fraunhofer = dict(   # http://en.wikipedia.org/wiki/Abbe_number
    i  =  365.01e-9, # Hg UV
    h  =  404.66e-9, # Hg violet
    g  =  435.84e-9, # Hg blue
    Fp =  479.99e-9, # Cd blue
    F  =  486.13e-9, # H  blue
    e  =  546.07e-9, # Hg green
    d  =  587.56e-9, # He yellow
    D  =  589.30e-9, # Na yellow
    Cp =  643.85e-9, # Cd red
    C  =  656.27e-9, # H  red
    r  =  706.52e-9, # He red
    Ap =  768.20e-9, # K  IR
    s  =  852.11e-9, # Cs IR
    t  = 1013.98e-9, # Hg IR
)

lambda_F = fraunhofer["F"]
lambda_d = fraunhofer["d"]
lambda_C = fraunhofer["C"]



class Thermal(object):
    def __init__(self, d, e, tref=20., lref=lambda_d):
        self.d = d
        self.e = e
        self.tref = tref
        self.lref = lref

    def dn_thermal(self, t, n, wavelength=None):
        dt = t - self.tref
        if wavelength is None:
            w = self.lref
        else:
            w = wavelength/1e-6
        # schott Constants of the formula dn/dT
        dn = (n**2 - 1)/(2*n)*(
            self.d[0]*dt + self.d[1]*dt**2 + self.d[2]*dt**3 +
            (self.e[0]*dt + self.e[1]*dt**2)/(w**2 - self.lref**2)
        )
        return dn

    def dict(self):
        return {"d": self.d, "e": self.e, "tref": self.tref, "lref": self.lref}


@public
class Material(NameMixin):
    def __init__(self, name="-", solid=True, mirror=False, catalog=None,
                 thermal=None):
        self.name = name
        self.solid = solid
        self.mirror = mirror
        self.catalog = catalog
        self.thermal = thermal

    @classmethod
    def make(cls, name):
        if name is None:
            return None
        if isinstance(name, Material):
            return name
        if isinstance(name, dict):
            return super(Material, self).make(name)
        if type(name) is float:
            return ModelMaterial(n=name)
        if type(name) is tuple:
            return AbbeMaterial(nd=name[0], vd=name[1])
        try:
            return AbbeMaterial.from_string(name)
        except ValueError:
            pass
        parts = name.lower().split("/")
        name = parts.pop()
        source, catalog = None, None
        if parts:
            catalog = parts.pop()
        if parts:
            source = parts.pop()
        if catalog in (None, "basic") and name in basic:
            return basic[name]
        from .library import Library
        lib = Library.one()
        return lib.get("glass", name, catalog, source)

    def __str__(self):
        if self.catalog is not None:
            return "%s/%s" % (self.catalog, self.name)
        else:
            return self.name

    def dict(self):
        dat = {}
        if self.name:
            dat["name"] = self.name
        if not self.solid:
            dat["solid"] = self.solid
        if self.mirror:
            dat["mirror"] = self.mirror
        if self.catalog:
            dat["catalog"] = self.catalog
        if self.thermal:
            dat["thermal"] = self.thermal.dict()
        return dat

    @simple_cache
    def refractive_index(self, wavelength):
        return 1.

    def dispersion(self, short, mid, long):
        dn = self.delta_n(short, long)
        if dn:
            return (self.refractive_index(mid) - 1)/dn
        else:
            return np.inf

    def delta_n(self, short, long):
        return (self.refractive_index(short) - self.refractive_index(long))

    @property
    def nd(self):
        return self.refractive_index(lambda_d)

    @property
    def vd(self):
        return self.dispersion(lambda_F, lambda_d, lambda_C)


@public
class ModelMaterial(Material):
    def __init__(self, n=1., **kwargs):
        super(ModelMaterial, self).__init__(**kwargs)
        self.n = n

    def refractive_index(self, wavelength):
        return self.n

    def dict(self):
        dat = super(ModelMaterial, self).dict()
        dat["n"] = self.n
        return dat


@public
class AbbeMaterial(Material):
    def __init__(self, n=1., v=np.inf, lambda_ref=lambda_d,
                 lambda_long=lambda_C, lambda_short=lambda_F, **kwargs):
        super(AbbeMaterial, self).__init__(**kwargs)
        self.n = n
        self.v = v
        self.lambda_ref = lambda_ref
        self.lambda_short = lambda_short
        self.lambda_long = lambda_long

    @classmethod
    def from_string(cls, txt, name=None):
        txt = str(txt)
        val = [float(_) for _ in txt.split("/")]
        if len(val) == 1:
            n, = val
            v = np.inf
        if len(val) == 2:
            n, v = val
        else:
            raise ValueError
        if name is None:
            name = "-"
        return cls(name=name, n=n, v=v)

    @simple_cache
    def refractive_index(self, wavelength):
        return (self.n + (wavelength - self.lambda_ref) /
                (self.lambda_long - self.lambda_short) *
                (1 - self.n)/self.v)

    def dict(self):
        dat = super(AbbeMaterial, self).dict()
        dat["n"] = self.n
        dat["v"] = self.v
        if self.lambda_ref != lambda_d:
            dat["lambda_ref"] = self.lambda_ref
        if self.lambda_short != lambda_F:
            dat["lambda_short"] = self.lambda_short
        if self.lambda_long != lambda_C:
            dat["lambda_long"] = self.lambda_long
        return dat


@public
class CoefficientsMaterial(Material):
    def __init__(self, coefficients, typ="sellmeier", **kwargs):
        super(CoefficientsMaterial, self).__init__(**kwargs)
        if not hasattr(self, "n_%s" % typ):
            warnings.warn("unknown dispersion %s (%s)" % (typ, self.name))
        self.typ = typ
        self.coefficients = np.atleast_1d(coefficients)

    @simple_cache
    def refractive_index(self, wavelength):
        n = getattr(self, "n_%s" % self.typ)
        n = n(wavelength/1e-6, self.coefficients)
        if self.mirror:
            n = -n
        return n

    # http://refractiveindex.info/download/database/rii-database-2015-03-11.zip
    # http://home.comcast.net/~mbiegert/Blog/DispersionCoefficient/dispeqns.pdf

    def n_schott(self, w, c):
        n = c[0] + c[1]*w**2
        for i, ci in enumerate(c[2:]):
            n += ci*w**(-2*(i + 1))
        return np.sqrt(n)

    def n_sellmeier(self, w, c):
        w2 = w**2
        c0, c1 = c.reshape(-1, 2).T
        return np.sqrt(1. + (c0*w2/(w2 - c1**2)).sum())

    def n_sellmeier_squared(self, w, c):
        w2 = w**2
        c0, c1 = c.reshape(-1, 2).T
        return np.sqrt(1. + (c0*w2/(w2 - c1)).sum())

    def n_sellmeier_squared_transposed(self, w, c):
        w2 = w**2
        c0, c1 = c.reshape(2, -1)
        return np.sqrt(1. + (c0*w2/(w2 - c1)).sum())

    def n_conrady(self, w, c):
        return c[0] + c[1]/w + c[2]/w**3.5

    def n_herzberger(self, w, c):
        l = 1./(w**2 - .028)
        return c[0] + c[1]*l + c[2]*l**2 + c[3]*w**2 + c[4]*w**4 + c[5]*w**6

    def n_sellmeier_offset(self, w, c):
        w2 = w**2
        c0, c1 = c[1:-1].reshape(-1, 2).T
        return np.sqrt(c[0] + (c0*w2/(w2 - c1**2)).sum())

    def n_sellmeier_squared_offset(self, w, c):
        w2 = w**2
        c0, c1 = c[1:-1].reshape(-1, 2).T
        return np.sqrt(c[0] + (c0*w2/(w2 - c1)).sum())

    def n_handbook_of_optics1(self, w, c):
        return np.sqrt(c[0] + (c[1]/(w**2 - c[2])) - (c[3]*w**2))

    def n_handbook_of_optics2(self, w, c):
        return np.sqrt(c[0] + (c[1]*w**2/(w**2 - c[2])) - (c[3]*w**2))

    def n_extended2(self, w, c):
        n = c[0] + c[1]*w**2 + c[6]*w**4 + c[7]*w**6
        for i, ci in enumerate(c[2:6]):
            n += ci*w**(-2*(i + 1))
        return np.sqrt(n)

    def n_hikari(self, w, c):
        n = c[0] + c[1]*w**2 + c[2]*w**4
        for i, ci in enumerate(c[3:]):
            n += ci*w**(-2*(i + 1))
        return np.sqrt(n)

    def n_gas(self, w, c):
        c0, c1 = c.reshape(2, -1)
        return 1. + (c0/(c1 - w**-2)).sum()

    def dict(self):
        dat = super(CoefficientsMaterial, self).dict()
        dat["typ"] = self.typ
        dat["coefficients"] = list(self.coefficients)
        return dat


# http://refractiveindex.info
vacuum = ModelMaterial(name="vacuum", catalog="basic", solid=False)
mirror = Material(name="mirror", catalog="basic", solid=False, mirror=True)
air = CoefficientsMaterial(
    name="air", catalog="basic", typ="gas", solid=False,
    coefficients=[.05792105, .00167917, 238.0185, 57.362])
basic = dict((m.name, m) for m in (vacuum, air, mirror))


class DefaultGlass(object):
    def __getitem__(self, key):
        return self.get(key)

    def get(self, key):
        return Material.make(key)

all_materials = DefaultGlass()
AllGlasses = all_materials
