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

from .utils import simple_cache
from .name_mixin import NameMixin


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


class Material(NameMixin):
    def __init__(self, name="-", solid=True, mirror=False, catalog=None):
        self.name = name
        self.solid = solid
        self.mirror = mirror
        self.catalog = catalog

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
        name = name.lower().split("/")
        if len(name) == 3:
            source, catalog, name = name
        elif len(name) == 2:
            source = None
            catalog, name = name
        else:
            source, catalog = None, None
            name, = name
        if catalog in (None, "basic"):
            try:
                return basic[name]
            except KeyError:
                pass
        from .library import Library
        lib = Library.one()
        return lib.get("glass", name, catalog)

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
        return dat

    @simple_cache
    def refractive_index(self, wavelength):
        return 1.

    def dispersion(self, short, mid, long):
        return (self.refractive_index(mid) - 1)/self.delta_n(short, long)

    def delta_n(self, short, long):
        return (self.refractive_index(short) - self.refractive_index(long))


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
        return (self.n + (wavelength - self.lambda_ref)
            /(self.lambda_long - self.lambda_short)
            *(1 - self.n)/self.v)

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


class SellmeierMaterial(Material):
    def __init__(self, sellmeier, thermal=None, **kwargs):
        super(SellmeierMaterial, self).__init__(**kwargs)
        self.sellmeier = np.atleast_2d(sellmeier)
        self.thermal = thermal

    @simple_cache
    def refractive_index(self, wavelength):
        w2 = (wavelength/1e-6)**2
        c0, c1 = self.sellmeier.T
        n = np.sqrt(1. + (c0*w2/(w2 - c1)).sum())
        if self.mirror:
            n = -n
        return n

    def dn_thermal(self, t, n, wavelength):
        d0, d1, d2, e0, e1, tref, lref = self.thermal
        dt = t-tref
        w = wavelength/1e-6
        dn = (n**2-1)/(2*n)*(d0*dt+d1*dt**2+d2*dt**3+
                (e0*dt+e1*dt**2)/(w**2-lref**2))
        return dn

    def dict(self):
        dat = super(SellmeierMaterial, self).dict()
        dat["sellmeier"] = [list(_) for _ in self.sellmeier]
        if self.thermal:
            dat["thermal"] = self.thermal
        return dat


class GasMaterial(SellmeierMaterial):
    def __init__(self, **kwargs):
        super(GasMaterial, self).__init__(solid=False, **kwargs)

    @simple_cache
    def refractive_index(self, wavelength):
        w2 = (wavelength/1e-6)**-2
        c0, c1 = self.sellmeier.T
        n  = 1. + (c0/(c1 - w2)).sum()
        if self.mirror:
            n = -n
        return n


# http://refractiveindex.info
vacuum = ModelMaterial(name="vacuum", catalog="basic",
        solid=False)
mirror = Material(name="mirror", catalog="basic",
        solid=False, mirror=True)
air = GasMaterial(name="air", catalog="basic",
        sellmeier=[[5792105E-8, 238.0185], [167917E-8, 57.362]])

basic = dict((m.name, m) for m in (vacuum, air, mirror))


class DefaultGlass(object):
    def __getitem__(self, key):
        return self.get(key)
    def get(self, key):
        return Material.make(key)

all_materials = DefaultGlass()
AllGlasses = all_materials
