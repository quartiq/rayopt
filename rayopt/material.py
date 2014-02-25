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

import io
import codecs
import shelve
import anydbm
import os.path
import cPickle as pickle
import glob

import numpy as np

from .utils import sfloat, sint, simple_cache


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


class Material(object):
    def __init__(self, name="-", solid=True, mirror=False, catalog=None):
        self.name = name
        self.solid = solid
        self.mirror = mirror
        self.catalog = catalog

    def __str__(self):
        if self.catalog is not None:
            return "%s/%s" % (self.catalog, self.name)
        else:
            return self.name

    @simple_cache
    def refractive_index(self, wavelength):
        return 1.

    def dispersion(self, short, mid, long):
        return (self.refractive_index(mid) - 1)/self.delta_n(short, long)

    def delta_n(self, short, long):
        return (self.refractive_index(short) - self.refractive_index(long))

 
class ModelMaterial(Material):
    def __init__(self, nd=1., vd=np.inf, **kwargs):
        super(ModelMaterial, self).__init__(**kwargs)
        self.nd = nd
        self.vd = vd

    @classmethod
    def from_string(cls, txt, name=None):
        txt = str(txt)
        v = map(float, txt.split("/"))
        if len(v) == 1:
            nd, = v
            vd = np.inf
        if len(v) == 2:
            nd, vd = v
        else:
            raise ValueError
        if name is None:
            name = "-"
        return cls(name=name, nd=nd, vd=vd)

    @simple_cache
    def refractive_index(self, wavelength):
        return (self.nd + (wavelength - lambda_d)/(lambda_C - lambda_F)
                *(1 - self.nd)/self.vd)


class SellmeierMaterial(Material):
    def __init__(self, sellmeier, thermal=None, nd=None, vd=None, **kwargs):
        super(SellmeierMaterial, self).__init__(**kwargs)
        self.sellmeier = np.atleast_2d(sellmeier)
        if nd is None:
            nd = self.refractive_index(lambda_d)
        self.nd = nd
        if vd is None:
            vd = self.dispersion(lambda_F, lambda_d, lambda_C)
        self.vd = vd
        self.thermal = thermal

    @simple_cache
    def refractive_index(self, wavelength):
        w2 = (wavelength/1e-6)**2
        c0 = self.sellmeier[:, 0]
        c1 = self.sellmeier[:, 1]
        n2 = 1. + (c0*w2/(w2 - c1)).sum()
        n = np.sqrt(n2)
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


class GasMaterial(SellmeierMaterial):
    def __init__(self, **kwargs):
        super(GasMaterial, self).__init__(solid=False, **kwargs)

    @simple_cache
    def refractive_index(self, wavelength):
        w2 = (wavelength/1e-6)**2
        c0 = self.sellmeier[:, 0]
        c1 = self.sellmeier[:, 1]
        #n2 = 1. + (c0*w2/(w2 - c1)).sum()
        #n = np.sqrt(n2)
        n  = 1. + (c0/(c1 - w2)).sum()
        if self.mirror:
            n = -n
        return n


# http://refractiveindex.info
vacuum = ModelMaterial(name="VACUUM", catalog="basic",
        nd=1., vd=np.inf, solid=False)
air = GasMaterial(name="AIR", catalog="basic",
        sellmeier=[[5792105E-8, 238.0185], [167917E-8, 57.362]])
mirror = Material(name="MIRROR", catalog="basic",
        mirror=True, solid=False)

basic = dict((m.name, m) for m in (vacuum, air, mirror))


def load_catalogs(catdb, catalogs):
    kw = dict(protocol=pickle.HIGHEST_PROTOCOL, writeback=False)
    try:
        db = shelve.open(catdb, "r", **kw)
        if not db.keys():
            db.close()
            raise anydbm.error
    except anydbm.error:
        # keeping it open writeable corrupts it
        db = shelve.open(catdb, "c", **kw)
        defaults = {}
        for f in catalogs:
            _, name = os.path.split(f)
            name, ext = os.path.splitext(name.lower())
            if ext == ".agf":
                cf = load_catalog_zemax(f, name)
            elif ext == ".glc":
                cf = load_catalog_oslo(f, name)
            else:
                raise ValueError("glass catalog extension %s unknown" %
                        ext)
            for k, v in cf.items():
                db[("%s/%s" % (name, k.lower())).encode("ascii")] = v
                defaults[k.lower()] = name
        name = "basic"
        for k, v in basic.items():
            db["%s/%s" % (name, k.lower())] = v
            defaults[k.lower()] = name
        db["__default__"] = defaults
        db.close()
        db = shelve.open(catdb, "r", **kw)
    return db

def load_default_catalogs():
    global catalogs
    dir, _ = os.path.split(__file__)
    dir = os.path.join(dir, "../glass")
    cats = []
    for ext in "*.agf", "*.glc":
        for ext in ext.upper(), ext.lower():
            pattern = os.path.join(dir, ext)
            cats.extend(sorted(glob.glob(pattern)))
    catdb = os.path.join(dir, "all.shelve")
    catalogs = load_catalogs(catdb, cats)
    return catalogs

catalogs = {}
load_default_catalogs()

def get_material(name):
    if isinstance(name, Material):
        return name
    if type(name) is type(1.):
        return ModelMaterial(nd=name)
    if type(name) is tuple:
        return ModelMaterial(nd=name[0], vd=name[1])
    try:
        return ModelMaterial.from_string(name)
    except ValueError:
        pass
    name = str(name).lower()
    if "/" in name:
        return catalogs[name]
    else:
        cat = catalogs["__default__"][name]
        return catalogs["%s/%s" % (cat, name)]

class DefaultGlass(object):
    def __getitem__(self, key):
        cat = catalogs["__defaults__"][key]
        return catalogs["%s/%s" % (cat, key)]

all_materials = DefaultGlass()
AllGlasses = all_materials
