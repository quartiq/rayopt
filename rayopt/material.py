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

import shelve, os.path, glob, cPickle as pickle

import numpy as np


def sfloat(a):
    try: return float(a)
    except: return None


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


def simple_cache(f):
    cache = {}
    def wrapper(self, *args):
        key = self, args
        if key not in cache:
            cache[key] = f(self, *args)
        return cache[key]
    wrapper.cache = cache
    return wrapper


class Material(object):
    def __init__(self, name="", solid=True, mirror=False, sellmeier=None,
            nd=1., vd=np.inf, thermal=None):
        self.name = name
        self.solid = solid
        self.mirror = mirror
        self.thermal = thermal
        self.nd = nd
        self.vd = vd
        if sellmeier is not None:
            self.sellmeier = np.atleast_2d(sellmeier)
            self.nd = self.refractive_index(lambda_d)
            self.vd = ((self.nd - 1)/(
                self.refractive_index(lambda_F)-
                self.refractive_index(lambda_C)))
        else:
            self.sellmeier = None

    @classmethod
    def from_string(cls, txt, name=None):
        v = map(float, txt.split("/"))
        if len(v) == 1:
            nd, = v
            vd = np.inf
        if len(v) == 2:
            nd, vd = v
        else:
            raise ValueError
        if name is None:
            name = txt
        return cls(name=txt, solid=nd>1, nd=nd, vd=vd)

    def __str__(self):
        return self.name

    @simple_cache
    def refractive_index(self, wavelength):
        if self.sellmeier is None:
            return self.nd*np.ones_like(wavelength)
        w2 = (np.array(wavelength)/1e-6)**2
        c0 = self.sellmeier[:, (0,)]
        c1 = self.sellmeier[:, (1,)]
        n2 = 1. + (c0*w2/(w2 - c1)).sum(0)
        n = np.sqrt(n2)
        if self.mirror:
            n = -n
        return n.reshape(w2.shape)

    def dispersion(self, wavelength_short, wavelength_mid,
            wavelength_long):
        if self.sellmeier is None:
            return self.vd
        return (self.refractive_index(wavelength_mid) - 1)/(
                self.refractive_index(wavelength_short)-
                self.refractive_index(wavelength_long))

    def delta_n(self, wavelength_short, wavelength_long):
        if self.sellmeier is None:
            return (self.nd - 1)/self.vd
        return (self.refractive_index(wavelength_short)-
                self.refractive_index(wavelength_long))

    def dn_thermal(self, t, n, wavelength):
        d0, d1, d2, e0, e1, tref, lref = self.thermal
        dt = t-tref
        w = wavelength/1e-6
        dn = (n**2-1)/(2*n)*(d0*dt+d1*dt**2+d2*dt**3+
                (e0*dt+e1*dt**2)/(w**2-lref**2))
        return dn


class Gas(Material):
    def __init__(self, solid=False, **kwargs):
        super(Gas, self).__init__(solid=solid, **kwargs)

    @simple_cache
    def refractive_index(self, wavelength):
        if self.sellmeier is None:
            return self.nd*np.ones_like(wavelength)
        w2 = (np.array(wavelength)/1e-6)**2
        c0 = self.sellmeier[:, (0,)]
        c1 = self.sellmeier[:, (1,)]
        #n2 = 1. + (c0*w2/(w2 - c1)).sum(0)
        #n = np.sqrt(n2)
        n  = 1. + (c0/(c1 - w2)).sum(0)
        if self.mirror:
            n = -n
        return n.reshape(w2.shape)

# http://refractiveindex.info
vacuum = Gas(name="vacuum", nd=1., vd=np.inf)
vacuum_mirror = Gas(name="vacuum_mirror", mirror=True, nd=-1., vd=np.inf)
air = Gas(name="air", sellmeier=[[5792105E-8, 238.0185], [167917E-8, 57.362]])
air_mirror = Gas(name="air_mirror", mirror=True, sellmeier=air.sellmeier)

basics = dict((m.name, m) for m in (vacuum, vacuum_mirror, air, air_mirror))


def load_catalog_zemax(fil, name=None):
    catalog = {}
    dat = open(fil)
    for line in dat:
        try:
            cmd, args = line.split(" ", 1)
            if cmd == "CC":
                pass # obj.name = args
            elif cmd == "NM":
                args = args.split()
                g = Material(name=args[0], nd=sfloat(args[3]),
                        vd=sfloat(args[4]))
                g.glasscode = sfloat(args[2])
                g.catalog = name
                catalog[g.name] = g
            elif cmd == "GC":
                g.comment = args
            elif cmd == "ED":
                args = map(sfloat, args.split())
                g.alpham3070, g.alpha20300, g.density = args[0:3]
            elif cmd == "CD":
                s = np.array(map(sfloat, args.split())).reshape((-1,2))
                g.sellmeier = np.array([si for si in s if not si[0] == 0])
            elif cmd == "TD":
                s = map(sfloat, args.split())
                g.thermal = s
            elif cmd == "OD":
                g.chemical = map(sfloat, s[1:])
                g.price = sfloat(s[0])
            elif cmd == "LD":
                s = map(sfloat, args.split())
                pass
            elif cmd == "IT":
                s = map(sfloat, args.split())
                if not hasattr(g, "transmission"):
                    g.transmission = {}
                g.transmission[(s[0], s[2])] = s[1]
            else:
                print cmd, args, "not handled"
        except Exception, e:
            print cmd, args, "failed parsing", e
    catalog[g.name] = g
    return catalog


def load_catalogs(all, catalogs):
    try:
        db = shelve.open(all, "r", protocol=pickle.HIGHEST_PROTOCOL,
                writeback=False)
        if not db.keys():
            db.close()
            raise
    except:
        db = shelve.open(all, "c", protocol=pickle.HIGHEST_PROTOCOL,
                writeback=False)
        for f in catalogs:
            _, name = os.path.split(f)
            name, _ = os.path.splitext(name)
            cf = load_catalog_zemax(f, name)
            db.update(cf)
        db.update(basics)
        db.close()
        db = shelve.open(all, "r", protocol=pickle.HIGHEST_PROTOCOL,
                writeback=False)
    return db


catpath = "/home/rj/work/nist/pyrayopt/glass/"
cats = "misc infrared schott ohara hoya corning heraeus hikari sumita"
cats = cats.split()[::-1]
cats = [os.path.join(catpath, "%s.agf" % _) for _ in cats]
all = os.path.join(catpath, "all.shelve")
all_materials = load_catalogs(all, cats)
