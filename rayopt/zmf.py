# -*- coding: utf-8 -*-
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

from struct import Struct, unpack, error
from StringIO import StringIO

import numpy as np

from rayopt.formats import system_from_zemax


class Stockcat(object):
    stockcat = Struct("< I")
    lens = Struct("< 100s H H I I I I I I d d")
    codes = "?EBPM"
    
    @classmethod
    def from_zemax(cls, f):
        self = cls()
        stockcat = self.stockcat.unpack(f.read(self.stockcat.size))
        self.version = stockcat[0]
        self.lenses = []
        self.lensdict = {}
        while True:
            i = f.tell()
            try:
                li = self.lens.unpack(f.read(self.lens.size))
                ld = f.read(li[8])
            except error:
                break
            ln = li[0].strip("\0")
            code = self.codes[li[4]]
            efl, enpd = li[9:]
            data = self.decrypt(efl, enpd, ld)
            lens = dict(name=ln,
                vendor=li[1], x=li[2], elements=li[3], code=code,
                asphere=li[5], grin=li[6], toroidal=li[7],
                efl=efl, enpd=enpd, data=data,
                )
            self.lenses.append(lens)
            self.lensdict[ln] = lens
        return self

    def decrypt(self, efl, enpd, cipher):
        cipher = np.fromstring(cipher, np.uint8)
        iv = np.cos(6*efl + 3*enpd)
        key = self.key(iv, len(cipher))
        plain = cipher ^ key
        assert np.all(plain & 0x80 == 0)
        return plain.tostring()
    
    @staticmethod
    def key(iv, n):
        p = np.arange(n)
        a = np.sin(17*(p + 3))
        b = np.cos(655*np.pi/180*iv)
        c = 13.2*(a + b + iv)*(p + 1)
        d = np.array([int(("%.8E" % _)[4:7]) for _ in c])
        return d.astype(np.uint8)


def stockcat_from_zmf(fil):
    f = open(fil, "rb")
    c = Stockcat.from_zemax(f)
    for k in c.lenses:
        d = StringIO(k["data"])
        s = system_from_zemax(d)
        k["system"] = s
    return c


if __name__ == "__main__":
    import glob
    p = "glass/Stockcat/"
    for f in glob.glob(p + "*.zmf") + glob.glob(p + "*.ZMF"):
        try:
            c = stockcat_from_zmf(f)
            print(c)
        except Exception, e:
            print(f, e)

