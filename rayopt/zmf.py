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

import struct
from StringIO import StringIO

import numpy as np

from rayopt.formats import system_from_zemax


class Stockcat(list):
    stockcat = struct.Struct("< I")
    lens = struct.Struct("< 100s H H I I I I I I d d")
    codes = "?EBPM"
    
    @classmethod
    def from_zemax(cls, f):
        self = cls()
        stockcat = f.read(self.stockcat.size)
        stockcat = self.stockcat.unpack(stockcat)
        self.version = stockcat[0]
        while True:
            try:
                li = f.read(self.lens.size)
                li = self.lens.unpack(li)
            except struct.error:
                break
            ln = li[0].strip("\0")
            try:
                code = self.codes[li[4]]
            except IndexError:
                code = li[4]
            cipher = f.read(li[8])
            lens = dict(name=ln,
                vendor=li[1], x=li[2], elements=li[3], code=code,
                asphere=li[5], grin=li[6], toroidal=li[7],
                efl=li[9], enpd=li[10], cipher=cipher,
                )
            self.append(lens)
        return self

    def decrypt(self, lens):
        efl, enpd = lens["efl"], lens["enpd"]
        cipher = lens["cipher"]
        cipher = np.fromstring(cipher, np.uint8)
        iv = np.cos(6*efl + 3*enpd)
        key = self.key(iv, len(cipher))
        plain = cipher ^ key
        nhigh = np.count_nonzero(plain & 0x80)
        assert nhigh/max(1, len(plain)) < .1, plain.tostring()
        return plain.tostring()
    
    @staticmethod
    def key(iv, n):
        p = np.arange(n)
        a = np.sin(17*(p + 3))
        b = np.cos(655*np.pi/180*iv)
        c = 13.2*(a + b + iv)*(p + 1)
        d = (int(("%.8E" % _)[4:7]) for _ in c)
        e = np.fromiter(d, np.uint8, len(c))
        return e


def stockcat_from_zmf(fil):
    f = open(fil, "rb")
    c = Stockcat.from_zemax(f)
    for k in c:
        s = c.decrypt(k)
        s = StringIO(s)
        s = system_from_zemax(s)
        k["system"] = s
    return c


if __name__ == "__main__":
    import glob, sys
    fs = sys.argv[1:]
    if not fs:
        p = "glass/Stockcat/"
        fs = glob.glob(p + "*.zmf") + glob.glob(p + "*.ZMF")
    for f in fs:
        c = stockcat_from_zmf(f)
        #print(c)

