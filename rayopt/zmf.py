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

from __future__ import (print_function, absolute_import, division,
    unicode_literals)

import struct
import sqlite3
import os.path
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
        assert self.version in (1001,)
        while True:
            li = f.read(self.lens.size)
            if len(li) != self.lens.size:
                break
            li = self.lens.unpack(li)
            if li[8] > 1e4:
                raise ValueError((li, f.tell() - self.lens.size))
            ln = li[0].decode("latin1").strip("\0")
            try:
                code = self.codes[li[4]]
            except IndexError:
                code = "%i" % li[4]
            cipher = f.read(li[8])
            lens = dict(name=ln,
                x=li[1], y=li[2], elements=li[3], code=code,
                aspheric=li[5], grin=li[6], toroidal=li[7],
                efl=li[9], enpd=li[10], cipher=cipher,
                )
            self.append(lens)
        return self

    def decrypt(self, lens):
        efl, enpd = lens["efl"], lens["enpd"]
        cipher = lens["cipher"]
        cipher = np.fromstring(cipher, np.uint8)
        iv = np.cos(6.*efl + 3.*enpd)
        key = self.key(iv, len(cipher))
        plain = cipher ^ key
        if len(plain) > 100:
            nhigh = np.count_nonzero(plain & 0x80)
            if nhigh/len(plain) > .1:
                print(lens)
                print(lens["name"], lens["vendor"])
                #print(repr(plain.tostring()))
                #raise ValueError
                return ""
        return plain.tostring()
    
    @staticmethod
    def key(iv, n):
        p = np.arange(n)
        a = np.sin(17.*(p + 3.))
        b = np.cos(655.*(np.pi/180.)*iv)
        c = 13.2*(a + b + iv)*(p + 1.)
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


def zmf_to_sql(fil, db):
    f = open(fil, "rb")
    cat = Stockcat.from_zemax(f)
    vendor = os.path.splitext(os.path.basename(fil))[0]
    conn = sqlite3.connect(db)
    conn.text_factory = str
    cu = conn.cursor()
    cu.execute("""create table if not exists lenses (
        name text, vendor text, unknown1 integer, unknown2 integer,
        elements integer, code character, aspheric integer,
        toroidal integer, grin integer, efl real, enpd real,
        data blob, primary key (name, vendor))""")
    conn.commit()
    for lens in cat:
        s = cat.decrypt(lens)
        cu.execute("""insert or replace into lenses values (?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?)""", (lens["name"], vendor, lens["x"],
                lens["y"], lens["elements"], lens["code"], lens["aspheric"],
                lens["toroidal"], lens["grin"], lens["efl"], lens["enpd"], s))
    conn.commit()
    return conn

if __name__ == "__main__":
    import glob, sys
    fs = sys.argv[1:]
    if not fs:
        p = "glass/Stockcat/"
        fs = glob.glob(p + "*.zmf") + glob.glob(p + "*.ZMF")
    for f in fs:
        print(f)
        #c = stockcat_from_zmf(f)
        c = zmf_to_sql(f, "all.db")
        #print(c)

