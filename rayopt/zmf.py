# -*- coding: utf-8 -*-
# <nbformat>3</nbformat>

# <codecell>

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

from struct import Struct, error, pack, unpack
import codecs, zlib

# <codecell>

class Stockcat(object):
    # version
    stockcat = Struct("< I")

    # 100c lens name 30120
    # int database code 45094
    # int 1
    # int elements 3
    # int code 0 1
    # int blob length (after efl and length) 903
    # double efl 25.39
    # double enpd 14.5
    lens = Struct("< 100s H H I I I I I I d d")

    @classmethod
    def from_zemax(cls, f):
        self = cls()
        self.catalog = self.stockcat.unpack(f.read(self.stockcat.size))
        self.lenses = []
        self.lensdict = {}
        while True:
            try:
                li = self.lens.unpack(f.read(self.lens.size))
            except error:
                break
            ln = li[0].strip("\0")
            ld = f.read(li[8])
            self.lenses.append((ln,) + li[1:] + (ld,))
            self.lensdict[ln] = li[1:] + (ld,)
        return self

# <codecell>

fname = "/home/rjordens/work/nist/pyrayopt/glass/Stockcat/thorlabs.zmf"
f = open(fname, "rb")
c = Stockcat.from_zemax(f)
# print c.lenses[:1]

# <codecell>

l1, l2 = c.lensdict["AC064-015-B"], c.lensdict["AC064-015-C"]
n = 10
s1, s2 = map(ord, l1[-1][:10]), map(ord, l2[-1][:10])
print map(bin, s1)
print map(bin, s2)

# <codecell>

print map(bin, unpack("<8B", pack("<d", l1[8])))
print map(bin, unpack("<8B", pack("<d", l2[8])))

# <codecell>

zlib.decompress(l1[-1][10:])

# <codecell>

codecs.getdecoder("zip")(l1[-1])

# <codecell>

codecs.getencoder("zip")("2322324")

# <codecell>

import binascii
binascii.b2a_uu((""+l1[-1])[:45])

# <codecell>

import zlib
import gzip
for i in range(500):
    try:
        print zlib.decompress(l1[-1][i:], -15)
    except:
        break
    print i

# <codecell>

map(lambda c: bin(ord(c)), l1[-1][:10])

# <codecell>

s=l1[-1]
s.decode("zip")

# <codecell>


