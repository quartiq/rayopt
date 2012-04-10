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

def compress(uncompressed):
    """Compress a string to a list of output symbols."""
 
    # Build the dictionary.
    dict_size = 256
    dictionary = dict((chr(i), chr(i)) for i in xrange(dict_size))
    # in Python 3: dictionary = {chr(i): chr(i) for i in range(dict_size)}
 
    w = ""
    result = []
    for c in uncompressed:
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            result.append(dictionary[w])
            # Add wc to the dictionary.
            dictionary[wc] = dict_size
            dict_size += 1
            w = c
 
    # Output the code for w.
    if w:
        result.append(dictionary[w])
    return result
 
 
def decompress(compressed):
    """Decompress a list of output ks to a string."""
 
    # Build the dictionary.
    dict_size = 256
    dictionary = dict((chr(i), chr(i)) for i in xrange(dict_size))
    # in Python 3: dictionary = {chr(i): chr(i) for i in range(dict_size)}
 
    w = result = compressed.pop(0)
    for k in compressed:
        if k in dictionary:
            entry = dictionary[k]
        elif k == dict_size:
            entry = w + w[0]
        else:
            raise ValueError('Bad compressed k: %s' % k)
        result += entry
 
        # Add w+entry[0] to the dictionary.
        dictionary[dict_size] = w + entry[0]
        dict_size += 1
 
        w = entry
    return result
 
 
# How to use:
compressed = compress('TOBEORNOTTOBEORTOBEORNOT')
print (compressed)
decompressed = decompress(compressed)
print (decompressed)

# <codecell>

fname = "/home/rj/work/nist/pyrayopt/glass/Stockcat/thorlabs.zmf"
f = open(fname, "rb")
c = Stockcat.from_zemax(f)
# print c.lenses[:1]
import glob
cs = []
for f in glob.glob("/home/rj/work/nist/pyrayopt/glass/Stockcat/*.zmf"):
    #print f
    ci = Stockcat.from_zemax(open(f, "rb"))
    for li in ci.lenses[-10:]:
        pass
        # print li[:-1], `li[-1][:10]`
    cs.extend(ci.lenses)

# <codecell>

l1, l2 = c.lensdict["AC064-015-B"], c.lensdict["AC064-015-A"]
n = 40
s1, s2 = map(ord, l1[-1][:n]), map(ord, l2[-1][:n])
print l1[:9], l2[:9]
print map(bin, unpack("<8B", pack("<d", l1[8])))
print map(bin, unpack("<8B", pack("<d", l2[8])))
print map(bin, s1)
print map(bin, s2)

# <codecell>

zlib.decompress(l1[-1][10:], -15)

# <codecell>

"".join(compress(list(l1[-1])))

# <codecell>

import binascii
"".join(binascii.b2a_uu((""+l1[-1])[n*45:(n+1)*45]) for n in range(len(l1[-1])/45+1))

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

o, l = 0, 10000
s0 = np.concatenate([map(ord, li[-1][o:o+l]) for li in cs])

# <codecell>

s0h = s0[::2][:-1] | (s0[1::2]<<8)

# <codecell>

r = 8
s0n, s0f, s0x = plt.hist(s0, bins=range(0, 257, r), histtype="step")
n = s0.shape[0]/256.*r
sn = n**.5
plt.plot([0, 256], [n, n])
plt.plot([0, 256], [n+sn, n+sn])
plt.plot([0, 256], [n-sn, n-sn])
for i in range(0, 256, 8):
    plt.text(i, n/2, "%08i" % int(bin(i)[2:]), rotation="vertical", ha="left")

# <codecell>

print s0n[-12:-4], s0f[-13:-5]
map(bin, s0f[-13:-5])

# <codecell>

_ = plt.psd(s0n, NFFT=256)

# <codecell>

g = []
for f in glob.glob("/home/rj/work/nist/pyrayopt/glass/Glasscat/*.agf"):
    # print f
    g.append(np.fromfile(f, dtype=np.uint8))
g = np.concatenate(g)
n, o, p = plt.hist(g, bins=range(0, 257, 1), histtype="step")
co = map(chr, o[:-1])+["X"]
for ni, oi, coi in zip(n, o, co):
    pass # print ni, oi, coi

# <codecell>


