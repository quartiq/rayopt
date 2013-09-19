# -*- coding: utf8 -*-
#
#   pyrayopt - raytracing for optical imaging systems
#   Copyright (C) 2013 Robert Jordens <jordens@phys.ethz.ch>
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

import numpy as np


def dir_to_angles(r):
    return r/np.linalg.norm(r)

def tanarcsin(u):
    if u.ndim == 2 and u.shape[1] == 3:
        return u[:, :2]/u[:, 2:]
    u2 = np.square(u)
    if u.ndim == 2:
        u2 = u2[:, :2].sum(axis=1)[:, None]
    return u/np.sqrt(1 - u2)

def sinarctan(u):
    u2 = np.square(u)
    if u.ndim == 2:
        assert u.shape[1] < 3
        u2 = u2[:, :2].sum(axis=1)[:, None]
    return u/np.sqrt(1 + u2)



