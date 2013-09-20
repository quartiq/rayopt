# -*- coding: utf8 -*-
#
#   pyrayopt - raytracing for optical imaging systems
#
#   Copyright (C) 2011-2013 Robert Jordens <jordens@phys.ethz.ch>
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

# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

from __future__ import division
import cython
import numpy as np
cimport numpy as np
np.import_array()
from libc.math cimport sqrt, fabs, M_PI

dtype = np.double
ctypedef np.double_t dtype_t
ctypedef int intc_t
