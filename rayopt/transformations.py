# -*- coding: utf-8 -*-
# transformations.py

# Copyright (c) 2006, Christoph Gohlke
# Copyright (c) 2006-2008, The Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Homogeneous Transformation Matrices and Quaternions.

A library for calculating 4x4 matrices for translating, rotating, mirroring,
scaling, shearing, projecting, orthogonalizing, and superimposing arrays of
homogenous coordinates as well as for converting between rotation matrices,
Euler angles, and quaternions.

:Authors:
  `Christoph Gohlke <http://www.lfd.uci.edu/~gohlke/>`__,
  Laboratory for Fluorescence Dynamics, University of California, Irvine

:Version: 20081223

Requirements
------------

* `Python 2.5 <http://www.python.org>`__
* `Numpy 1.2 <http://numpy.scipy.org>`__
* `vlfdlib.c 20081121 <http://www.lfd.uci.edu/~gohlke/>`__
  (optional for faster quaternion functions)

Notes
-----

Matrices (M) can be inverted using numpy.linalg.inv(M), concatenated using
numpy.dot(M0, M1), or used to transform homogeneous coordinates (v) using
numpy.dot(v, M) for shape (4, \*) "point of arrays", respectively
numpy.dot(v, M.T) for shape (\*, 4) "array of points".

Calculations are carried out with numpy.float64 precision.

Vector, point, quaternion, and matrix function arguments are expected to be
"array like", i.e. tuple, list, or numpy arrays.

Return types are numpy arrays unless specified otherwise.

Angles are in radians unless specified otherwise.

Quaternions ix+jy+kz+w are represented as [x, y, z, w].

A triple of Euler angles can be applied/interpreted in 24 ways, which can
be specified using a 4 character string or encoded 4-tuple:

  *Axes 4-string*: e.g. 'sxyz' or 'ryxy'

  - first character : rotations are applied to 's'tatic or 'r'otating frame
  - remaining characters : successive rotation axis 'x', 'y', or 'z'

  *Axes 4-tuple*: e.g. (0, 0, 0, 0) or (1, 1, 1, 1)

  - inner axis: code of axis ('x':0, 'y':1, 'z':2) of rightmost matrix.
  - parity : even (0) if inner axis 'x' is followed by 'y', 'y' is followed
    by 'z', or 'z' is followed by 'x'. Otherwise odd (1).
  - repetition : first and last axis are same (1) or different (0).
  - frame : rotations are applied to static (0) or rotating (1) frame.

References
----------

(1) Matrices and transformations. Ronald Goldman.
    In "Graphics Gems I", pp 472-475. Morgan Kaufmann, 1990.
(2) More matrices and transformations: shear and pseudo-perspective.
    Ronald Goldman. In "Graphics Gems II", pp 320-323. Morgan Kaufmann, 1991.
(3) Decomposing a matrix into simple transformations. Spencer Thomas.
    In "Graphics Gems II", pp 320-323. Morgan Kaufmann, 1991.
(4) Recovering the data from the transformation matrix. Ronald Goldman.
    In "Graphics Gems II", pp 324-331. Morgan Kaufmann, 1991.
(5) Euler angle conversion. Ken Shoemake.
    In "Graphics Gems IV", pp 222-229. Morgan Kaufmann, 1994.
(6) Arcball rotation control. Ken Shoemake.
    In "Graphics Gems IV", pp 175-192. Morgan Kaufmann, 1994.
(7) A discussion of the solution for the best rotation to relate two sets
    of vectors. W Kabsch. Acta Cryst. 1978. A34, 827-828.

Examples
--------

>>> alpha, beta, gamma = 0.123, -1.234, 2.345
>>> origin, xaxis, yaxis, zaxis = (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)
>>> I = identity_matrix()
>>> Rx = rotation_matrix(alpha, xaxis)
>>> Ry = rotation_matrix(beta, yaxis)
>>> Rz = rotation_matrix(gamma, zaxis)
>>> R = concatenate_transforms(Rx, Ry, Rz)
>>> euler = euler_from_matrix(R, 'rxyz')
>>> numpy.allclose([alpha, beta, gamma], euler)
True
>>> Re = euler_matrix(alpha, beta, gamma, 'rxyz')
>>> is_same_transform(R, Re)
True
>>> al, be, ga = euler_from_matrix(Re, 'rxyz')
>>> is_same_transform(Re, euler_matrix(al, be, ga, 'rxyz'))
True
>>> qx = quaternion_about_axis(alpha, xaxis)
>>> qy = quaternion_about_axis(beta, yaxis)
>>> qz = quaternion_about_axis(gamma, zaxis)
>>> q = quaternion_multiply(qx, qy)
>>> q = quaternion_multiply(q, qz)
>>> Rq = quaternion_matrix(q)
>>> is_same_transform(R, Rq)
True
>>> S = scaling_matrix(1.23, origin)
>>> T = translation_matrix((1, 2, 3))
>>> Z = shear_matrix(beta, xaxis, origin, zaxis)
>>> R = random_rotation_matrix(numpy.random.rand(3))
>>> M = concatenate_transforms(T, R, Z, S)
>>> scale, shear, angles, trans, persp = decompose_matrix(M)
>>> numpy.allclose(scale, 1.23)
True
>>> numpy.allclose(trans, (1, 2, 3))
True
>>> numpy.allclose(shear, (0, math.tan(beta), 0))
True
>>> is_same_transform(R, euler_matrix(axes='sxyz', *angles))
True
>>> v0 = numpy.random.rand(10, 4)
>>> v0[:, 3] = 0
>>> M = numpy.dot(T, R)
>>> v1 = numpy.dot(v0, M.T) + numpy.random.normal(0, 1e-9, 10).reshape(-1, 1)
>>> M = superimpose_matrix(v0, v1)
>>> numpy.allclose(v1, numpy.dot(v0, M.T))
True

"""

from __future__ import division

import warnings
import math

import numpy

# Documentation in HTML format can be generated with Epydoc
__docformat__ = "restructuredtext en"

_EPS = numpy.finfo(float).eps * 4.0


def concatenate_transforms(*matrices):
    """Return concatenation of series of transformation matrices.

    >>> M = numpy.arange(16.).reshape((4, 4))
    >>> numpy.allclose(M, concatenate_transforms(M))
    True
    >>> numpy.allclose(numpy.dot(M, M.T), concatenate_transforms(M, M.T))
    True

    """
    M = numpy.identity(4, dtype=numpy.float64)
    for i in matrices:
        M = numpy.dot(M, i)
    return M


def is_same_transform(matrix0, matrix1):
    """Return True if two matrices perform same transformation.

    >>> is_same_transform(numpy.identity(4), numpy.identity(4))
    True

    """
    v = numpy.identity(4, dtype=numpy.float64)[:3]
    return numpy.allclose(numpy.dot(v, matrix0.T), numpy.dot(v, matrix1.T))


def norm(vector):
    """Return length of vector, i.e. its euclidean norm.

    >>> n = norm((1, -2, 3))
    >>> numpy.allclose(n, 3.74165738677394)
    True

    """
    # return numpy.linalg.norm(vector)
    return numpy.sqrt(numpy.dot(vector, vector))


def identity_matrix():
    """Return 4x4 identity/unit matrix.

    >>> I = identity_matrix()
    >>> numpy.allclose(I, numpy.dot(I, I))
    True
    >>> numpy.sum(I), numpy.trace(I)
    (4.0, 4.0)

    """
    return numpy.identity(4, dtype=numpy.float64)


def translation_matrix(direction):
    """Return matrix to translate by direction vector.

    >>> v = numpy.arange(5.)
    >>> numpy.allclose(v[:3], translation_matrix(v)[:3, 3])
    True

    """
    M = numpy.identity(4, dtype=numpy.float64)
    M[:3, 3] = direction[:3]
    return M


def translation_from_matrix(matrix):
    """Return translation vector from translation matrix.

    >>> v0 = numpy.arange(3.)
    >>> v1 = translation_from_matrix(translation_matrix(v0))
    >>> numpy.allclose(v0, v1)
    True

    """
    return numpy.array(matrix, copy=False)[:3, 3].copy()


def mirror_matrix(point, normal):
    """Return matrix to mirror at plane defined by point and normal vector.

    >>> v = numpy.arange(1., 4.)
    >>> numpy.allclose(2., numpy.trace(mirror_matrix(v, v)))
    True
    >>> numpy.allclose(2., numpy.trace(mirror_matrix(v, tuple(v))))
    True

    """
    normal = numpy.array(normal[:3], dtype=numpy.float64, copy=True)
    normal /= norm(normal)
    M = numpy.identity(4, dtype=numpy.float64)
    M[:3, :3] -= 2.0 * numpy.outer(normal, normal)
    M[:3, 3] = 2.0 * numpy.dot(point[:3], normal) * normal
    return M


def mirror_from_matrix(matrix):
    """Return mirror plane point and normal vector from mirror matrix.

    >>> M0 = mirror_matrix((1, -2, 3), (-4, 5, 6))
    >>> point, normal = mirror_from_matrix(M0)
    >>> M1 = mirror_matrix(point, normal)
    >>> is_same_transform(M0, M1)
    True

    """
    M = numpy.array(matrix, dtype=numpy.float64, copy=False).T
    # normal: unit eigenvector corresponding to eigenvalue -1
    l, V = numpy.linalg.eig(M[:3, :3])
    i = numpy.where(abs(numpy.real(l) + 1.0) < 1e-9)[0][0]
    normal = numpy.real(V[:, i]).squeeze()
    # point: any unit eigenvector corresponding to eigenvalue 1
    l, V = numpy.linalg.eig(M)
    i = numpy.where(abs(numpy.real(l) - 1.0) < 1e-9)[0][-1]
    point = numpy.real(V[:, i]).squeeze()
    return point, normal


def rotation_matrix(angle, direction=(1, 0, 0), point=None):
    """Return matrix to rotate about axis defined by point and direction.

    >>> angle = 0.123
    >>> R0 = rotation_matrix(angle, (1, 2, 3), (-6, 5, 4))
    >>> R1 = rotation_matrix(angle-2*math.pi, (1, 2, 3), (-6, 5, 4))
    >>> is_same_transform(R0, R1)
    True
    >>> R0 = rotation_matrix(angle, (1, 2, 3), (-6, 5, 4))
    >>> R1 = rotation_matrix(-angle, (-1, -2, -3), (-6, 5, 4))
    >>> is_same_transform(R0, R1)
    True
    >>> I = numpy.identity(4, float)
    >>> v = numpy.arange(1., 5.)
    >>> numpy.allclose(I, rotation_matrix(math.pi*2, v))
    True
    >>> numpy.allclose(I, rotation_matrix(math.pi*2, v, tuple(v)))
    True
    >>> numpy.allclose(I, rotation_matrix(math.pi*2, tuple(v), v))
    True
    >>> numpy.allclose(2., numpy.trace(rotation_matrix(math.pi/2, v, v[::-1])))
    True

    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = numpy.array(direction[:3], dtype=numpy.float64, copy=True)
    direction /= norm(direction)
    # rotation matrix around unit vector
    R = numpy.identity(3, dtype=numpy.float64)
    R *= cosa
    R += numpy.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += numpy.array((( 0.0,         -direction[2],  direction[1]),
                      ( direction[2], 0.0,          -direction[0]),
                      (-direction[1], direction[0],  0.0)),
                      dtype=numpy.float64)
    M = numpy.identity(4, dtype=numpy.float64)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = numpy.array(point[:3], dtype=numpy.float64, copy=False)
        M[:3, 3] = point - numpy.dot(R, point)
    return M


def rotation_from_matrix(matrix):
    """Return rotation angle and axis from rotation matrix.

    >>> R0 = rotation_matrix(0.123, (1, -2, 1), (-1, 2, -3))
    >>> angle, direction, point = rotation_from_matrix(R0)
    >>> R1 = rotation_matrix(angle, direction, point)
    >>> is_same_transform(R0, R1)
    True

    """
    R = numpy.array(matrix, dtype=numpy.float64, copy=False)
    R33 = R[:3, :3]
    # direction: unit eigenvector of R33 corresponding to eigenvalue of 1
    l, W = numpy.linalg.eig(R33.T)
    i = numpy.where(abs(numpy.real(l) - 1.0) < 1e-9)[0][-1]
    direction = numpy.real(W[:, i]).squeeze()
    # point: unit eigenvector of R33 corresponding to eigenvalue of 1
    l, Q = numpy.linalg.eig(R)
    i = numpy.where(abs(numpy.real(l) - 1.0) < 1e-9)[0][-1]
    point = numpy.real(Q[:3, i]).squeeze()
    # rotation angle depending on direction
    cosa = (numpy.trace(R33) - 1.0) / 2.0
    if abs(direction[2]) > 1e-9:
        sina = (R[1, 0] + (cosa-1.0)*direction[0]*direction[1]) / direction[2]
    elif abs(direction[1]) > 1e-9:
        sina = (R[0, 2] + (cosa-1.0)*direction[0]*direction[2]) / direction[1]
    else:
        sina = (R[2, 1] + (cosa-1.0)*direction[1]*direction[2]) / direction[0]
    angle = math.atan2(sina, cosa)
    return angle, direction, point


def scaling_matrix(factor, origin=(0, 0, 0), direction=None):
    """Return matrix to scale by factor around origin in direction.

    Use factor -1 for point symmetry.

    >>> S = scaling_matrix(-1)
    >>> S = scaling_matrix(-1, (0, 1, 2))
    >>> S = scaling_matrix(-1, (0, 1, 2), (2, 3, 4))

    """
    origin = numpy.array(origin[:3], dtype=numpy.float64, copy=True)
    if direction is None:
        # uniform scaling
        M = numpy.identity(4, dtype=numpy.float64)
        M *= factor
        M[:3, 3] = (1.0-factor) * origin
        M[3, 3] = 1.0
    else:
        # nonuniform scaling
        direction = numpy.array(direction[:3], dtype=numpy.float64, copy=True)
        direction /= norm(direction)
        M = numpy.identity(4, dtype=numpy.float64)
        M[:3, :3] -= (1.0-factor) * numpy.outer(direction, direction)
        M[:3, 3] = ((1.0-factor) * numpy.dot(origin, direction)) * direction
    return M


def scaling_from_matrix(matrix, uniform=False):
    """Return scaling factor, origin and direction from scaling matrix.

    >>> S0 = scaling_matrix(-1.234, (1, -2, 3))
    >>> factor, origin, direction = scaling_from_matrix(S0)
    >>> S1 = scaling_matrix(factor, origin, direction)
    >>> is_same_transform(S0, S1)
    True
    >>> S0 = scaling_matrix(1.234, (1, -2, 3), direction=(-4, 5, 6))
    >>> factor, origin, direction = scaling_from_matrix(S0)
    >>> S1 = scaling_matrix(factor, origin, direction)
    >>> is_same_transform(S0, S1)
    True

    """
    M = numpy.array(matrix, dtype=numpy.float64, copy=False).T
    M33 = M[:3, :3]
    factor = numpy.trace(M33) - 2.0
    try:
        # direction: unit eigenvector corresponding to eigenvalue factor
        l, V = numpy.linalg.eig(M33)
        i = numpy.where(abs(numpy.real(l) - factor) < 1e-9)[0][0]
        direction = numpy.real(V[:, i]).squeeze()
    except IndexError:
        # uniform scaling
        factor = (factor + 2.0) / 3.0
        direction = None
    # origin: any eigenvector corresponding to eigenvalue 1
    l, V = numpy.linalg.eig(M)
    i = numpy.where(abs(numpy.real(l) - 1.0) < 1e-9)[0][-1]
    origin = numpy.real(V[:3, i]).squeeze()
    return factor, origin, direction


def projection_matrix(point, normal, direction=None,
                      perspective=None, pseudo=False):
    """Return matrix to project onto plane defined by point and normal.

    Using either perspective point, projection direction, or none of both.

    If pseudo is True, perspective projections will preserve relative depth.

    >>> P = projection_matrix((0, 0, 0), (1, 0, 0))
    >>> numpy.allclose(P[1:, 1:], identity_matrix()[1:, 1:])
    True
    >>> P = projection_matrix((0, 1, 2), (3, 4, 5))
    >>> P = projection_matrix((0, 1, 2), (3, 4, 5), direction=(6, 7, 8))
    >>> P = projection_matrix((0, 1, 2), (3, 4, 5), perspective=(6, 7, 8))
    >>> P = projection_matrix((0, 1, 2), (3, 4, 5), perspective=(6, 7, 8),
    ...                                             pseudo=True)

    """
    M = numpy.identity(4, dtype=numpy.float64)
    point = numpy.array(point[:3], dtype=numpy.float64, copy=False)
    normal = numpy.array(normal[:3], dtype=numpy.float64, copy=True)
    normal /= norm(normal)
    if perspective is not None:
        # perspective projection
        perspective = numpy.array(perspective[:3], dtype=numpy.float64,
                                  copy=False)
        if pseudo:
            # preserve relative depth
            M[:3, :3] *= numpy.dot(point-perspective, normal)
            M[:3, :3] += numpy.outer(normal, perspective) #.T ?
            M[:3, :3] += numpy.outer(normal, normal)
            M[:3, 3] = - (numpy.dot(point, normal) * (perspective+normal))
            M[3, :3] = normal
            M[3, 3] = -numpy.dot(perspective, normal)
        else:
            M[:3, :3] *= numpy.dot(perspective-point, normal)
            M[:3, :3] -= numpy.outer(normal, perspective)
            M[:3, 3] = numpy.dot(point, normal) * perspective
            M[3, :3] = -normal
            M[3, 3] = numpy.dot(perspective, normal)
    elif direction is not None:
        # parallel projection
        direction = numpy.array(direction[:3], dtype=numpy.float64, copy=False)
        scale = numpy.dot(direction, normal)
        M[:3, :3] -= numpy.outer(normal, direction) / scale
        M[:3, 3] = direction * (numpy.dot(point, normal) / scale)
    else:
        # orthogonal projection
        M[:3, :3] -= numpy.outer(normal, normal)
        M[:3, 3] = numpy.dot(point, normal) * normal
    return M


def projection_from_matrix(matrix):
    """Return projection plane and perspective point from projection matrix.

    Return values are same as arguments for projection_matrix function:
    point, normal, direction, perspective, and pseudo.

    >>> P0 = projection_matrix((1, -2, 3), (-4, 5, 6))
    >>> result = projection_from_matrix(P0)
    >>> P1 = projection_matrix(*result)
    >>> is_same_transform(P0, P1)
    True
    >>> P0 = projection_matrix((1, -2, 3), (-4, 5, 6), (-7, -8, 9))
    >>> result = projection_from_matrix(P0)
    >>> P1 = projection_matrix(*result)
    >>> is_same_transform(P0, P1)
    True

    """
    M = numpy.array(matrix, dtype=numpy.float64, copy=False).T
    M33 = M[:3, :3]
    l, V = numpy.linalg.eig(M)
    i = numpy.where(abs(numpy.real(l) - 1.0) < 1e-9)[0]
    if len(i):
        # point: any eigenvector corresponding to eigenvalue 1
        point = numpy.real(V[:3, i[-1]]).squeeze()
        # direction: unit eigenvector corresponding to eigenvalue 0
        l, V = numpy.linalg.eig(M33)
        i = numpy.where(abs(numpy.real(l)) < 1e-9)[0][0]
        direction = numpy.real(V[:, i]).squeeze()
        direction /= norm(direction)
        # normal: unit eigenvector of M33.T corresponding to eigenvalue 0
        l, V = numpy.linalg.eig(M33.T)
        i = numpy.where(abs(numpy.real(l)) < 1e-9)[0]
        if len(i):
            # parallel projection
            normal = numpy.real(V[:, i[0]]).squeeze()
            normal /= norm(normal)
            return point, normal, direction, None, False
        else:
            # orthogonal projection, where normal equals direction vector
            return point, direction, None, None, False
    else:
        raise NotImplementedError("perspective projection not implemented.")


def shear_matrix(angle, direction, point, normal):
    """Return matrix to shear by angle along direction vector on shear plane.

    The shear plane is defined by a point and normal vector. The direction
    vector must be orthogonal to the plane's normal vector.

    A point P is transformed by the shear matrix into P" such that
    the vector P-P" is parallel to the direction vector and its extent is
    given by the angle of P-P'-P", where P' is the orthogonal projection
    of P onto the shear plane.

    >>> S = shear_matrix(1.234, (1, -2, 3), (-4, 5, 6), (0, 3, 2))
    >>> numpy.allclose(1.0, numpy.linalg.det(S))
    True

    """
    normal = numpy.array(normal[:3], dtype=numpy.float64, copy=True)
    normal /= norm(normal)
    direction = numpy.array(direction[:3], dtype=numpy.float64, copy=True)
    direction /= norm(direction)
    angle = math.tan(angle)
    M = numpy.identity(4, dtype=numpy.float64)
    M[:3, :3] += angle * numpy.outer(direction, normal)
    M[:3, 3] = -angle * numpy.dot(point, normal) * direction
    return M


def shear_from_matrix(matrix):
    """Return shear angle, direction and plane from shear matrix.

    >>> S0 = shear_matrix(1.234, (1, -2, 3), (-4, 5, 6), (0, 3, 2))
    >>> angle, direction, point, normal = shear_from_matrix(S0)
    >>> S1 = shear_matrix(angle, direction, point, normal)
    >>> is_same_transform(S0, S1)
    True

    """
    M = numpy.array(matrix, dtype=numpy.float64, copy=False)
    M33 = M[:3, :3]
    # normal: cross independent eigenvectors corresponding to the eigenvalue 1
    l, V = numpy.linalg.eig(M33)
    i = numpy.where(abs(numpy.real(l) - 1.0) < 1e-9)[0]
    V = numpy.real(V[:, i]).squeeze().T
    for i0, i1 in ((0, 1), (0, 2), (1, 2)):
        normal = numpy.cross(V[i0], V[i1])
        lenorm = norm(normal)
        if lenorm > 1e-9:
            break
    else:
        ValueError("No two linear independent eigenvectors found")
    normal /= lenorm
    # direction and angle
    direction = numpy.dot(M33 - numpy.identity(3), normal)
    angle = norm(direction)
    direction /= angle
    angle = math.atan(angle)
    # point: eigenvector corresponding to eigenvalue 1
    l, V = numpy.linalg.eig(M)
    i = numpy.where(abs(numpy.real(l) - 1.0) < 1e-9)[0][-1]
    point = numpy.real(V[:3, i]).squeeze()
    return angle, direction, point, normal


def decompose_matrix(matrix):
    """Return sequence of transformations from transformation matrix.

    matrix : array_like
        Non-degenerative homogeneous transformation matrix

    Return tuple of:
        scale : vector of scaling factors
        shear : list of shear factors for x-y, x-z, y-z axes
        rotate : list of Euler angles about static x, y, z axes
        translate : translation vector along x, y, z axes
        perspective : vector of perspectives along x, y, z, w axes

    Raise ValueError if matrix is of wrong type or degenerative.

    >>> T0 = translation_matrix((1, 2, 3))
    >>> scale, shear, angles, trans, persp = decompose_matrix(T0)
    >>> T1 = translation_matrix(trans)
    >>> numpy.allclose(T0, T1)
    True
    >>> S = scaling_matrix(0.123)
    >>> scale, shear, angles, trans, persp = decompose_matrix(S)
    >>> scale[0]
    0.123
    >>> R0 = euler_matrix(1, 2, 3)
    >>> scale, shear, angles, trans, persp = decompose_matrix(R0)
    >>> R1 = euler_matrix(*angles)
    >>> numpy.allclose(R0, R1)
    True

    """
    M = numpy.array(matrix, dtype=numpy.float64, copy=True).T
    if abs(M[3, 3]) < _EPS:
        raise ValueError("M[3, 3] is zero")
    M /= M[3, 3]
    P = M.copy()
    P[:, 3] = 0, 0, 0, 1
    if not numpy.linalg.det(P):
        raise ValueError("Matrix is singular")

    scale = numpy.zeros((3, ), dtype=numpy.float64)
    shear = [0, 0, 0]
    rotate = [0, 0, 0]

    if any(abs(M[:3, 3]) > _EPS):
        perspective = numpy.dot(M[:, 3], numpy.linalg.inv(P)) #.T #?
        M[:, 3] = 0, 0, 0, 1
    else:
        perspective = numpy.zeros((4, ), dtype=numpy.float64)

    translate = M[3, :3].copy()
    M[3, :3] = 0

    row = M[:3, :3].copy()
    scale[0] = norm(row[0])
    row[0] /= scale[0]
    shear[0] = numpy.dot(row[0], row[1])
    row[1] -= row[0] * shear[0]
    scale[1] = norm(row[1])
    row[1] /= scale[1]
    shear[0] /= scale[1]
    shear[1] = numpy.dot(row[0], row[2])
    row[2] -= row[0] * shear[1]
    shear[2] = numpy.dot(row[1], row[2])
    row[2] -= row[1] * shear[2]
    scale[2] = norm(row[2])
    row[2] /= scale[2]
    shear[1:] /= scale[2]

    if numpy.dot(row[0], numpy.cross(row[1], row[2])) < 0:
        scale *= -1
        row *= -1

    rotate[1] = math.asin(-row[0, 2])
    if math.cos(rotate[1]):
        rotate[0] = math.atan2(row[1, 2], row[2, 2])
        rotate[2] = math.atan2(row[0, 1], row[0, 0])
    else:
        #rotate[0] = math.atan2(row[1, 0], row[1, 1])
        rotate[0] = math.atan2(-row[2, 1], row[1, 1])
        rotate[2] = 0.0

    return scale, shear, rotate, translate, perspective


def orthogonalization_matrix(lengths=(10., 10., 10.), angles=(90., 90., 90.)):
    """Return orthogonalization matrix for crystallographic cell coordinates.

    >>> O = orthogonalization_matrix()
    >>> numpy.allclose(O[:3, :3], numpy.identity(3, float) * 10)
    True
    >>> O = orthogonalization_matrix([9.8, 12.0, 15.5], [87.2, 80.7, 69.7])
    >>> numpy.allclose(numpy.sum(O), 43.063229)
    True

    """
    lena, lenb, lenc = lengths
    angles = numpy.radians(angles)
    sina, sinb, sinc = numpy.sin(angles)
    cosa, cosb, cosg = numpy.cos(angles)
    co = (cosa * cosb - cosg) / (sina * sinb)
    return numpy.array((
        ( lena*sinb*math.sqrt(1.0-co*co),  0.0,  0.0, 0.0),
        (-lena*sinb*co,              lenb*sina,  0.0, 0.0),
        ( lena*cosb,                 lenb*cosa, lenc, 0.0),
        ( 0.0,                             0.0,  0.0, 1.0)),
        dtype=numpy.float64)


def superimpose_matrix(v0, v1, compute_rmsd=False):
    """Return matrix to transform given vector set into second vector set.

    v0 and v1 are shape (\*, 3) or (\*, 4) arrays of at least 3 vectors.
    Minimize weighted sum of squared deviations according to W. Kabsch.

    >>> v0 = numpy.random.rand(3, 4)
    >>> M = superimpose_matrix(v0, v0)
    >>> numpy.allclose(M, identity_matrix())
    True
    >>> v0 = numpy.random.rand(3, 4)
    >>> v0[:, 3] = 0
    >>> R = random_rotation_matrix(numpy.random.rand(3))
    >>> v1 = numpy.dot(v0, R.T)
    >>> M = superimpose_matrix(v0, v1)
    >>> numpy.allclose(v1, numpy.dot(v0, M.T))
    True

    """
    v0 = numpy.array(v0, dtype=numpy.float64, copy=False)[:, :3]
    v1 = numpy.array(v1, dtype=numpy.float64, copy=False)[:, :3]

    if v0.shape != v1.shape or v0.shape[0] < 3:
        raise ValueError("Vector sets are of wrong shape or type.")

    # move centroids to origin
    t0 = numpy.mean(v0, axis=0)
    t1 = numpy.mean(v1, axis=0)
    v0 = v0 - t0
    v1 = v1 - t1
    # Singular Value Decomposition of covariance matrix
    u, s, vh = numpy.linalg.svd(numpy.dot(v1.T, v0))
    # rotation matrix from SVD orthonormal bases
    R = numpy.dot(u, vh)
    if numpy.linalg.det(R) < 0.0:
        # R does not constitute right handed system
        R -= numpy.outer(u[:, 2], vh[2, :]*2.0)
        s[-1] *= -1.0
    # homogeneous transformation matrix
    M = numpy.identity(4, dtype=numpy.float64)
    T = numpy.identity(4, dtype=numpy.float64)
    M[:3, :3] = R
    T[:3, 3] = t1
    M = numpy.dot(T, M)
    T[:3, 3] = -t0
    M = numpy.dot(M, T)
    # compute root mean square error from SVD sigma
    if compute_rmsd:
        r = numpy.cumsum(v0*v0) + numpy.cumsum(v1*v1)
        rmsd = numpy.sqrt(abs(r - (2.0 * sum(s)) / len(v0)))
        return M, rmsd
    return M


def euler_matrix(ai, aj, ak, axes='sxyz'):
    """Return homogeneous rotation matrix from Euler angles and axis sequence.

    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple

    >>> R = euler_matrix(1, 2, 3, 'syxz')
    >>> numpy.allclose(numpy.sum(R[0]), -1.34786452)
    True
    >>> R = euler_matrix(1, 2, 3, (0, 1, 0, 1))
    >>> numpy.allclose(numpy.sum(R[0]), -0.383436184)
    True

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sh = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ch = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci*ch, ci*sh
    sc, ss = si*ch, si*sh

    M = numpy.identity(4, dtype=numpy.float64)
    if repetition:
        M[i, i] = cj
        M[i, j] = sj*si
        M[i, k] = sj*ci
        M[j, i] = sj*sh
        M[j, j] = -cj*ss+cc
        M[j, k] = -cj*cs-sc
        M[k, i] = -sj*ch
        M[k, j] = cj*sc+cs
        M[k, k] = cj*cc-ss
    else:
        M[i, i] = cj*ch
        M[i, j] = sj*sc-cs
        M[i, k] = sj*cc+ss
        M[j, i] = cj*sh
        M[j, j] = sj*ss+cc
        M[j, k] = sj*cs-sc
        M[k, i] = -sj
        M[k, j] = cj*si
        M[k, k] = cj*ci
    return M


def euler_from_matrix(matrix, axes='sxyz'):
    """Return Euler angles from rotation matrix for specified axis sequence.

    axes : One of 24 axis sequences as string or encoded tuple

    Note that many Euler angle triplets can describe one matrix.

    >>> R0 = euler_matrix(1, 2, 3, 'syxz')
    >>> al, be, ga = euler_from_matrix(R0, 'syxz')
    >>> R1 = euler_matrix(al, be, ga, 'syxz')
    >>> numpy.allclose(R0, R1)
    True

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    M = numpy.array(matrix, dtype=numpy.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        if sy > _EPS:
            ax = math.atan2( M[i, j],  M[i, k])
            ay = math.atan2( sy,       M[i, i])
            az = math.atan2( M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2( sy,       M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        if cy > _EPS:
            ax = math.atan2( M[k, j],  M[k, k])
            ay = math.atan2(-M[k, i],  cy)
            az = math.atan2( M[j, i],  M[i, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2(-M[k, i],  cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az


def euler_from_quaternion(quaternion, axes='sxyz'):
    """Return Euler angles from quaternion for specified axis sequence.

    >>> angles = euler_from_quaternion([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(angles, [0.123, 0, 0])
    True

    """
    return euler_from_matrix(quaternion_matrix(quaternion), axes)


def quaternion_from_euler(ai, aj, ak, axes='sxyz'):
    """Return quaternion from Euler angles and axis sequence.

    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple

    >>> q = quaternion_from_euler(1, 2, 3, 'ryxz')
    >>> numpy.allclose(q, [0.310622, -0.718287, 0.444435, 0.435953])
    True

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ti = ai*0.5
    tj = aj*0.5
    tk = ak*0.5
    ci = math.cos(ti)
    si = math.sin(ti)
    cj = math.cos(tj)
    sj = math.sin(tj)
    ck = math.cos(tk)
    sk = math.sin(tk)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    quaternion = numpy.empty((4, ), dtype=numpy.float64)
    if repetition:
        quaternion[i] = cj*(cs + sc)
        quaternion[j] = sj*(cc + ss)
        quaternion[k] = sj*(cs - sc)
        quaternion[3] = cj*(cc - ss)
    else:
        quaternion[i] = cj*sc - sj*cs
        quaternion[j] = cj*ss + sj*cc
        quaternion[k] = cj*cs - sj*sc
        quaternion[3] = cj*cc + sj*ss

    if parity:
        quaternion[j] *= -1
    return quaternion


def quaternion_about_axis(angle, axis=(1, 0, 0)):
    """Return quaternion for rotation about axis.

    >>> q = quaternion_about_axis(0.123, (1, 0, 0))
    >>> numpy.allclose(q, [0.06146124, 0, 0, 0.99810947])
    True

    """
    quaternion = numpy.zeros((4, ), dtype=numpy.float64)
    quaternion[:3] = axis[:3]
    qlen = norm(quaternion)
    if qlen:
        quaternion *= math.sin(angle/2.0) / qlen
    quaternion[3] = math.cos(angle/2.0)
    return quaternion


def _replace_by(module_function, warn=False):
    """Try replace decorated function by module.function."""

    def decorate(func, module_function=module_function, warn=warn):
        try:
            module, function = module_function.split('.')
            func, oldfunc = getattr(__import__(module), function), func
            #functools.update_wrapper(func, oldfunc) # doesn't work
            globals()['__old_' + func.__name__] = oldfunc
        except Exception, e:
            if warn:
                warnings.warn("Failed to import %s" % module_function)
        return func
    return decorate


@_replace_by('_vlfdlib.quaternion_to_matrix')
def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(R, rotation_matrix(0.123))
    True

    """
    q = numpy.array(quaternion, dtype=numpy.float64, copy=True)[:4]
    nq = numpy.dot(quaternion, quaternion)
    if not nq:
        return numpy.identity(4, dtype=numpy.float64)
    q *= math.sqrt(2.0 / nq)
    q = numpy.outer(q, q)
    return numpy.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=numpy.float64)


@_replace_by('_vlfdlib.quaternion_from_matrix')
def quaternion_from_matrix(matrix):
    """Return quaternion from rotation matrix.

    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.0164262, 0.0328524, 0.0492786, 0.9981095])
    True

    """
    q = numpy.empty((4, ), dtype=numpy.float64)
    M = numpy.array(matrix, dtype=numpy.float64, copy=False)[:4, :4]
    t = numpy.trace(M)
    if t > M[3, 3]:
        q[3] = t
        q[2] = M[1, 0] - M[0, 1]
        q[1] = M[0, 2] - M[2, 0]
        q[0] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
    q *= 0.5 / math.sqrt(t * M[3, 3])
    return q


@_replace_by('_vlfdlib.quaternion_multiply')
def quaternion_multiply(q1, q0):
    """Return multiplication of two quaternions.

    >>> q = quaternion_multiply([1, 2, 3, 4], [5, 6, 7, 8])
    >>> numpy.allclose(q, [24, 48, 48, -6])
    True

    """
    x0, y0, z0, w0 = q0
    x1, y1, z1, w1 = q1
    return numpy.array((
         x1*w0 + y1*z0 - z1*y0 + w1*x0,
        -x1*z0 + y1*w0 + z1*x0 + w1*y0,
         x1*y0 - y1*x0 + z1*w0 + w1*z0,
        -x1*x0 - y1*y0 - z1*z0 + w1*w0))


def quaternion_from_sphere_points(point0, point1):
    """Return quaternion from two points on unit sphere.

    point0 : E.g. sphere coordinates of cursor at mouse down
    point1 : E.g. current sphere coordinates of cursor

    >>> q = quaternion_from_sphere_points([0, 1, 0], [-0.998, 0.998, 0.0615])
    >>> numpy.allclose(q, [0.0615, 0.0, 0.998, 0.998])
    True

    """
    x, y, z = numpy.cross(point0, point1)
    return x, y, z, numpy.dot(point0, point1)


def quaternion_to_sphere_points(quaternion):
    """Return two points on unit sphere from quaternion.

    >>> p0, p1 = quaternion_to_sphere_points([0.0615, 0.0, 0.998, 0.998])
    >>> numpy.allclose((p0, p1), ([0, 1, 0], [-0.998, 0.998, 0.0615]))
    True

    """
    l = norm(quaternion[:2])
    p0 = numpy.array((-quaternion[1]/l, quaternion[0]/l, 0) if l else
                     (0, 1, 0), dtype=numpy.float64)
    p1 = numpy.array((quaternion[3]*p0[0] - quaternion[3]*p0[1],
                      quaternion[3]*p0[1] + quaternion[3]*p0[0],
                      quaternion[0]*p0[1] - quaternion[1]*p0[0]),
                      dtype=numpy.float64)
    if quaternion[3] < 0.0:
        p0 *= -1.0
    return p0, p1


def random_quaternion(rand):
    """Return uniform random unit quaternions from array of uniform deviates.

    rnd: array like
        Three independent random variables that are uniformly distributed
        between 0 and 1 for each returned quaternion.

    >>> q = random_quaternion(numpy.random.rand(3, 10))

    """
    rnd = numpy.array(rand, copy=False).reshape(3, -1)
    r1 = numpy.sqrt(1.0 - rand[0])
    r2 = numpy.sqrt(rand[0])
    t = math.pi * 2.0 * rand[1:3]
    c1, c2 = numpy.cos(t)
    s1, s2 = numpy.sin(t)
    return numpy.array((s1*r1, c1*r1, s2*r2, c2*r2)).T


def random_rotation_matrix(rnd):
    """Return uniform random rotation matrix from array of 3 uniform deviates.

    rnd: array like
        Three independent random variables that are uniformly distributed
        between 0 and 1 for each returned quaternion.

    >>> R = random_rotation_matrix(numpy.random.rand(3))

    """
    return quaternion_matrix(random_quaternion(rnd))


class Arcball(object):
    """Virtual Trackball Control.

    >>> ball = Arcball([320, 320], 320)
    >>> ball.click([500, 250])
    >>> R = ball.drag([475, 275])
    >>> numpy.allclose(numpy.sum(R), 3.90583455)
    True

    """

    def __init__(self, center=None, radius=1.0, initial=None):
        """Initializes virtual trackball control.

        center : Window coordinates of trackball center
        radius : Radius of trackball in window coordinates
        initial : Initial quaternion or rotation matrix

        """
        self.axis = None
        self.radius = radius
        self.center = numpy.zeros((3, ), dtype=numpy.float64)
        self.place(center, radius)
        self.v0 = numpy.array([0, 0, 1], dtype=numpy.float64)
        if initial is None:
            self.q0 = numpy.array([0, 0, 0, 1], dtype=numpy.float64)
        else:
            try:
                self.q0 = quaternion_from_matrix(initial)
            except Exception:
                self.q0 = initial
        self.qnow = self.q0

    def place(self, center=None, radius=1.0):
        """Place Arcball, e.g. when window size changes."""
        self.radius = float(radius)
        if center is None:
            self.center[:2] = 0, 0
        else:
            self.center[:2] = center[:2]

    def click(self, position, axis=None):
        """Set axis constraint and initial cursor window coordinates."""
        self.axis = axis
        self.q0 = self.qnow
        self.v0 = self._map_to_sphere(position, self.center, self.radius)

    def drag(self, position):
        """Return rotation matrix from updated cursor window coordinates."""
        v0 = self.v0
        v1 = self._map_to_sphere(position, self.center, self.radius)

        if self.axis is not None:
            v0 = self._constrain_to_axis(v0, self.axis)
            v1 = self._constrain_to_axis(v1, self.axis)

        t = numpy.cross(v0, v1)
        if numpy.dot(t, t) < _EPS:
            self.qnow = self.q0 # v0 and v1 coincide: no additional rotation
        else:
            q1 = [t[0], t[1], t[2], numpy.dot(v0, v1)]
            self.qnow = quaternion_multiply(q1, self.q0)

        return quaternion_matrix(self.qnow)

    def _map_to_sphere(self, position, center, radius):
        """Map window coordinates to unit sphere coordinates."""
        v = numpy.array([position[0], position[1], 0.0], dtype=numpy.float64)
        v -= center
        v /= radius
        v[1] *= -1
        l = numpy.dot(v, v)
        if l > 1.0:
            v /= math.sqrt(l) # position outside of sphere
        else:
            v[2] = math.sqrt(1.0 - l)
        return v

    def _constrain_to_axis(self, point, axis):
        """Return sphere point perpendicular to axis."""
        v = numpy.array(point, dtype=numpy.float64, copy=True)
        a = numpy.array(axis, dtype=numpy.float64, copy=True)
        a /= numpy.dot(a, v)
        v -= a # on plane
        n = numpy.dot(v, v)
        if n > 0.0:
            v /= math.sqrt(n)
            return v
        if a[2] == 1.0:
            return numpy.array([1, 0, 0], dtype=numpy.float64)
        v[:] = -a[1], a[0], 0
        v /= norm(v)
        return v

    def _nearest_axis(self, point, *axes):
        """Return axis, which arc is nearest to point."""
        nearest = None
        mx = -1.0
        for axis in axes:
            t = numpy.dot(self._constrain_to_axis(point, axis), point)
            if t > mx:
                nearest = axis
                mx = d
        return nearest


_NEXT_AXIS = [1, 2, 0, 1] # axis sequences for Euler angles

_AXES2TUPLE = { # axes string -> (inner axis, parity, repetition, frame)
    "sxyz": (0, 0, 0, 0), "sxyx": (0, 0, 1, 0), "sxzy": (0, 1, 0, 0),
    "sxzx": (0, 1, 1, 0), "syzx": (1, 0, 0, 0), "syzy": (1, 0, 1, 0),
    "syxz": (1, 1, 0, 0), "syxy": (1, 1, 1, 0), "szxy": (2, 0, 0, 0),
    "szxz": (2, 0, 1, 0), "szyx": (2, 1, 0, 0), "szyz": (2, 1, 1, 0),
    "rzyx": (0, 0, 0, 1), "rxyx": (0, 0, 1, 1), "ryzx": (0, 1, 0, 1),
    "rxzx": (0, 1, 1, 1), "rxzy": (1, 0, 0, 1), "ryzy": (1, 0, 1, 1),
    "rzxy": (1, 1, 0, 1), "ryxy": (1, 1, 1, 1), "ryxz": (2, 0, 0, 1),
    "rzxz": (2, 0, 1, 1), "rxyz": (2, 1, 0, 1), "rzyz": (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=False)

