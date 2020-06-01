RayOpt
========

.. image:: https://travis-ci.org/quartiq/rayopt.svg
    :target: https://travis-ci.org/quartiq/rayopt

.. image:: https://ci.appveyor.com/api/projects/status/6e97f8o94v7r5bpb/branch/master?svg=true
    :target: https://ci.appveyor.com/project/jordens/rayopt

.. image:: https://codecov.io/github/jordens/rayopt/coverage.svg?branch=master
    :target: https://codecov.io/github/jordens/rayopt?branch=master

.. image:: https://anaconda.org/jordens/rayopt/badges/installer/conda.svg
    :target: https://anaconda.org/jordens/rayopt

.. image:: https://img.shields.io/pypi/l/rayopt.svg
    :target: https://pypi.python.org/pypi/rayopt


Introduction
------------

Optics design (lenses, cavities, gaussian optics, lasers).
Geometric, paraxial, and gaussian ray tracing.


Installation
------------

Install like any usual Python package using `pip`, `easy_install`, or plain
`setup.py`. Anaconda packages from three operating systems and three current
Python versions are available through `Anaconda
<https://anaconda.org/jordens/rayopt>`_. Install with::

  conda install -c https://conda.anaconda.org/jordens/channel/ci rayopt

The distribution already contains all materials from http://refractiveindex.info/.

More materials
--------------

More materials and lenses catalogs can be obtained from the freely available
versions of Oslo and Zemax, copied to `catalog/` and then parsed using
`rayopt/library.py`.

Zemax
.....

More materials and lenses catalogs can be obtained from the freely available
versions of Oslo and Zemax, copied to `catalog/` and then parsed using
`rayopt/library.py` (see there for details on where the files are expected)

Get `Zemax optics studio <https://my.zemax.com/en-US/OpticStudio-downloads/>`_.
You can either install the software or unpack it with
`innoextract <https://constexpr.org/innoextract/>`_. Depending on your chosen
method the paths have to be adapted: ::

    $ python -m rayopt.library \
    Users/$USER/My\ Documents/Zemax/Glasscat \
    Users/$USER/My\ Documents/Zemax/Stockcat


OSLO
.....

For OSLO, download and install OSLO.::

    get and install http://www.lambdares.com/images/OSLO/OSLO662_EDU_Installer.exe
    $ python -m rayopt.library \
    Users/Public/Documents/OSLO66\ EDU/bin/lmo \
    Users/Public/Documents/OSLO66\ EDU/bin/glc


Examples
--------

Usage examples are at in their `own repository
<https://github.com/jordens/rayopt-notebooks>`_ as `IPython
Notebooks
<http://nbviewer.ipython.org/github/jordens/rayopt-notebooks/tree/master/>`_,
specifically also the `Tutorial
<http://nbviewer.ipython.org/github/jordens/rayopt-notebooks/blob/master/tutorial.ipynb>`_.

Notes
-----

Distance
........

The choice of prescription specification is a little different from most other
lens design and ray tracing programs. RayOpt associates with an element
(surface):

* `distance` (or directional `offset`, measured in the global, unrotated coordinate
  system) of the element's apex relative to the previous element's apex.
* orientation (x-y-z Euler `angles` in the rotating frame) with respect to
  the directional offset
* element properties (type of element, `curvature`, aspheric and conic coefficients,
  focal length of an ideal element)
* optionally, the `material` after the element (behind the surface)

Ray data are given at (ray intercepts) or just after (direction cosines,
paraxial slopes) the respective element unless stated otherwise (e.g. incidence
angles).

The choice of associating the "distance to" and not the "thickness after"
with a surface has several advantages: shifts, offsets, tolerances can be implemented
in a more straight forward manner, ray propagation becomes more natural and
efficient (transfer, intercept, refraction), ray data at the surfaces' apex planes does
not need to be tracked. The "thickness after" does not have much meaning in
ray trace data as it can only be used later when tracing toward the next element and its
direction is typically ill defined. Compared to most other programs the
distance data is the thickness data shifted by one element towards the object.

Object and Image
................

Object and image are located at the first (index 0) and last (index -1)
surface respectively. This naturally allows tracking their positions,
material and shape data and supports curved objects and images naturally.
Further data like pupils data are maintained in the two
`Conjugate` s of the `System`.

Therefore, a minimal system of a single lens consists of fours surfaces: object,
the two lens surfaces (one of which can be the aperture stop) and the image
surface. The `offset` data of the first (object) surface does play a role in
ray tracing but it can be useful as it locates the global coordinate system's
origin. The `material` of the last (image) surface is used as it can cause
incident rays to be evanescent at the image surface. This can also be compared
to other programs where the thickness of the image surface is never relevant or
the material in object space and the position of the lens has to be tracked
somewhere else depending on the implementation.

Literature
----------

* Warren J. Smith: Modern Optical Engineering, McGraw-Hill 2000: concise
  and methods derivation from paraxial all the way to arbitrary ray tracing,
  with terminology explained and examples given
* Michael Bass (ed): Mandbook of Optics I and II, OSA/McGraw-Hill 1995:
  physical foundations, broad on optics, comprehensive on theory, some info on
  numerics, some example designs
* Daniel Malacara: Handbook of Optical Design, Marcel Dekker Inc. 1994:
  Introduction, Aberations, Examples, more info on terminology, especially in
  ray tracing programs and codes
* Daniel Malacara: Geometrical and Instrumental Optics, Academic Press Inc. 1988:
  less info about algorithms and numerical methds, more examples and use cases,
  speciality lens designs
* Robert R. Shannon: The Art and Science of Optical Design, Cambridge
  University Press 1997: many examples with Oslo and Zemax, not very thorough
  on numerical methods and foundations, good for material comparison with own
  codes.
* Michael J. Kidger: Intermediate Optical Design, SPIE Press 2004:
  info on optimization techniques and algorithms, higher order aberrations,
  lots of example designs
* Milton Laikin: Lens Design, CRC Press 2007: little bit of basic theory, lots
  of basic and paradigmatic example designs
* Oslo Optics `manual <https://www.lambdares.com/wp-content/uploads/support/oslo/oslo_edu/oslo-user-guide.pdf>`_ and `reference <https://www.lambdares.com/wp-content/uploads/support/oslo/oslo_edu/oslo-optics-reference.pdf>`_
* Zemax
  `manual <https://neurophysics.ucsd.edu/Manuals/Zemax/ZemaxManual.pdf>`_
