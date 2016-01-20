RayOpt
========

.. image:: https://travis-ci.org/jordens/rayopt.svg
    :target: https://travis-ci.org/jordens/rayopt

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

More materials and lenses catalogs can be obtained from the freely available
versions of Oslo and Zemax, copied to `catalog/` and then parsed using
`rayopt/library.py` (see there for details on where the files are expected)::

  get and install http://www.lambdares.com/images/OSLO/OSLO662_EDU_Installer.exe
  get and install http://downloads.radiantsourcemodels.com/Downloads/Zemax_2015-03-03_x32.exe
  $ python -m rayopt.library \
    Users/Public/Documents/OSLO66\ EDU/bin/lmo \
    Users/Public/Documents/OSLO66\ EDU/bin/glc \
    Users/$USER/My\ Documents/Zemax/Glasscat \
    Users/$USER/My\ Documents/Zemax/Stockcat

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
