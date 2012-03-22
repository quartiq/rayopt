# -*- coding: utf8 -*-
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

import numpy as np
from traits.api import (HasTraits, Str, Array, Float, Instance, List)

from scipy.optimize import (newton, fsolve)

from .elements import Element, Object, Image
from .raytrace import ParaxialTrace, Rays

class System(HasTraits):
    name = Str
    wavelengths = Array(dtype=np.float64, shape=(None,))
    heights = Array(dtype=np.float64, shape=(None, 2))
    temperature = Float(21.)
    scale = Float(1e-3)
    object = Instance(Object)
    elements = List(Element)
    image = Instance(Image)

    def revert(self):
        m = self.object.material
        self.object.material = self.elements[-1].material
        for e in self.elements:
            if hasattr(e, "material"):
                m, e.material = e.material, m
        d = self.image.origin
        self.image.origin = self.elements[0].origin
        self.elements.reverse()
        for e in self.elements:
            e.revert()
            d, e.origin = e.origin, d

    def __add__(self, other):
        self.elements += other.elements
        return self

    def __str__(self):
        s = ""
        s += "System: %s\n" % self.name
        s += "Scale: %g m\n" % self.scale
        s += "Temperature: %g C\n" % self.temperature
        s += "Wavelengths: %s nm\n" % ",".join("%.0f" % (w/1e-9)
                    for w in self.wavelengths)
        s += "Surfaces:\n"
        s += "%2s %1s %12s %12s %10s %15s %5s %5s\n" % (
                "#", "T", "Distance to", "ROC", "Diameter", 
                "Material after", "N", "V")
        if self.object:
            dia = (self.object.radius == np.inf and
                self.object.field_angle or self.object.radius)
            s += "%-2s %1s %-12s %-12s %10.5g %15s %5.2f %5.2f\n" % (
                "", self.object.typestr, "", "", dia,
                self.object.material,
                self.object.material.nd, self.object.material.vd)
        for i,e in enumerate(self.elements):
            curv = getattr(e, "curvature", 0)
            roc = curv == 0 and np.inf or 1/curv
            mat = getattr(e, "material", None)
            n = getattr(mat, "nd", np.nan)
            v = getattr(mat, "vd", np.nan)
            s += "%-2i %1s %12.7g %12.6g %10.5g %15s %5.2f %5.2f\n" % (
                i, e.typestr, e.origin[2], roc, e.radius*2, mat, n, v)
        if self.image:
            s += "%2s %1s %12.7g %-12s %10.5g %15s %-5s %-5s\n" % (
                "", self.image.typestr, self.image.origin[2], "",
                self.image.radius*2, "", "", "")
        return s

    def paraxial_trace(self):
        p = ParaxialTrace(length=len(self.elements)+2)
        p.wavelength = self.wavelengths[0]
        p.wavelength_long = max(self.wavelengths)
        p.wavelength_short = min(self.wavelengths)
        p.refractive_indices[0] = self.object.material.refractive_index(
                p.wavelength)
        #p.heights[0], p.angles[0] = (18.5, -6.3), (0, .25) # photo
        #p.heights[0], p.angles[0] = (6.25, -7.102), (0, .6248) # dbl gauss
        #p.heights[0], p.angles[0] = (0, -.15), (0.25, -.0004) # k_z_i
        #p.heights[0], p.angles[0] = (5, 0), (0, .01) # k_z_o
        #p.heights[0], p.angles[0] = (self.object.radius, 0), (0, .5) # schwarzschild
        p.heights[0] = (self.object.radius, 0)
        p.angles[0] = (0, .5)
        #print "h at aperture:", self.height_at_aperture_paraxial(p)
        self.propagate_paraxial(p)
        #print "heights:", p.heights
        #print "angles:", p.angles
        #print "incidences:", p.incidence
        # p.aberration3 *= -2*p.image_height*p.angles[-1,0] # seidel
        # p.aberration3 *= -p.image_height/p.angles[-1,0] # longit
        # p.aberration3 *= p.image_height # transverse
        s = ""
        s += "%2s %1s% 10s% 10s% 10s% 10s% 10s% 10s% 10s\n" % (
                "#", "T", "TSC", "CC", "TAC", "TPC", "DC", "TAchC", "TchC")
        for i in range(1,len(p.aberration3)-1):
            ab = p.aberration3[i]
            s += "%-2s %1s% 10.4g% 10.4g% 10.4g% 10.4g% 10.4g% 10.4g% 10.4g\n" % (
                    i-1, self.elements[i-1].typestr,
                    ab[0], ab[1], ab[2], ab[3], ab[4], ab[5], ab[6])
        ab = p.aberration3.sum(0)
        s += "%-2s %1s% 10.4g% 10.4g% 10.4g% 10.4g% 10.4g% 10.4g% 10.4g\n" % (
              " ∑", "", ab[0], ab[1], ab[2], ab[3], ab[4], ab[5], ab[6])
        print s
        print "focal length:", p.focal_length
        print "front numerical aperture:", p.front_numerical_aperture
        print "back numerical aperture:", p.back_numerical_aperture
        print "front focal length:", p.front_focal_length
        print "back focal length:", p.back_focal_length
        print "image height:", p.image_height
        print "entrance pupil position:", p.entrance_pupil_position
        print "exit pupil position:", p.exit_pupil_position
        print "entrance pupil height:", p.entrance_pupil_height
        print "exit pupil height:", p.exit_pupil_height
        print "front f number:", p.front_f_number
        print "back f number:", p.back_f_number
        print "front airy radius:", p.front_airy_radius
        print "back airy radius:", p.back_airy_radius
        print "magnification:", p.magnification
        print "angular magnification:", p.angular_magnification
        return p

    def propagate_paraxial(self, rays):
        for i,e in enumerate(self.elements):
            e.propagate_paraxial(i+1, rays)
            e.aberration3(i+1, rays)
        self.image.propagate_paraxial(i+2, rays)
    
    def height_at_aperture_paraxial(self, rays):
        for i,e in enumerate(self.elements):
            e.propagate_paraxial(i+1, rays)
            if isinstance(e, Aperture):
                return rays.heights[i+1]

    def propagate(self, rays):
        for a, b in zip([self.object] + self.elements,
		        self.elements + [self.image]):
            a_rays, rays = b.propagate(rays)
            yield a, a_rays
	yield b, rays

    def propagate_through(self, rays):
        for element, rays in self.propagate(rays):
            pass
        return rays

    def height_at_aperture(self, rays):
        for element, in_rays in self.propagate(rays):
            if isinstance(element, Aperture):
                return in_rays.end_positions[...,(0,1)]/element.radius

    def chief_and_marginal(self, height, rays,
            paraxial_chief=True,
            paraxial_marginal=True):
        assert sum(1 for e in self.elements
		if isinstance(e, Aperture)) == 1
       
        def stop_for_pos(x,y):
	    # returns relative aperture height given object angles and
	    # relative object height
            rays.positions, rays.angles = self.object.rays_to_height(
                    (x,y), height)
            return self.height_at_aperture(rays)[0]

        d = 1e-3 # arbitrary to get newton started, TODO: better scale

        if paraxial_chief:
            d0 = stop_for_pos(0,0)
            chief = -d*d0/(stop_for_pos(d,d)-d0)
        else:
            chief = fsolve(lambda p: stop_for_pos(*p),
                    (0,0), xtol=1e-2, epsfcn=d)

        if paraxial_marginal:
            dmarg = d/(stop_for_pos(*(chief+d))-stop_for_pos(*chief))
            marg_px, marg_py = chief+dmarg
            marg_nx, marg_ny = chief-dmarg
        else:
            marg_px = newton(lambda x: stop_for_pos(x, chief[1])[0]-1,
                    chief[0]+d)
            marg_nx = newton(lambda x: stop_for_pos(x, chief[1])[0]+1,
                    chief[0]-d)
            marg_py = newton(lambda y: stop_for_pos(chief[0], y)[1]-1,
                    chief[1]+d)
            marg_ny = newton(lambda y: stop_for_pos(chief[0], y)[1]+1,
                    chief[1]-d)

        return chief, (marg_px, marg_nx, marg_py, marg_ny)

    def get_ray_bundle(self, wavelength, height, number, **kw):
        rays = Rays(wavelength=wavelength, height=height)
        c, m = self.chief_and_marginal(height, rays, **kw)
	print c, m
        p, a = self.object.rays_for_point(height, c, m, number)
        rays.positions = p
        rays.angles = a
        return rays

    def solve(self):
        pass

    def optimize(self, rays, parameters, demerits, constraints=(),
            method="ralg"):

        def objective_function(x):
            for i,p in enumerate(parameters):
                p.set_value(self, x[i])
            p = self.paraxial_trace()
            r = [self.propagate_through(ir) for ir in rays]
            d = [np.array(de(self, p, r)).reshape((-1,))*de.weight for de in demerits]
            return np.concatenate(d)

        x0 = np.array([p.get_value(self) for p in parameters])
        # bs = 2
        # bounds = [(min(p/bs, p*bs), max(p/bs, p*bs)) for p in x0]
        #from numpy.random import randn
        #x0 *= 1+randn(len(x0))*.1

        eqs = [c for c in constraints if c.equality]
        ineqs = [c for c in constraints if not c.equality]

        def equality_constraints(x):
            return np.concatenate([c(self) for c in eqs])
        def inequality_constraints(x):
            return np.concatenate([c(self) for c in ineqs])

        from openopt import NLP
        problem = NLP(objective_function, x0,
                c=ineqs and inequality_constraints or None,
                h=eqs and equality_constraints or None,
                lb=np.array([p.bounds[0] for p in parameters]),
                ub=np.array([p.bounds[1] for p in parameters]),
                #scale=[p.scale for p in parameters],
                diffInt=[p.scale*1e-2 for p in parameters],
                ftol=1e-10, gtol=1e-10, xtol=1e-14,
                maxCPUTime=2e3, maxNonSuccess=30,
                maxFunEvals=2000, iprint=1, plot=1)
        res = problem.solve(method)
        print res
        x, f = res.xf, res.ff
        for i,p in enumerate(parameters):
             p.set_value(self, x[i])
        return x0,x,f
        


