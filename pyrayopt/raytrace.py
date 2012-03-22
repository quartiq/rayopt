#!/usr/bi0/python
# -*- coding: utf8 -*-

"""
Raytracing like Spencer and Murty 1962, J Opt Soc Am 52, 6
with some improvements
"""

import itertools
import cPickle as pickle

from enthought.traits.api import (HasTraits, List, Float, Array, Dict,
        Bool, Str, Instance, Trait, cached_property, Property, Callable,
        Tuple, Enum)

from numpy import (array, sqrt, ones_like, float64, dot, sign, zeros,
        linalg, where, nan, nan_to_num, finfo, inf, mgrid, ones,
        concatenate, linspace, putmask, zeros_like, extract)

from numpy.ma import masked_invalid

from scipy.optimize import (newton, fsolve)

from qo.theory.transformations import (euler_matrix, translation_matrix)

def sfloat(a):
    try: return float(a)
    except: return None

def dotprod(a,b):
    return (a*b).sum(-1)

def dir_to_angles(x,y,z):
    r = array([x,y,z], dtype=float64)
    return r/linalg.norm(r)

lambda_f = 486.1e-9
lambda_d = 589.3e-9
lambda_c = 656.3e-9

class Material(HasTraits):
    name = Str
    comment = Str
    glasscode = Float
    nd = Float
    vd = Float
    density = Float
    alpham3070 = Float
    alpha20300 = Float
    chemical = Tuple
    thermal = Tuple
    price = Float
    transmission = Dict
    sellmeier = Array(dtype=float64, shape=(None, 2))

    def __str__(self):
        return self.name

    def refractive_index(self, wavelength):
        w2 = (wavelength/1e-6)**2
        c0 = self.sellmeier[:,0]
        c1 = self.sellmeier[:,1]
        n2 = 1.+(c0*w2/(w2-c1)).sum(-1)
        return sqrt(n2)

    def _nd_default(self):
        return self.refractive_index(lambda_d)

    def dispersion(self, wavelength_short, wavelength_mid,
            wavelength_long):
        return (self.refractive_index(wavelength_mid)-1)/(
                self.refractive_index(wavelength_short)-
                self.refractive_index(wavelength_long))

    def delta_n(self, wavelength_short, wavelength_long):
        return (self.refractive_index(wavelength_short)-
                self.refractive_index(wavelength_long))

    def _vd_default(self):
        return (self.nd-1)/(
                self.refractive_index(lambda_f)-
                self.refractive_index(lambda_c))

    def dn_thermal(self, t, n, wavelength):
        d0, d1, d2, e0, e1, tref, lref = self.thermal
        dt = t-tref
        w = wavelength/1e-6
        dn = (n**2-1)/(2*n)*(d0*dt+d1*dt**2+d2*dt**3+
                (e0*dt+e1*dt**2)/(w**2-lref**2))
        return dn


class FictionalMaterial(Material):
    def refractive_index(self, wavelength):
        return ones_like(wavelength)*self.nd

    def dispersion(self, wavelength_short, wavelength_mid,
            wavelength_long):
        return ones_like(wavelength_mid)*self.vd

    def delta_n(self, wavelength_short, wavelength_long):
        return (self.nd-1)/self.vd*ones_like(wavelength_short)


class GlassCatalog(HasTraits):
    db = Dict(Str, Instance(Material))
    name = Str

    def __getitem__(self, name):
        return self.db[name]

    def import_zemax(self, fil):
        dat = open(fil)
        for line in dat:
            try:
                cmd, args = line.split(" ", 1)
                if cmd == "CC":
                    self.name = args
                elif cmd == "NM":
                    args = args.split()
                    g = Material(name=args[0], glasscode=float(args[2]),
                            nd=float(args[3]), vd=float(args[4]))
                    self.db[g.name] = g
                elif cmd == "GC":
                    g.comment = args
                elif cmd == "ED":
                    args = map(float, args.split())
                    g.alpham3070, g.alpha20300, g.density = args[0:3]
                elif cmd == "CD":
                    s = array(map(float, args.split())).reshape((-1,2))
                    g.sellmeier = array([si for si in s if not si[0] == 0])
                elif cmd == "TD":
                    s = map(float, args.split())
                    g.thermal = s
                elif cmd == "OD":
                    g.chemical = map(float, s[1:])
                    g.price = s[0]=="-" and None or float(s[0])
                elif cmd == "LD":
                    s = map(float, args.split())
                    pass
                elif cmd == "IT":
                    s = map(float, args.split())
                    g.transmission[(s[0], s[2])] = s[1]
                else:
                    print cmd, args, "not handled"
            except Exception, e:
                print cmd, args, "failed parsing", e
        self.db[g.name] = g

    @classmethod
    def cached_or_import(cls, fil):
        filpick = fil + ".pickle"
        try:
            c = pickle.load(open(filpick))
        except IOError:
            c = cls()
            c.import_zemax(fil)
            pickle.dump(c, open(filpick, "wb"), protocol=2)
        return c

# http://refractiveindex.info
vacuum = FictionalMaterial(name="vacuum", nd=1., vd=inf)

air = Material(name="air", sellmeier=[
    [5792105E-8, 238.0185],
    [167917E-8, 57.362],
    ], vd=inf)
def air_refractive_index(wavelength):
        w2 = (wavelength/1e-6)**-2
        c0 = air.sellmeier[:,0]
        c1 = air.sellmeier[:,1]
        n  = 1.+(c0/(c1-w2)).sum(-1)
        return n
air.refractive_index = air_refractive_index


class Rays(HasTraits):
    # wavelength for all rays
    wavelength = Float(lambda_d)
    # refractive index we are in
    refractive_index = Float(1.)
    # start positions
    positions = Array(dtype=float64, shape=(None, 3))
    # angles
    angles = Array(dtype=float64, shape=(None, 3))
    # geometric length of the rays
    lengths = Array(dtype=float64, shape=(None,))
    # end positions
    end_positions = Property
    # total optical path lengths to start (including previous paths)
    optical_path_lengths = Array(dtype=float64, shape=(None,))

    def transform(self, t):
        n = len(self.positions)
        p = self.positions.T.copy()
        a = self.angles.T.copy()
        p.resize((4, n))
        a.resize((4, n))
        p[3,:] = 1
        p = dot(t, p)
        a = dot(t, a)
        p.resize((3, n))
        a.resize((3, n))
        return Rays(positions=p.T, angles=a.T)

    def _get_end_positions(self):
        return self.positions + (self.lengths*self.angles.T).T


class ParaxialTrace(HasTraits):
    wavelength = Float
    wavelength_short = Float
    wavelength_long = Float
    refractive_indices = Array(dtype=float64, shape=(None,))
    dispersions = Array(dtype=float64, shape=(None,))

    # marginal/axial,
    # principal/chief
    heights = Array(dtype=float64, shape=(None,2))
    angles = Array(dtype=float64, shape=(None,2))
    incidence = Array(dtype=float64, shape=(None,2))

    aberration3 = Array(dtype=float64, shape=(None,7))
    aberration5 = Array(dtype=float64, shape=(None,7))

    lagrange = Property
    focal_length = Property
    back_focal_length = Property
    front_focal_length = Property
    image_height = Property
    entrance_pupil_height = Property
    entrance_pupil_position = Property
    exit_pupil_height = Property
    exit_pupil_position = Property
    front_f_number = Property
    back_f_number = Property
    back_numerical_aperture = Property
    front_numerical_aperture = Property
    front_airy_radius = Property
    back_airy_radius = Property
    magnification = Property
    angular_magnification = Property

    def __init__(self, length=None, **k):
        super(ParaxialTrace, self).__init__(**k)
        if length is not None:
            self.refractive_indices = zeros((length,), dtype=float64)
            self.heights = zeros((length,2), dtype=float64)
            self.angles = zeros((length,2), dtype=float64)
            self.incidence = zeros((length,2), dtype=float64)
            self.dispersions = zeros((length,), dtype=float64)
            self.aberration3 = zeros((length,7), dtype=float64)
            self.aberration5 = zeros((length,7), dtype=float64)

    def _get_lagrange(self):
        return self.refractive_indices[0]*(
                self.angles[0,0]*self.heights[0,1]-
                self.angles[0,1]*self.heights[0,0])

    def _get_focal_length(self):
        return -self.lagrange/self.refractive_indices[0]/(
                self.angles[0,0]*self.angles[-2,1]-
                self.angles[0,1]*self.angles[-2,0])

    def _get_front_focal_length(self):
        return -self.heights[1,0]/self.angles[0,0]

    def _get_back_focal_length(self):
        return -self.heights[-2,0]/self.angles[-2,0]

    def _get_image_height(self):
        return self.lagrange/(self.refractive_indices[-2]*
                self.angles[-2,0])
        
    def _get_back_numerical_aperture(self):
        return abs(self.refractive_indices[-2]*self.angles[-2,0])

    def _get_front_numerical_aperture(self):
        return abs(self.refractive_indices[0]*self.angles[0,0])

    def _get_entrance_pupil_position(self):
        return -self.heights[1,1]/self.angles[1,1]

    def _get_exit_pupil_position(self):
        return -self.heights[-2,1]/self.angles[-2,1]

    def _get_entrance_pupil_height(self):
        return self.heights[1,0]+\
                self.entrance_pupil_position*self.angles[0,0]

    def _get_exit_pupil_height(self):
        return self.heights[-2,0]+\
                self.entrance_pupil_position*self.angles[-2,0]

    def _get_front_f_number(self):
        #return self.focal_length/(2*self.entrance_pupil_height)
        return self.refractive_indices[-2]/(
                2*self.front_numerical_aperture)

    def _get_back_f_number(self):
        #return self.focal_length/(2*self.exit_pupil_height)
        return self.refractive_indices[0]/(
                2*self.back_numerical_aperture)

    def _get_back_airy_radius(self):
        return 1.22*self.wavelength/(2*self.back_numerical_aperture)

    def _get_front_airy_radius(self):
        return 1.22*self.wavelength/(2*self.front_numerical_aperture)

    def _get_magnification(self):
        return (self.refractive_indices[0]*self.angles[0,0])/(
                self.refractive_indices[-2]*self.angles[-2,0])

    def _get_angular_magnification(self):
        return (self.refractive_indices[-2]*self.angles[-2,1])/(
                self.refractive_indices[0]*self.angles[0,1])


class Element(HasTraits):
    typestr = "E"
    origin = Array(dtype=float64, shape=(3,))
    angles = Array(dtype=float64, shape=(3,))
    transform = Property(depends_on="origin, angles")
    inverse_transform = Property(depends_on="transform")
    material = Trait(None, Material)
    radius = Float

    #@cached_property
    def _get_transform(self):
        r = euler_matrix(axes="rxyz", *self.angles)
        t = translation_matrix(-self.origin)
        return dot(r,t)

    #@cached_property
    def _get_inverse_transform(self):
        return linalg.inv(self.transform)

    def transform_to(self, rays):
        return rays.transform(self.transform)

    def transform_from(self, rays):
        return rays.transform(self.inverse_transform)

    def intercept(self, positions, angles):
	# ray length to intersection with element
        # only reference plane, overridden in subclasses
	# solution for z=0
        s = -positions[...,2]/angles[...,2] # nan_to_num()
        return where(s>=0, s, nan)

    def propagate(self, in_rays):
        out_rays = self.transform_to(in_rays)
        # length up to surface
        in_rays.lengths = self.intercept(
                out_rays.positions, out_rays.angles)
        out_rays.optical_path_lengths = in_rays.optical_path_lengths+\
                in_rays.lengths*in_rays.refractive_index
        # new transverse position
        out_rays.positions = out_rays.positions + \
                (in_rays.lengths*out_rays.angles.T).T
        out_rays.wavelength = in_rays.wavelength
        if self.material is None:
            out_rays.refractive_index = in_rays.refractive_index
        else:
            out_rays.refractive_index = self.material.refractive_index(
                    out_rays.wavelength)
            m = in_rays.refractive_index/out_rays.refractive_index
            out_rays.angles = self.refract(
                    out_rays.positions, out_rays.angles, m)
        return in_rays, out_rays
   
    def propagate_paraxial(self, index, rays):
        rays.heights[index] = rays.heights[index-1]+\
                self.origin[2]*rays.angles[index-1]
        rays.angles[index] = rays.angles[index-1]
        rays.refractive_indices[index] = \
                rays.refractive_indices[index-1]
        rays.dispersions[index] = rays.dispersions[index-1]
        rays.incidence[index] = 1-rays.incidence[index-1]

    def aberration3(self, index, rays):
        rays.aberration3[index] = 0

    def aberration5(self, index, rays):
        rays.aberration5[index] = 0

    def revert(self):
        pass


class Interface(Element):
    typestr = "F"

    def shape_func(self, p):
        raise NotImplementedError

    def shape_func_deriv(self, p):
        raise NotImplementedError

    def intercept(self, p, a):
        s = zeros_like(p[:,0])
        for i in range(p.shape[0]):
            try:
                s[i] = newton(func=lambda s: self.shape_func(p[i]+s*a[i]),
                    fprime=lambda s: dot(self.shape_func_deriv(p[i]+s*a[i]),
                        a[i]), x0=-p[i,2]/a[i,2], tol=1e-7, maxiter=15)
            except RuntimeError:
                s[i] = nan
        return where(s>=0, s, nan)

    def refract(self, f, a, m):
	# General Ray-Tracing Procedure
        # G. H. SPENCER and M. V. R. K. MURTY
        # JOSA, Vol. 52, Issue 6, pp. 672-676 (1962)
	# doi:10.1364/JOSA.52.000672
	# sign(m) for reflection
        fp = self.shape_func_deriv(f)
        fp2 = dotprod(fp, fp)
        o = m*dotprod(a, fp)/fp2
	if m**2 == 1:
	    g = -2*o
	else:
            p = (m**2-1)/fp2
            g = sign(m)*sqrt(o**2-p)-o
        r = m*a+(g*fp.T).T
	#print "rfr", self, f, a, g, r
	return r

    def revert(self):
        raise NotImplementedError


class Spheroid(Interface):
    typestr = "S"
    curvature = Float(0)
    conic = Float(1) # assert self.radius**2 < 1/(self.conic*self.curvature**2)
    aspherics = Array(dtype=float64)

    def shape_func(self, p):
        x, y, z = p.T
        r2 = x**2+y**2
        j = range(len(self.aspherics))
        o = dotprod(self.aspherics,
                array([r2**(i+2) for i in j]).T)
        return z-self.curvature*r2/(1+
                    sqrt(1-self.conic*self.curvature**2*r2))-o

    def shape_func_deriv(self, p):
        x, y, z = p.T
        r2 = x**2+y**2
        j = range(len(self.aspherics))
        o = dotprod(2*self.aspherics,
                nan_to_num(array([(i+2)*r2**(i+1) for i in j])).T)
        e = self.curvature/sqrt(1-self.conic*self.curvature**2*r2)+o
        return array([-x*e, -y*e, ones_like(e)]).T

    def intercept(self, p, a):
        if len(self.aspherics) == 0:
            # replace the newton-raphson with the analytic solution
            c = self.curvature
            if c == 0:
                return Element.intercept(self, p, a)
            else:
                k = array([1,1,self.conic])
                d = c*dotprod(a,k*p)-a[...,2]
                e = c*dotprod(a,k*a)
                f = c*dotprod(p,k*p)-2*p[...,2]
                s = (-sqrt(d**2-e*f)-d)/e
        else:
            return Interface.intercept(self, p, a)
        return where(s*sign(self.origin[2])>=0, s, nan)

    def propagate_paraxial(self, index, rays):
        c = self.curvature
        if len(self.aspherics) > 0:
            c += 2*self.aspherics[0]
        u0 = rays.angles[index-1]
        y0 = rays.heights[index-1]
        t = self.origin[2]
        y = y0+t*u0
        rays.heights[index] = y
        n0 = rays.refractive_indices[index-1]
        n = self.material.refractive_index(rays.wavelength)
        rays.refractive_indices[index] = n
        mu = n0/n
        u = mu*u0+c*(mu-1)*y
        rays.angles[index] = u
    
    def aberration3(self, index, rays):
        c = self.curvature
        u0 = rays.angles[index-1]
        u = rays.angles[index]
        y = rays.heights[index]
        n0 = rays.refractive_indices[index-1]
        n = rays.refractive_indices[index]
        mu = n0/n
        i = c*y+u0
        rays.incidence[index] = i
        l = n*(u[0]*y[1]-u[1]*y[0])
        s = .5*n0*(1-mu)*y*(u+i)/l
        tsc = s[0]*i[0]**2
        cc = s[0]*i[0]*i[1]
        tac = s[0]*i[1]**2
        tpc = (1-mu)*c*l/n0/2
        dc = s[1]*i[0]*i[1]+.5*(u[1]**2-u0[1]**2)
        dn0 = rays.dispersions[index-1]
        dn = self.material.delta_n(rays.wavelength_short,
             rays.wavelength_long)
        rays.dispersions[index] = dn
        tachc, tchc = -y[0]*i/l*(dn0-mu*dn)

        if len(self.aspherics) > 0:
           k = (4*self.aspherics[0]+(self.conic-1)*c**3/2)*(n-n0)/l
           tsc += k*y[0]**4
           cc += k*y[0]**3*y[1]
           tac += k*y[0]**2*y[1]**2
           dc += k*y[0]*y[1]**3
        rays.aberration3[index] = [
                tsc, cc, tac, tpc, dc, tachc, tchc]

    def revert(self):
        self.curvature *= -1
        self.aspherics *= -1


class Object(Element):
    typestr = "O"
    radius = Float(inf)
    field_angle = Float(.1)
    material = Trait(air, Material)
    apodization = Enum(("constant", "gaussian", "cos3"))

    def rays_to_height(self, xy, height):
        if self.radius == inf:
            p = array([(xy[0],xy[1],zeros_like(xy[0]))])
            a = array([(height[0]*self.field_angle,
                        height[1]*self.field_angle,
                        sqrt(1-(height[0]*self.field_angle)**2
                              -(height[1]*self.field_angle)**2))])
        else:
            p = array([(height[0]*self.radius,
                        height[1]*self.radius,
                        zeros_like(height[0]))])
            a = array([(xy[0], xy[1], sqrt(1-xy[0]**2-xy[1]**2))])
        return p, a

    def rays_for_point(self, height, chief, marg, num):
        chief_x, chief_y = chief
        marg_px, marg_nx, marg_py, marg_ny = marg
        mmarg_x, mmarg_y = marg_px+marg_nx, marg_py+marg_ny
        dmarg_x, dmarg_y = marg_px-marg_nx, marg_py-marg_ny

        x, y = mgrid[marg_nx:marg_px:num*1j, marg_ny:marg_py:num*1j]
        x, y = x.flatten(), y.flatten()
        r2 = (((x-mmarg_x)/dmarg_x)**2+((y-mmarg_y)/dmarg_y)**2)<.25
        x, y = extract(r2, x), extract(r2, y)
        x = concatenate(([chief_x], linspace(marg_nx, marg_px, num),
            ones((num,))*chief_x, x))
        y = concatenate(([chief_y], ones((num,))*chief_y,
            linspace(marg_nx, marg_px, num), y))
        p, a = self.rays_to_height((x,y),
                (height[0]*ones_like(x),height[1]*ones_like(y)))
        return p[0].T, a[0].T


class Aperture(Element):
    typestr = "A"
    radius = Float

    def propagate(self, in_rays, stop=False):
        in_rays, out_rays = super(Aperture, self).propagate(in_rays)
        if stop:
            r = (out_rays.positions[...,(0,1)]**2).sum(axis=-1)
            putmask(out_rays.positions[...,2], r>self.radius**2, nan)
        return in_rays, out_rays


class Image(Element):
    typestr = "I"
    radius = Float


class System(HasTraits):
    name = Str
    wavelengths = Array(dtype=float64, shape=(None,))
    heights = Array(dtype=float64, shape=(None, 2))
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
            dia = (self.object.radius == inf and
                self.object.field_angle or self.object.radius)
            s += "%-2s %1s %-12s %-12s %10.5g %15s %5.2f %5.2f\n" % (
                "", self.object.typestr, "", "", dia,
                self.object.material,
                self.object.material.nd, self.object.material.vd)
        for i,e in enumerate(self.elements):
            curv = getattr(e, "curvature", 0)
            roc = curv == 0 and inf or 1/curv
            mat = getattr(e, "material", None)
            n = getattr(mat, "nd", nan)
            v = getattr(mat, "vd", nan)
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
            d = [array(de(self, p, r)).reshape((-1,))*de.weight for de in demerits]
            return concatenate(d)

        x0 = array([p.get_value(self) for p in parameters])
        # bs = 2
        # bounds = [(min(p/bs, p*bs), max(p/bs, p*bs)) for p in x0]
        #from numpy.random import randn
        #x0 *= 1+randn(len(x0))*.1

        eqs = [c for c in constraints if c.equality]
        ineqs = [c for c in constraints if not c.equality]

        def equality_constraints(x):
            return concatenate([c(self) for c in eqs])
        def inequality_constraints(x):
            return concatenate([c(self) for c in ineqs])

        from openopt import NLP
        problem = NLP(objective_function, x0,
                c=ineqs and inequality_constraints or None,
                h=eqs and equality_constraints or None,
                lb=array([p.bounds[0] for p in parameters]),
                ub=array([p.bounds[1] for p in parameters]),
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
        

class Demerit(HasTraits):
    name = Str
    func = Callable
    weight = Float(1)

    def __call__(self, system, ptrace, rays):
        return self.func(system, ptrace, rays)

demerit_rms_position = Demerit(name="rms size",
    func=lambda system, ptrace, rays:
    [masked_invalid(r.positions[...,(0,1)]).std(axis=0) for r in rays])

demerit_rms_angle = Demerit(name="rms angle",
    func=lambda system, ptrace, rays:
    [masked_invalid(r.angles[...,(0,1)]).std(axis=0) for r in rays])

demerit_mean_angle = Demerit(name="mean angle",
    func=lambda system, ptrace, rays:
    [masked_invalid(r.angles[...,(0,1)]).mean(axis=0) for r in rays])

demerit_aberration3 = Demerit(name="primary aberrations",
    func=lambda system, ptrace, rays:
    ptrace.aberration3.sum(0)*ptrace.image_height)


class Parameter(HasTraits):
    name = Str
    bounds = Tuple((-inf, inf))
    scale = Float

    def __init__(self, name, bounds=None, scale=1, **k):
        super(Parameter, self).__init__(name=name,
                bounds=bounds, scale=scale, **k)

    def set_value(self, system, value):
        #exec "system.%s=%s" % (self.name, value)
        setattr(system, self.name, value)

    def get_value(self, system):
        #return eval("system.%s" % self.name)
        return getattr(system, self.name)


class Constraint(HasTraits):
    equality = Bool(True)

    def __call__(self, system):
        pass


class MaterialThickness(Constraint):
    minimum = Float(1e-3)
    maximum = Float(10e-2)
    equality = False
    
    def __call__(self, system):
        r = []
        for i,e in enumerate(system.elements[:-1]):
            en = system.elements[i+1]
            if isinstance(e, Aperture):
                continue
            if isinstance(en, Aperture):
                en = system.elements[i+2]
            if e.material not in (air, vacuum):
                center = en.origin[2]
                edge = (center+
                          e.shape_func(array([(0, e.radius, 0)]))-
                          en.shape_func(array([(0, en.radius, 0)])))
                r.append(self.minimum-min(center, edge))
                r.append(max(center, edge)-self.maximum)
                print i, center, edge
        return array(r)

# ------------------------------------------------------------------------

#catpath = "/home/rj/.wine/drive_c/Program Files/ZEMAX/Glasscat/"
catpath = "/home/rjordens/work/nist/glass/"
schott = GlassCatalog.cached_or_import(catpath+"SCHOTT.AGF")
#schott = GlassCatalog.cached_or_import(catpath+"schott-17-03-2009.agf")
ohara = GlassCatalog.cached_or_import(catpath+"ohara.agf")
misc = GlassCatalog.cached_or_import(catpath+"MISC.AGF")
infrared = GlassCatalog.cached_or_import(catpath+"INFRARED.AGF")

slide_projector = System(elements=[
    Spheroid(curvature=1/.55622, origin=(0,0,0),
        radius=.033, material=schott["N-SK4"]),
    Spheroid(curvature=-1/.15868, origin=(0,0,.00632),
        radius=.033),
    Spheroid(curvature=1/.039820, origin=(0,0,.00061),
        radius=.028, material=schott["N-SK4"]),
    Spheroid(curvature=1/.089438, origin=(0,0,.01464),
        radius=.023),
    Spheroid(curvature=-1/.078577, origin=(0,0,.01008),
        radius=.018, material=schott["N-SF5"]),
    Spheroid(curvature=1/.032370, origin=(0,0,.00254),
        radius=.016),
    Aperture(origin=(0,0,.01632), radius=.013),
    Spheroid(curvature=1/.11910, origin=(0,0,.00056),
        radius=.015, material=schott["N-SK4"]),
    Spheroid(curvature=-1/.042139, origin=(0,0,.00442),
        radius=.015),
    Spheroid(origin=(0,0,.0735), radius=.02),
    ])

tessar = System(elements=[
    Spheroid(curvature=1/.022585, origin=(0,0,.004),
        radius=.017/2, material=schott["N-LAF21"]),
    Spheroid(curvature=1/3.175, origin=(0,0,.0035),
        radius=.017/2),
    Spheroid(curvature=-1/.03977, origin=(0,0,.004006),
        radius=.013/2, material=schott["N-SF15"]),
    Spheroid(curvature=1/.020748, origin=(0,0,.0015),
        radius=.012/2),
    Aperture(origin=(0,0,.002), radius=.0116/2),
    Spheroid(curvature=-1/.50296, origin=(0,0,.004061),
        radius=.015/2, material=schott["F5"]),
    Spheroid(curvature=1/.047474, origin=(0,0,.0015),
        radius=.015/2, material=schott["N-LAF21"]),
    Spheroid(curvature=-1/.028859, origin=(0,0,.0035),
        radius=.015/2),
    Spheroid(origin=(0,0,.042584), radius=.043634/2),
    ])

# Surface Radius   Thickness Material Diameter
lithograph_table = """
0       0.0000   0.1615            0.145
1       -0.79639 0.0418   Silica   0.189
2       -0.31776 0.0013            0.197
3       0.31768  0.0247   Silica   0.198
4       -0.90519 0.0014            0.198
5       -1.53140 0.0138   Silica   0.196
6       0.86996  0.1068            0.199
7       -0.96937 0.0138   Silica   0.171
8       0.30419  0.0567            0.169
9       -4.54831 0.0179   Silica   0.176
10      0.36328  0.0559            0.180
11      -0.13317 0.0179   Silica   0.185
12      0.83902  0.0148            0.240
13      0.0000   0.0581   Silica   0.273
14      -0.24698 0.0014            0.273
15      0.97362  0.0653   Silica   0.330
16      -0.34739 0.0176            0.330
17      1.12092  0.0447   Silica   0.342
18      -0.71734 0.0009            0.342
19      0.82256  0.0323   Silica   0.332
20      -1.18244 0.0009            0.332
21      0.31346  0.0345   Silica   0.300
22      1.42431  0.1071            0.296
23      -2.67307 0.0138   Silica   0.187
24      0.15590  0.0569            0.162
25      -0.21770 0.0171   Silica   0.154
26      1.01511  0.0839            0.154
27      -0.11720 0.0176   Silica   0.161
28      3.98510  0.0171            0.183 Stop
29      -0.40426 0.0339   Silica   0.193
30      -0.21818 0.0019            0.214
31      -7.96203 0.0650   Silica   0.246
32      -0.23180 0.0014            0.265
33      1.01215  0.0411   Silica   0.289
34      -0.50372 0.0008            0.289
35      0.40042  0.0441   Silica   0.287
36      3.75600  0.0014            0.281
37      0.24527  0.0429   Silica   0.267
38      0.49356  0.0013            0.267
39      0.17401  0.1101   Silica   0.233
40      0.11185  0.0832            0.139
41      0.07661  0.0263   Silica   0.082
42      0.18719  0.0227            0.065
43      0.0000   0.0000            0.036
"""

def system_from_table(data, scale):
    s = System(scale=scale)
    pos = 0.
    for line in data.splitlines():
        p = line.split()
        if not p:
            continue
        roc = float(p[1])
        if roc == 0:
            curv = 0
        else:
            curv = 1/roc
        if p[-1]=="Stop":
            del p[-1]
            s.elements.append(Aperture(
                origin=(0,0,0),
                radius=rad))
        rad = float(p[-1])/2
        if "Silica" in p:
            mat = misc["SILICA"]
        else:
            mat = air
        e = Spheroid(
            curvature=curv,
            origin=(0,0,pos),
            radius=rad,
            material=mat)
        s.elements.append(e)
        if s.object is None:
            s.object = Object(radius=rad, origin=(0,0,0))
        pos = float(p[2])
    return s

lithograph = system_from_table(lithograph_table, scale=.0254)

def system_from_oslo(fil):
    s = System()
    th = 0.
    for line in fil.readlines():
        p = line.split()
        if not p:
            continue
        cmd, args = p[0], p[1:]

        if cmd == "LEN":
            s.name = " ".join(args[1:-2]).strip("\"")
        elif cmd == "UNI":
            s.scale = float(args[0])*1e-3
            e = Spheroid()
            e.origin = (0,0,0)
        elif cmd == "AIR":
            e.material = air
        elif cmd == "TH":
            th = float(args[0])
            if th > 1e2:
                th = 0
        elif cmd == "AP":
            e.radius = float(args[0])
        elif cmd == "GLA":
            e.material = {"SILICA": misc["SILICA"],
                          "SFL56": schott["SFL56"],
                          "SF6": schott["SF6"],
                          "CAF2": misc["CAF2"],
                          "O_S-BSM81": ohara["S-BSM81"],}[args[0]]
        elif cmd == "AST":
            s.elements.append(Aperture(radius=e.radius, origin=(0,0,0)))
        elif cmd == "RD":
            e.curvature = 1/(float(args[0]))
        elif cmd in ("NXT", "END"):
            s.elements.append(e)
            e = Spheroid()
            e.origin = (0,0,th)
        elif cmd in ("//", "DES", "EBR", "GIH", "DLRS", "WW", "WV"):
            pass
        else:
            print cmd, "not handled", args
            continue
        #assert len(s.elements) - 1 == int(args[0])
    return s

lithium = system_from_oslo(open("examples/lithium_objective.len"))
del lithium.elements[0]
lithium.elements[4:4] = [Aperture(origin=(0,0,1e-9), radius=10)]
lithium.elements[-1].material = vacuum
lithium.elements[-2].material = vacuum
lithium.wavelengths=(670e-9, 780e-9, 1064e-9)
lithium.heights=((0, 0), (0, .707), (0, 1))
lithium.object = Object(origin=(0,0,inf), field_angle=.0028)
lithium.image = Image(origin=(0,0,4))
del lithium.elements[0]
del lithium.elements[-1]
#print str(lithium)
#lithium.paraxial_trace()

def system_from_zemax(fil):
    s = System(object=Object(), image=Image())
    next_pos = 0.
    a = None
    for line in fil.readlines():
        line = line.strip().split(" ", 1)
        cmd, args = line[0], line[1:]
        if args:
            args = args[0]
        if not cmd:
            continue

        if cmd in ("VERS", "MODE", "NOTE"):
            pass
        elif cmd == "UNIT":
            if args.split()[0] == "MM":
                s.scale = 1e-3
        elif cmd == "NAME":
            s.name = args.strip("\"")
        elif cmd == "SURF":
            e = Spheroid(origin=(0,0,next_pos))
            s.elements.append(e)
        elif cmd in ("TYPE", "HIDE", "MIRR", "SLAB", "POPS"):
            pass
        elif cmd == "CURV":
            e.curvature = float(args.split()[0])
        elif cmd == "DISZ":
            next_pos = float(args)
        elif cmd == "GLAS":
            args = args.split()
            name = args[0]
            for db in schott, ohara, infrared, misc:
                if name in db.db:
                    e.material = db[name]
            if not e.material:
                e.material = {
                        "SILICA": misc["SILICA"],
                        "CAF2": misc["CAF2"],
                        "O_S-BSM81": ohara["S-BSM81"],}[name]
        elif cmd == "COMM":
            pass
        elif cmd == "DIAM":
            args = args.split()
            e.radius = float(args[0])/2
            if a is not None:
                a.radius = e.radius
            a = None
        elif cmd == "STOP":
            a = Aperture(radius=e.radius, origin=(0,0,0))
            s.elements.append(a)
        elif cmd == "WAVN":
            s.wavelengths = [float(i)*1e-6 for i in args.split() if
                    float(i) > 0 and float(i) != .55]
        elif cmd == "XFLN":
            ohx = [float(i) for j,i in enumerate(args.split()) if
                    float(i) > 0 or j == 0]
        elif cmd == "YFLN":
            ohy = [float(i) for j,i in enumerate(args.split()) if
                    float(i) > 0 or j == 0]
        elif cmd == "ENPD":
            s.object.radius = float(args)/2
            s.object.origin = (0,0,0)
        elif cmd in ("GCAT", "OPDX", "TOL", "MNUM", "MOFF", "FTYP",
                     "SDMA", "RAIM", "GFAC", "PUSH", "PICB", "ROPD",
                     "PWAV", "POLS", "GLRS", "BLNK", "COFN", "NSCD",
                     "GSTD", "CONF", "DMFS", "ISNA", "VDSZ", "PUPD", "ENVD",
                     "ZVDX", "ZVDY", "ZVCX", "ZVCY", "ZVAN",
                     "VDXN", "VDYN", "VCXN", "VCYN", "VANN",
                     "FWGT", "FWGN", "WWGT", "WWGN",
                     "WAVL", "WAVM", "XFLD", "YFLD",
                     "MNCA", "MNEA", "MNCG", "MNEG", "MXCA", "MXCG",
                     "EFFL", "RGLA"):
            pass
        else:
            print cmd, "not handled", args
            continue
        #assert len(s.elements) - 1 == int(args[0])
    s.heights = [(x,y) for x in ohx for y in ohy]
    return s

lithium2 = system_from_zemax(open("examples/lithium_objective.zmx"))
del lithium2.elements[:2]
lithium2.object.origin=(0,0,inf)
lithium2.image.origin[2]=lithium2.elements[-1].origin[2]
del lithium2.elements[-1]

k_z_imaging = System(
    name="potassium z-imaging",
    wavelengths=(767e-9,), # 532e-9, 1064e-9),
    heights=((0,0), (0,.707), (0,1)),
    object=Object(radius=.15, material=vacuum),
    elements=[
    Spheroid(curvature=0, origin=(0,0,9), angles=(.07,0,0),
        radius=5, material=misc["SILICA"]),
    Spheroid(curvature=0, origin=(0,0,4),
        radius=8, material=air),
    Spheroid(curvature=0, origin=(0,0,2), angles=(-.07,0,0),
        radius=6, material=ohara["S-LAH64"]),
    Spheroid(
        #curvature=1/-13.30184, conic=1-.49026,
        #aspherics=(5.6889e-6, 3.4996e-8, 2.7467e-10, -1.7292e-12),
        #curvature=1/-13.33018, conic=5.09043e-01,
        #aspherics=(5.71627e-06, 3.54087e-08, 2.76486e-10, 0.),
        curvature=-7.52695472e-02, conic=5.12297710e-01,
        aspherics=(5.63076589e-06, 3.44957698e-08,   2.76486389e-10, 0),
        origin=(0,0,6), radius=6, material=air),
    Aperture(origin=(0,0,18), radius=5),
    Spheroid(curvature=1/238.73, origin=(0,0,200), # 01-LAO277
        radius=20, material=schott["SSK4A"]),
    Spheroid(curvature=1/-156.53, origin=(0,0,4),
        radius=20, material=schott["SF8"]),
    Spheroid(curvature=1/-942.12, origin=(0,0,3),
        radius=20, material=air),
    Spheroid(curvature=1/128.59, origin=(0,0,1), # 01-LAM277
        radius=18, material=schott["SF8"]),
    Spheroid(curvature=1/197.93, origin=(0,0,7),
        radius=18, material=air),
    ], image=Image(origin=(0,0,196.8)))
#print str(k_z_imaging)
#k_z_imaging.paraxial_trace()

k_z_objective = System(
    name="potassium z-objective",
    wavelengths=(767e-9,), # 532e-9, 1064e-9),
    heights=((0,0), (0,.707), (0,1)),
    object=Object(radius=inf, field_angle=.01, material=air),
    elements=[
    Aperture(origin=(0,0,0), radius=5),
    Spheroid(
        #curvature=1/-13.30184, conic=1-.49026,
        #aspherics=(5.6889e-6, 3.4996e-8, 2.7467e-10, -1.7292e-12),
        #curvature=1/-13.33018, conic=5.09043e-01,
        #aspherics=(5.71627e-06, 3.54087e-08, 2.76486e-10, 0.),
        #curvature=7.52695472e-02, conic=5.12297710e-01,
        #curvature=0.07518237, conic=0.36771952,
        #curvature=0.079356, conic=0.5009,
        #aspherics=(-5.63076589e-06, +3.44957698e-08, -2.76486389e-10, 0),
        #aspherics=(-1e-5,1e-8,-1e-10,1e-12),
        #aspherics=(0,0,0,0),
        curvature=7.66026881e-02, conic=5.01588906e-01,
        aspherics=(-1.41159165e-06,  -1.49867039e-07,   3.82005287e-09,
                    -4.55278941e-11),
        origin=(0,0,10), radius=6, material=ohara["S-LAH64"]),
    Spheroid(curvature=2.36235627e-03, origin=(0,0,6),
        radius=6, material=air),
    Spheroid(curvature=0, origin=(0,0,2), #angles=(.05,0,0),
        radius=6, material=misc["SILICA"]),
    Spheroid(curvature=0, origin=(0,0,4),
        radius=5, material=vacuum),
    ], image=Image(origin=(0,0,9), #angles=(-.05,0,0)
        )
    )
#print str(k_z_objective)
#print -k_z_objective.elements[1].shape_func(array(((0,0,0,0,0,0,0),
#    linspace(0,6,7),(0,0,0,0,0,0,0))).T)
#k_z_objective.paraxial_trace()

simple_lens = System(
    name="simple biconvex",
    wavelengths=(lambda_d, lambda_c, lambda_f),
    heights=((0,0), (0,.707), (0,1)),
    object=Object(radius=20, material=vacuum),
    elements=[
    Spheroid(curvature=1/50., origin=(0,0,200),
        radius=50, material=FictionalMaterial(nd=1.5, vd=62.5)),
    Spheroid(curvature=-1/50., origin=(0,0,15),
        radius=50, material=vacuum),
    ],
    image=Image(origin=(0,0,65.517)))
#simple_lens.paraxial_trace()

simple_photo = System(
    name="photo triplet, f/2.7, f=100 U.S.-Pat 2,453,260 (1948-Pestrecov)",
    wavelengths=(lambda_d, lambda_c, lambda_f),
    heights=((0,0), (0,.707), (0,1)),
    object=Object(radius=inf, material=air, field_angle=.25),
    elements=[
    Spheroid(curvature=1/40.94, origin=(0,0,0),
        radius=20, material=FictionalMaterial(nd=1.617, vd=55.)),
    Spheroid(curvature=0, origin=(0,0,8.74),
        radius=20, material=air),
    Spheroid(curvature=1/-55.65, origin=(0,0,11.05),
        radius=20, material=FictionalMaterial(nd=1.649, vd=33.8)),
    Aperture(radius=30, origin=(0,0,2.78)),
    Spheroid(curvature=1/39.75, origin=(0,0,0),
        radius=20, material=air),
    Spheroid(curvature=1/107.56, origin=(0,0,7.63),
        radius=20, material=FictionalMaterial(nd=1.617, vd=55.)),
    Spheroid(curvature=1/-43.33, origin=(0,0,9.54),
        radius=20, material=air),
    ],
    image=Image(origin=(0,0,79.34)))
#print str(simple_photo)
#simple_photo.paraxial_trace()

double_gauss = System(
    name="4 glass double gauss, intermediate optical design",
    wavelengths=(lambda_d, lambda_c, lambda_f),
    heights=((0,0), (0,.707), (0,1)),
    object=Object(radius=inf, material=air, field_angle=.6248),
    elements=[
    Spheroid(curvature=1/25.907, origin=(0,0,0),
        radius=10.6, material=schott["SK4"]),
    Spheroid(curvature=1/147.340, origin=(0,0,5.083),
        radius=8.98, material=air),
    Spheroid(curvature=1/34.8044, origin=(0,0,2.355),
        radius=6.05, material=schott["F4"]),
    Spheroid(curvature=1/17.34, origin=(0,0,1.694),
        radius=5.25, material=air),
    Aperture(radius=5.147, origin=(0,0,2.542)),
    Spheroid(curvature=1/-17.34, origin=(0,0,2.542),
        radius=5.2, material=schott["F4"]),
    Spheroid(curvature=1/-34.8, origin=(0,0,1.694),
        radius=6., material=air),
    Spheroid(curvature=1/-138.87, origin=(0,0,2.355),
        radius=8.6, material=schott["SK4"]),
    Spheroid(curvature=1/-24.396, origin=(0,0,5.083),
        radius=10.24, material=air),
    ],
    image=Image(origin=(0,0,88.7558)))
#print str(double_gauss)
#double_gauss.revert()
#print str(double_gauss)
#double_gauss.paraxial_trace()

f=5.
t=2e-3
vacuum_mirror = FictionalMaterial(nd=-1., vd=inf)
schwarzschild = System(
    name="Schwarzschild",
    wavelengths=(313e-9,),
    heights=((0,0), ),# (0,.707), (0,1)),
    object=Object(radius=.1, material=vacuum),
    elements=[
    Aperture(radius=2.5, origin=(0,0,4.5)),
    Spheroid(curvature=-1/22.25, origin=(0,0,23.279-t),
        radius=13.5, material=vacuum_mirror),
    Spheroid(curvature=-1/6.674, origin=(0,0,-15.278+t),
        radius=2.5, material=vacuum),
    ],
    image=Image(origin=(0,0,182.438-t+14.5)))
    #object=Object(radius=inf, material=vacuum),
    #elements=[
    #Spheroid(curvature=1/(5**.5-1)/f, origin=(0,0,4*f),
    #    radius=3, material=vacuum),
    #Spheroid(curvature=1/(5**.5+1)/f, origin=(0,0,-2*f),
    #    radius=20, material=vacuum_mirror),
    #Aperture(radius=5, origin=(0,0,(5**.5+1)*f)),
    #    ],
    #image=Image(origin=(0,0,f)))
#     object=Object(radius=.1, material=vacuum),
#     elements=[
#     Aperture(radius=10, origin=(0,0,f+.096)),
#     Spheroid(curvature=-1/(5**.5+1)/f, origin=(0,0,(5**.5+1)*f),
#         radius=20, material=vacuum_mirror),
#     Spheroid(curvature=-1/(5**.5-1)/f, origin=(0,0,-2*f),
#         radius=3, material=vacuum),
#         ],
#     image=Image(origin=(0,0,250)))

#schwarzschild.paraxial_trace()

