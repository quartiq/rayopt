# -*- coding: utf8 -*-
#
#   rayopt - raytracing for optical imaging systems
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

from __future__ import print_function, absolute_import, division

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec

from . import GeometricTrace, GaussianTrace
from .utils import tanarcsin
from .special_sums import polar_sum


class CenteredFormatter(mpl.ticker.ScalarFormatter):
    """Acts exactly like the default Scalar Formatter, but yields an empty
    label for ticks at "center"."""
    center = 0.

    def __call__(self, value, pos=None):
        if value == self.center:
            return ""
        return super(CenteredFormatter, self).__call__(value, pos)


class Analysis(object):
    figwidth = 12.
    update = False
    resize = False
    align = False
    close = None
    print = True
    update_conjugates = True
    refocus_paraxial = True
    trace_gaussian = False
    print_gaussian = False
    print_system = True
    resize_full = False
    refocus_full = True
    print_full = False
    plot_paraxial = False
    plot_gaussian = False
    plot_full = False
    plot_rays = 3
    plot_transverse = True
    plot_spots = True
    defocus = 5
    plot_opds = True
    plot_longitudinal = True

    def __init__(self, system, run=True, **kwargs):
        self.system = system
        self.text = []
        self.figures = []
        for k, v in kwargs.items():
            assert hasattr(self, k)
            setattr(self, k, v)
        if run:
            self.run()
        if self.print:
            for t in self.text:
                print(t)

    def run(self):
        if self.update:
            self.system.update()
        if self.close is not None:
            self.system.close(self.close)
        if self.align:
            self.system.paraxial.align()
        if self.refocus_paraxial:
            self.system.paraxial.refocus()
        if self.update_conjugates:
            self.system.paraxial.update_conjugates()
        if self.resize:
            self.system.paraxial.resize()
            self.system.fix_sizes()
        if self.trace_gaussian and self.system.object.finite:
            self.gaussian = GaussianTrace(self.system)
        if self.print_gaussian:
            self.text.append(str(self.gaussian))
        if self.resize_full:
            t = GeometricTrace(self.system)
            t.rays_paraxial()
            t.resize()
            self.system.fix_sizes()
        if self.refocus_full:
            t = GeometricTrace(self.system)
            t.rays_point((0, 0.), nrays=13, distribution="radau",
                         clip=False, filter=False)
            t.refocus()
        if self.print_system:
            self.text.append(str(self.system))
        t = GeometricTrace(self.system)
        t.rays_paraxial()
        if self.print_full:
            self.text.append(str(t))
        fig, ax = plt.subplots(figsize=(self.figwidth, self.figwidth))
        self.figures.append(fig)
        self.system.plot(ax)
        if self.plot_paraxial:
            self.system.paraxial.plot(ax)
        if self.plot_gaussian:
            self.gaussian.plot(ax)
        if self.plot_full:
            t.plot(ax)
        for h in min(self.system.fields), max(self.system.fields):
            t = GeometricTrace(self.system)
            t.rays_clipping((0, h))
            t.plot(ax)

        if self.plot_transverse is True:
            self.plot_transverse = self.system.fields
        if self.plot_transverse:
            figheight = self.figwidth*len(self.plot_transverse)/5
            fig = plt.figure(figsize=(self.figwidth, figheight))
            self.figures.append(fig)
            self.transverse(fig, self.plot_transverse)

        if self.plot_longitudinal:
            fig, ax = plt.subplots(1, 5,
                                   figsize=(self.figwidth, self.figwidth/5))
            self.figures.append(fig)
            self.longitudinal(ax, max(self.plot_transverse))

        if self.plot_spots is True:
            self.plot_spots = self.system.fields
        if self.plot_spots:
            figheight = self.figwidth*len(self.plot_spots)/self.defocus
            fig, ax = plt.subplots(len(self.plot_spots), self.defocus,
                                   figsize=(self.figwidth, figheight),
                                   sharex=True, sharey=True)
            self.figures.append(fig)
            self.spots(ax[::-1], self.plot_spots)

        if self.plot_opds is True:
            self.plot_opds = self.system.fields
        if self.plot_opds:
            figheight = self.figwidth*len(self.plot_opds)/4
            fig, ax = plt.subplots(len(self.plot_opds), 4,
                                   figsize=(self.figwidth, figheight))
            # , sharex=True, sharey=True)
            self.figures.append(fig)
            self.opds(ax[::-1], self.plot_opds)

        return self.text, self.figures

    @staticmethod
    def setup_axes(ax, xlabel=None, ylabel=None, title=None,
                   xzero=True, yzero=True):
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        if yzero:
            ax.spines["left"].set_position("zero")
            ax.yaxis.set_major_formatter(CenteredFormatter())
        if xzero:
            ax.spines["bottom"].set_position("zero")
            ax.xaxis.set_major_formatter(CenteredFormatter())
        ax.tick_params(bottom=True, top=False, left=True, right=False,
                       labeltop=False, labelright=False,
                       labelleft=True, labelbottom=True,
                       direction="out", axis="both")
        ax.xaxis.set_smart_bounds(True)
        ax.yaxis.set_smart_bounds(True)
        ax.locator_params(tight=True, nbins=5)
        kw = dict(rotation="horizontal")
        if xlabel:
            ax.set_xlabel(xlabel, horizontalalignment="right",
                          verticalalignment="bottom", **kw)
        if ylabel:
            ax.set_ylabel(ylabel, horizontalalignment="left",
                          verticalalignment="top", **kw)
        if title:
            ax.set_title(title)

    @staticmethod
    def post_setup_axes(ax):
        ax.relim()
        # ax.autoscale_view(True, True, True)
        xl, xu = ax.get_xlim()
        yl, yu = ax.get_ylim()
        if ax.spines["left"].get_position() == "zero":
            xl = 0
        if ax.spines["bottom"].get_position() == "zero":
            yl = 0
        ax.xaxis.set_label_coords(xu, yl, transform=ax.transData)
        ax.yaxis.set_label_coords(xl, yu, transform=ax.transData)

    @classmethod
    def pre_setup_fanplot(cls, fig, n):
        gs = gridspec.GridSpec(n, 4)
        axpx, axe, axpy = None, None, None
        ax = []
        for i in range(n):
            axm = fig.add_subplot(gs.new_subplotspec((i, 0), 1, 2),
                                  sharex=axpy, sharey=axe)
            axpy = axpy or axm
            axe = axe or axm
            axsm = fig.add_subplot(gs.new_subplotspec((i, 2), 1, 1),
                                   sharex=axpx, sharey=axe)
            axpx = axpx or axsm
            axss = fig.add_subplot(gs.new_subplotspec((i, 3), 1, 1),
                                   sharex=axpx, sharey=axe)
            ax.append((axm, axsm, axss))
            for axi, xl, yl in [
                    (axm, "PY", "EY"),
                    (axsm, "PX", "EY"),
                    (axss, "PX", "EX"),
                    ]:
                cls.setup_axes(axi, xl, yl)
        return ax[::-1]

    @classmethod
    def pre_setup_xyplot(cls, axi, **kwargs):
        cls.setup_axes(axi, **kwargs)
        axi.set_aspect("equal")
        axi.spines["left"].set_visible(False)
        axi.spines["bottom"].set_visible(False)
        axi.tick_params(bottom=False, left=False,
                        labelbottom=False, labelleft=False)

    def transverse(self, fig, heights=[0., .707, 1.],
                   wavelengths=None, nrays_line=152,
                   colors="grbcmyk"):
        if wavelengths is None:
            wavelengths = self.system.wavelengths
        ax = self.pre_setup_fanplot(fig, len(heights))
        p = self.system.object.pupil.distance
        for hi, axi in zip(heights, ax):
            axm, axsm, axss = axi
            axm.text(-.1, .5, "OY=%s" % hi, rotation="vertical",
                     transform=axm.transAxes,
                     verticalalignment="center")
            for wi, ci in zip(wavelengths, colors):
                t = GeometricTrace(self.system)
                t.rays_point((0, hi), wi, nrays=nrays_line,
                             distribution="tee", clip=True)
                # plot transverse image plane versus entrance pupil
                # coordinates
                y = t.y[-1, :, :2] - t.y[-1, t.ref, :2]
                py = t.y[0, :, :2] + p*tanarcsin(t.u[0])
                py -= py[t.ref]
                axm.plot(py[:t.ref, 1], y[:t.ref, 1], "-%s" % ci, label="%s" % wi)
                axsm.plot(py[t.ref:, 0], y[t.ref:, 1], "-%s" % ci, label="%s" % wi)
                axss.plot(py[t.ref:, 0], y[t.ref:, 0], "-%s" % ci, label="%s" % wi)
        for axi in ax:
            for axii in axi:
                self.post_setup_axes(axii)

    def spots(self, ax, heights=[1., .707, 0.],
              wavelengths=None, nrays=150, colors="grbcmyk"):
        paraxial = self.system.paraxial
        if wavelengths is None:
            wavelengths = self.system.wavelengths
        nd = ax.shape[1]
        for axi in ax.flat:
            self.pre_setup_xyplot(axi)
        z = paraxial.rayleigh_range[1]
        z = (np.arange(nd) - nd//2) * z
        for hi, axi in zip(heights, ax[:, 0]):
            axi.text(-.1, .5, "OY=%s" % hi, rotation="vertical",
                     transform=axi.transAxes, verticalalignment="center")
        for zi, axi in zip(z, ax[-1, :]):
            axi.text(.5, -.1, "DZ=%.1g" % zi,
                     transform=axi.transAxes, horizontalalignment="center")
        for hi, axi in zip(heights, ax):
            for wi, ci in zip(wavelengths, colors):
                r = paraxial.airy_radius[1]/paraxial.wavelength*wi
                t = GeometricTrace(self.system)
                t.rays_point((0, hi), wi, nrays=nrays,
                             distribution="hexapolar", clip=True)
                # plot transverse image plane hit pattern (ray spot)
                y = t.y[-1, :, :2] - t.y[-1, t.ref, :2]
                u = tanarcsin(t.i[-1])
                for axij, zi in zip(axi, z):
                    axij.add_patch(mpl.patches.Circle(
                        (0, 0), r, edgecolor=ci, facecolor="none"))
                    yi = y + zi*u
                    axij.plot(yi[:, 0], yi[:, 1], ".%s" % ci,
                              markersize=1, markeredgewidth=1, label="%s" % wi)
        for axi in ax:
            for axii in axi:
                self.post_setup_axes(axii)

    def opds(self, ax, heights=[0., .707, 1.],
             wavelength=None, nrays=1000, colors="grbcmyk"):
        paraxial = self.system.paraxial
        if wavelength is None:
            wavelength = self.system.wavelengths[0]
        mm = None
        rm = None
        for hi, axi in zip(heights, ax[:, 0]):
            axi.text(-.1, .5, "OY=%s" % hi, rotation="vertical",
                     transform=axi.transAxes, verticalalignment="center")
        for hi, axi in reversed(list(zip(heights, ax))):
            axo, axp, axe, axm = axi
            # TODO: link axes
            self.pre_setup_xyplot(axo)
            self.pre_setup_xyplot(axp)
            self.setup_axes(axe, "R", "E")
            self.setup_axes(axm, "F", "C")
            t = GeometricTrace(self.system)
            t.rays_point((0, hi), wavelength, nrays=nrays,
                         distribution="hexapolar", clip=True)
            try:
                x, y, o = t.opd()
            except ValueError:
                continue
            og = o[np.isfinite(o)]
            if mm is None:
                mm = np.fabs(og).max()
                v = np.linspace(-mm, mm, 21)
            axo.contour(x, y, o, v, cmap=plt.cm.RdBu_r)
            axo.text(.5, -.1, "PTP: %.3g" % og.ptp(),
                     transform=axo.transAxes, horizontalalignment="center")
            r = paraxial.airy_radius[1]/paraxial.wavelength*wavelength
            axp.add_patch(mpl.patches.Circle(
                (0, 0), r, edgecolor="green", facecolor="none"))
            x, y, psf = map(np.fft.fftshift, t.psf())
            x0 = (psf*x).sum()
            y0 = (psf*y).sum()
            x, y = x - x0, y - y0
            dx = x[1, 0] - x[0, 0]
            psfl = np.log10(psf)
            levels = psfl.max() - 1 - np.arange(4)
            axp.contour(x, y, psfl, levels, cmap=plt.cm.Reds, alpha=.2)
            levels = np.linspace(0, psf.max(), 21)
            axp.contour(x, y, psf, levels, cmap=plt.cm.Greys)
            ee = polar_sum(psf, (psf.shape[0]/2 + x0/dx,
                                 psf.shape[1]/2 + y0/dx), "azimuthal")
            ee = np.cumsum(ee)
            if rm is None:
                rm = np.searchsorted(ee, .9)*1.5*dx
            axp.set_xlim(-rm, rm)
            axp.set_ylim(-rm, rm)
            xe = np.arange(ee.size)*dx
            axe.plot(xe, ee, "k-")
            axe.set_xlim(0, rm)
            axe.set_ylim(0, 1)
            axe.set_aspect("auto")
            for i, ci in enumerate(("-", "--")):
                ot = np.fft.ifft(np.fft.ifftshift(psf.sum(i))*psf.size**.5)
                of = np.fft.fftfreq(ot.size, dx)
                ot, of = ot[:ot.size/2], of[:of.size/2]
                axm.plot(of, np.absolute(ot), "k"+ci)
                # axm.plot(of, ot.real, "k"+ci)
            axm.set_xlim(0, 1/r)
            axm.set_ylim(0, 1)
        for axi in ax:
            for axij in axi:
                self.post_setup_axes(axij)

    def longitudinal(self, ax, height=1.,
                     wavelengths=None, nrays=21, colors="grbcmyk"):
        # lateral color: image relative to image at wl[0]
        # focus shift paraxial focus vs wl
        # longitudinal spherical: marginal focus vs height (vs wl)
        if wavelengths is None:
            wavelengths = self.system.wavelengths
        axd, axc, axf, axs, axa = ax
        for axi, xl, yl, tl in [
                (axd, "EY", "REY", "DIST"),
                (axc, "EY", "DEY", "TCOLOR"),
                (axf, "EY", "DEZ", "ASTIG"),
                (axs, "PY", "DEZ", "SPHA"),
                (axa, "L", "DEZ", "LCOLOR"),
                ]:
            self.setup_axes(axi, xl, yl, tl, yzero=False, xzero=False)
        h = np.linspace(0, height*self.system.image.radius, nrays)
        for i, (wi, ci) in enumerate(zip(wavelengths, colors)):
            t = GeometricTrace(self.system)
            t.rays_line((0, height), wi, nrays=nrays)
            a, b, c = np.split(t.y[-1].T, (nrays, 2*nrays), axis=1)
            p, q, r = np.split(tanarcsin(t.i[-1]).T, (nrays, 2*nrays), axis=1)
            if i == 0:
                xd = (a[1] - h)/h
                xd[0] = np.nan
                axd.plot(a[1], xd, ci+"-", label="%s" % wi)
                a0 = a
            else:
                axc.plot(a[1], a[1] - a0[1], ci+"-", label="%s" % wi)
            xt = -(b[1]-a[1])/(q[1]-p[1])
            axf.plot(a[1], xt, ci+"-", label="EZt %s" % wi)
            xs = -(c[0]-a[0])/(r[0]-p[0])
            axf.plot(a[1], xs, ci+"--", label="EZs %s" % wi)
            t = GeometricTrace(self.system)
            t.rays_point((0, 0.), wi, nrays=nrays,
                         distribution="half-meridional", clip=True)
            p = self.system.object.pupil.distance
            py = t.y[0, :, 1] + p*tanarcsin(t.u[0])[:, 1]
            z = -t.y[-1, :, 1]/tanarcsin(t.i[-1])[:, 1]
            z[t.ref] = np.nan
            axs.plot(py, z, ci+"-", label="%s" % wi)
        wl, wu = min(wavelengths), max(wavelengths)
        ww = np.linspace(wl - (wu - wl)/4, wu + (wu - wl)/4, nrays)
        zc = []
        pd, ph = self.system.pupil((0, 0), wavelengths[0])
        t = GeometricTrace(self.system)
        for wwi in np.r_[wavelengths[0], ww]:
            y, u = self.system.aim((0, 0), (0, 1e-3), pd, ph)
            t.rays_given(y, u, wwi)
            t.propagate(clip=False)
            zc.append(-t.y[-1, 0, 1]/tanarcsin(t.i[-1, 0])[1])
        zc = np.array(zc[1:]) - zc[0]
        axa.plot(ww, zc, "-")
        for axi in ax:
            self.post_setup_axes(axi)
