import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from utils.ipbsd_utils import create_folder
from utils.utils_plotter import *


class Visualize:
    def __init__(self, export=False, filetype="emf", export_dir=None, flag=True):
        # Default color patterns
        self.color_grid = ['#840d81', '#6c4ba6', '#407bc1', '#18b5d8', '#01e9f5',
                           '#cef19d', '#a6dba7', '#77bd98', '#398684', '#094869']
        self.grayscale = ['#111111', '#222222', '#333333', '#444444', '#555555',
                          '#656565', '#767676', '#878787', '#989898', '#a9a9a9']

        # Set default font style
        font = {'size': 10}
        matplotlib.rc('font', **font)

        # Alternative font size
        self.FONTSIZE = 10

        # Exporting figures
        self.export = export
        self.filetype = filetype
        self.export_dir = export_dir
        self.flag = flag

    def plot_loss_curve(self, filename):
        """
        Plots the loss curves
        :param filename: str                    Loss curve path
        :return: None
        """
        with open(filename, "rb") as f:
            data = pickle.load(f)

        y = data["y"]
        y_fit = data["y_fit"]
        lam = data["mafe"]
        lam_fit = data["mafe_fit"]
        eal = data["eal"]
        PLS = data["PLS"]

        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
        plt.plot(y, lam, 'b')
        plt.plot(y_fit, lam_fit, 'r')
        # PLS points highlighted
        plt.scatter(y, lam, color='k', marker="o")

        # Fill the refined loss curve area
        ax.fill_between(y_fit, lam_fit, color='#FDEDEC')
        plt.yscale('log')
        plt.xlim(0.0, 1.0)
        plt.ylabel(r'MAFE, $\lambda$', fontsize=self.FONTSIZE)
        plt.xlabel(r'ELR, $y$', fontsize=self.FONTSIZE)
        plt.grid(True, which="major", axis='both', ls="--", lw=1.0)
        plt.grid(True, which="minor", axis='both', ls="--", lw=0.5)

        # EAL text
        plt.text(0.2, 0.15, f"EAL = {eal:.1f}%", ha="center", va="center", rotation=0, size=self.FONTSIZE, color='r',
                 bbox=dict(facecolor='none', edgecolor='red'), transform=ax.transAxes)

        # Plot PLS names
        for i in range(len(PLS)):
            plt.text(y[i] + 0.05, lam[i], PLS[i], ha="left", va="center", rotation=0, size=self.FONTSIZE)

        # Text for the curve names
        plt.text((y[1] + y[2]) / 2, (lam[1] + lam[2]) / 2, 'Approximate \nloss curve', ha="center", va="center",
                 size=self.FONTSIZE, color='b')
        plt.text(y[1], 0.2 * lam[1], 'Refined \nloss curve', ha="center", va="center",
                 size=self.FONTSIZE, color='r')

        if self.flag:
            plt.show()

        if self.export:
            export_figure(fig, filename=self.export_dir / "los_curve", filetype=self.filetype)

    def plot_spectrum(self, filename, x="Sd", y="Sa"):
        """
        Spectrum plotter
        :param filename: str                        File path
        :param x: str                               X axis nametag
        :param y: str                               Y axis nametag
        :return: None
        """
        spectrum = pd.read_csv(filename)
        xPlot = spectrum[x]
        yPlot = spectrum[y]

        # Labelling
        if x == "Sd":
            xlabel = r'Spectral displacement, $Sd$ [cm]'
        elif x == "Sa":
            xlabel = r'Spectral acceleration, $Sa$ [g]'
        else:
            xlabel = r'Period, $T$ [s]'

        if y == "Sd":
            ylabel = r'Spectral displacement, $Sd$ [cm]'
        elif y == "Sa":
            ylabel = r'Spectral acceleration, $Sa$ [g]'
        else:
            ylabel = r'Period, $T$ [s]'

        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
        plt.plot(xPlot, yPlot, color=self.color_grid[3])
        plt.ylabel(ylabel, fontsize=self.FONTSIZE)
        plt.xlabel(xlabel, fontsize=self.FONTSIZE)
        plt.grid(True, which="major", axis='both', ls="--", lw=1.0)
        plt.grid(True, which="minor", axis='both', ls="--", lw=0.5)
        plt.xlim([0, 1.1 * max(xPlot)])
        plt.ylim([0, 1.1 * max(yPlot)])
        plt.rc('xtick', labelsize=self.FONTSIZE)
        plt.rc('ytick', labelsize=self.FONTSIZE)

        if self.flag:
            plt.show()

        if self.export:
            export_figure(fig, filename=self.export_dir / "spectrum", filetype=self.filetype)

    def plot_solution_space(self, filename, spectrum_filename, direction=0):
        """
        Solution space visualization plotting
        :param filename: str                    File path
        :param spectrum_filename: str           Spectrum File path
        :param direction: int                   Direction of interest, 0 or 1
        :return: None
        """
        with open(filename, "rb") as f:
            data = pickle.load(f)

        spectrum = pd.read_csv(spectrum_filename)

        # Reading the information
        d = "x" if direction == 0 else "y"
        say = data["cy"]
        dy = data["dy"][direction]
        period_range = data["Period range"][d]
        muc = data["spo2ida"][d]["mc"]

        # Initial secant to yield period
        T1 = 2 * np.pi * (dy / say / 9.81) ** 0.5

        # Spectrum (e.g. at SLS)
        sa = spectrum["Sa"]
        sd = spectrum["Sd"]

        # Generate the figure
        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
        # plot the spectrum
        plt.plot(sd, sa, color="k", lw=0.5)
        plt.xlabel(r'Spectral displacement, $Sd$ [cm]', fontsize=self.FONTSIZE)
        plt.ylabel(r'Spectral acceleration, $Sa$ [g]', fontsize=self.FONTSIZE)
        plt.xlim([0, 2.0 * max(sd)])
        plt.ylim([0, 2.0 * max(sa)])
        # Get axis limits
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        # Plot the yield point and annotate
        dy *= 100
        plt.scatter(dy, say, color="r", marker="o")
        plt.plot([dy, dy], [0, say], color="r", ls="--")
        plt.plot([0, dy], [say, say], color="r", ls="--")
        plt.text(-0.8, 1.0 * say, f'{say:.2f}', ha="center", va="center", size=self.FONTSIZE, color='r')
        plt.text(1.0 * dy, -0.06, f'{dy:.1f}', ha="center", va="center", size=self.FONTSIZE, color='r')

        # Plot the backbone
        plt.plot([0.0, dy, dy * muc], [0.0, say, say], color=self.color_grid[2])

        # Annotate the ultimate point
        xlocMuc = min(0.9 * dy * muc, 0.9 * xmax)
        du = dy * muc
        plt.text(xlocMuc, 1.1 * say, f'{du:.1f}cm', ha="center", va="center", size=self.FONTSIZE, color='r')

        # Plot the period ranges
        tlower = period_range[0]
        salower = 0.8 * ymax
        sdlower = salower * 981 * (tlower / 2 / np.pi) ** 2
        tupper = period_range[1]
        sdupper = 0.6 * xmax
        saupper = sdupper / 981 * (np.pi * 2 / tupper) ** 2

        if saupper > 1.2 * salower:
            sdupper = 1.2 * salower * sdupper / saupper
            saupper = 1.2 * salower

        plt.plot([0, sdlower], [0, salower], color="k", lw=1)
        plt.plot([0, sdupper], [0, saupper], color="k", lw=1)

        # Annotate the period ranges
        plt.text(1.1 * sdupper, 1.1 * saupper, f'{tupper:.2f}s', ha="center", va="center", size=self.FONTSIZE,
                 color='k')
        plt.text(1.1 * sdlower, 1.1 * salower, f'{tlower:.2f}s', ha="center", va="center", size=self.FONTSIZE,
                 color='k')

        # Populate between the solution points (TODO, make more flexible)
        tRange = np.arange(tlower - 0.2, tupper + 0.2, 0.1)
        dispRange = np.linspace(dy - 0.1, dy + 4, tRange.shape[0])
        saRange = dispRange / 981 * (np.pi * 2 / tRange) ** 2
        plt.plot(dispRange, saRange, color="k")
        plt.text(4.5 * sdlower, 1.0 * salower, "Period range \n boundary", ha="center", va="center",
                 size=self.FONTSIZE, color='k')

        # Annotate period
        plt.text(dy, 1.3 * say, r"$T_1$=" + f"{T1:.2f}s", ha="center", va="center",
                 size=self.FONTSIZE, color=self.color_grid[2])

        # Highlight the design solution space
        # ax.fill_between([0, 10], [0, saupper], [0, salower])

        # Some more figure manipulation
        # Hide the grid
        plt.grid(False)
        # Hide the right and top spines
        for side in ['bottom', 'right', 'top', 'left']:
            ax.spines[side].set_visible(False)

        # get width and height of axes object to compute
        # matching arrowhead length and width
        dps = fig.dpi_scale_trans.inverted()
        bbox = ax.get_window_extent().transformed(dps)
        width, height = bbox.width, bbox.height

        # manual arrowhead width and length
        hw = 1. / 20. * (ymax - ymin)
        hl = 1. / 20. * (xmax - xmin)
        lw = 1.  # axis line width
        ohg = 0.3  # arrow overhang

        # compute matching arrowhead length and width
        yhw = hw / (ymax - ymin) * (xmax - xmin) * height / width
        yhl = hl / (xmax - xmin) * (ymax - ymin) * width / height

        # draw x and y axis
        ax.arrow(xmin, 0, xmax - xmin, 0., fc='k', ec='k', lw=lw,
                 head_width=hw, head_length=hl, overhang=ohg,
                 length_includes_head=True, clip_on=False)

        ax.arrow(0, ymin, 0., ymax - ymin, fc='k', ec='k', lw=lw,
                 head_width=yhw, head_length=yhl, overhang=ohg,
                 length_includes_head=True, clip_on=False)

        if self.flag:
            plt.show()

        if self.export:
            export_figure(fig, filename=self.export_dir / f"solution_space_{direction}", filetype=self.filetype)

    def plot_spo2ida_outputs(self, filename, direction=0):
        """
        SPO2IDA plotter
        :param filename: str
        :param direction: int
        :return: None
        """
        d = "x" if direction == 0 else "y"

        with open(filename, "rb") as f:
            data = pickle.load(f)

        data = data[d]
        R16 = data["R16"]
        R50 = data["R50"]
        R84 = data["R84"]

        fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
        plt.plot(data["spom"], data["spor"], color=self.grayscale[0], label="SPO")
        plt.plot(data["idacm"][0], data["idacr"][0], color=self.grayscale[2], label=r"IDA, $84^{th}$ percentile")
        plt.plot(data["idacm"][1], data["idacr"][1], color=self.grayscale[4], label=r"IDA, $50^{th}$ percentile")
        plt.plot(data["idacm"][2], data["idacr"][2], color=self.grayscale[6], label=r"IDA, $16^{th}$ percentile")
        plt.scatter([data["spom"][-1]] * 3, [R16, R50, R84], color="k", label="Collapse capacity")

        plt.xlabel(r"Ductility, $\mu$", fontsize=self.FONTSIZE)
        plt.ylabel(r'$q_\mathrm{\mu} = Sa/Sa_\mathrm{y}$', fontsize=self.FONTSIZE)
        plt.grid(True, which="major", axis='both', ls="--", lw=1.0)
        plt.grid(True, which="minor", axis='both', ls="--", lw=0.5)
        plt.xlim([0, int(max(data["spom"])) + 3])
        plt.ylim([0, int(max(data["idacr"][0])) + 2])
        plt.rc('xtick', labelsize=self.FONTSIZE)
        plt.rc('ytick', labelsize=self.FONTSIZE)
        plt.legend(frameon=False, loc='upper right', bbox_to_anchor=(1.7, 1), fontsize=self.FONTSIZE)

        # Annotating
        mc = data["spom"][2] / data["spom"][1]
        mf = data["spom"][3] / data["spom"][1]

        # Hardening ductility
        plt.text(mc, data["spor"][2] + 0.5, r'$\mu_c = %.1f$' % mc, ha="center", va="center", size=self.FONTSIZE,
                 color='b')
        # Fracturing ductility
        plt.text(mf, -0.8, r'$\mu_f = %.1f$' % mf, ha="left", va="center", size=self.FONTSIZE,
                 color='b')
        # Collapse capacity points
        plt.text(mf, R84 + 0.5, f'{R84:.2f}', ha="left", va="center", size=self.FONTSIZE, color='b')
        plt.text(mf, R50 + 0.5, f'{R50:.2f}', ha="left", va="center", size=self.FONTSIZE, color='b')
        plt.text(mf, R16 + 0.5, f'{R16:.2f}', ha="left", va="center", size=self.FONTSIZE, color='b')

        # Hardening slope
        ac = (data["spor"][2] - data["spor"][1]) / (mc - 1)
        plt.text(mc / 2, data["spor"][1] - 0.5, r'$a_c = %.2f$' % ac, ha="left", va="center", size=self.FONTSIZE,
                 color='b')

        # Softening slope
        app = (data["spor"][2] - data["spor"][3]) / (mc - mf)
        plt.text((mc + mf) / 2, data["spor"][1], r'$a_{pp} = %.2f$' % app, ha="left", va="center", size=self.FONTSIZE,
                 color='b')

        # Initial secant to yield period
        try:
            plt.text(0.1, 0.9, r'$T_1 = %.1f$' % data["T1"], ha="center", va="center", size=self.FONTSIZE, color='b',
                     transform=ax.transAxes)
        except:
            pass

        if self.flag:
            plt.show()

        if self.export:
            export_figure(fig, filename=self.export_dir / f"spo2ida_{direction}", filetype=self.filetype)

    def plot_spo(self, filename, solution_filename=None, spo2ida_filename=None, n_seismic=2, direction="x"):
        """
        SPO plotter
        :param filename: str
        :param solution_filename: str
        :param spo2ida_filename: str
        :param n_seismic: int
        :param direction: str
        :return: None
        """
        if solution_filename is not None:
            with open(solution_filename, "rb") as f:
                sol = pickle.load(f)
        if spo2ida_filename is not None:
            with open(spo2ida_filename, "rb") as f:
                spo2ida = pickle.load(f)

        with open(filename, "rb") as f:
            data = pickle.load(f)

        # IPBSD outputs
        if sol is not None:
            d = 0 if direction == "x" else 1
            # Yield Sa (SDOF) (used for designing the structure), reference value
            cy = sol["cy"]
            # Yield Sa (MDOF) including overstrength factor
            say = cy * sol["overstrength"][d] * sol["part_factor"][d]
            # Yield displacement (MDOF)
            dy = sol["dy"][d] * sol["overstrength"][d] * sol["part_factor"][d]
            # Yield base shear
            Vy = 9.81 * sol["Mstar"][d] * say
            period = 2 * np.pi * np.sqrt(sol["dy"][d] / cy / 9.81)
            print(f"[PERIOD] {period:.2f}")

        # Nonlinear model outputs
        try:
            model = data["SPO"][direction]
            spo = data["SPO_idealized"][direction]
        except:
            model = data["SPO"]
            spo = data["SPO_idealized"]

        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
        plt.plot(spo[0] * 100, spo[1], color=self.grayscale[0], label="Idealized shape")
        plt.plot(model[0] * 100, model[1], color=self.grayscale[-2], label="Nonlinear model")

        if spo2ida is not None and sol is not None:
            # Plotting the SPO2IDA shape
            plt.plot(spo2ida[direction]["spom"] * dy * 100, spo2ida[direction]["spor"] * Vy * n_seismic,
                     color="r", ls="--", label="Design")

        plt.xlabel("Top displacement [cm] ", fontsize=self.FONTSIZE)
        plt.ylabel('Base shear [kN]', fontsize=self.FONTSIZE)
        plt.grid(True, which="major", axis='both', ls="--", lw=1.0)
        plt.grid(True, which="minor", axis='both', ls="--", lw=0.5)
        plt.xlim([0, int(max(spo[0]) * 100) + 20])
        plt.ylim([0, int(max(spo[1])) + 300])
        plt.rc('xtick', labelsize=self.FONTSIZE)
        plt.rc('ytick', labelsize=self.FONTSIZE)
        plt.legend(frameon=False, loc='upper right', fontsize=self.FONTSIZE)

        if self.flag:
            plt.show()

        if self.export:
            export_figure(fig, filename=self.export_dir / f"spo2ida_{direction}", filetype=self.filetype)


if __name__ == "__main__":
    path = Path.cwd().parents[0]
    export_dir = path / "sample/figs"
    create_folder(export_dir)

    loss_curve = path / "sample/sample1/Cache/lossCurve.pickle"
    spectrum = path / "sample/sample1/Cache/sls_spectrum.csv"
    solution = path / "sample/sample1/Cache/ipbsd.pickle"
    spo2ida = path / "sample/sample1/Cache/spoAnalysisCurveShape.pickle"
    spo_model = path / "sample/sample1/Cache/modelOutputs.pickle"
    n_seismic = 1
    direction = "x"

    viz = Visualize(export=False, filetype="png", export_dir=export_dir, flag=True)
    viz.plot_loss_curve(loss_curve)
    viz.plot_spectrum(spectrum)
    viz.plot_solution_space(solution, spectrum, direction=0)
    viz.plot_spo2ida_outputs(spo2ida, 0)
    viz.plot_spo(spo_model, solution, spo2ida, n_seismic, direction)
