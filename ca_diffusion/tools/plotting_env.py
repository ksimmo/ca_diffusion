import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler

#TODO: maybe export this to a config file so it is more dynamic
def initialize_matplotlib():
    #I mainly copied these from: https://github.com/garrettj403/SciencePlots/blob/master/scienceplots/styles/science.mplstyle

    matplotlib.use("agg")
    #set plotting parameters
    plt.rcParams["figure.figsize"] = (3.3, 2.5)
    plt.rcParams["figure.dpi"] = 600

    plt.rcParams["axes.prop_cycle"] = cycler(color=['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e'])

    #font
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times"
    plt.rcParams["font.size"] = 8 #10
    #plt.rcParams["mathtext.fontset"] = "dejavuserif"

    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["xtick.major.size"] = 3
    plt.rcParams["xtick.major.width"] = 0.5
    plt.rcParams["xtick.minor.size"] = 1.5
    plt.rcParams["xtick.minor.width"] = 0.5
    plt.rcParams["xtick.minor.visible"] = True
    plt.rcParams["xtick.top"] = True

    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["ytick.major.size"] = 3
    plt.rcParams["ytick.major.width"] = 0.5
    plt.rcParams["ytick.minor.size"] = 1.5
    plt.rcParams["ytick.minor.width"] = 0.5
    plt.rcParams["ytick.minor.visible"] = True
    plt.rcParams["ytick.right"] = True

    #line widths
    plt.rcParams["axes.linewidth"] = 0.5
    plt.rcParams["grid.linewidth"] = 0.5
    plt.rcParams["lines.linewidth"] = 0.5 #1.

    plt.rcParams["errorbar.capsize"] = 2
    plt.rcParams["lines.markeredgewidth"] = 0.5

    plt.rcParams["legend.frameon"] = False
    plt.rcParams["legend.fontsize"] = 6 #"medium"


    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.pad_inches"] = 0.05

    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = "\n".join([ # plots will use this preamble
            r"\usepackage{amsmath}",
            r"\usepackage{amssymb}",
            #r"\usepackage[utf8]{inputenc}",
            r"\usepackage[T1]{fontenc}",
            #r"\usepackage[detect-all,locale=DE]{siunitx}",
            #r"\usepackage{pgfplots}"
            ])