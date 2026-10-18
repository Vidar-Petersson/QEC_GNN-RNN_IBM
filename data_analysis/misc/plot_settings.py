def set_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1/72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio  #* 1.69

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim

def set_square_size(width, fraction=1):
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27  # TeX points
    fig_width_in = fig_width_pt * inches_per_pt
    print("Width: ", fig_width_in)

    # Set height equal to width for a square figure
    fig_dim = (fig_width_in, 3.7795276878372768)

    return fig_dim

import shutil
import matplotlib.pyplot as plt

  
tex_fonts = {

    # Font sizes chosen to match the LaTeX report
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "font.size": 12,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,

    # Grid lines
    "axes.grid" : True,
    "axes.axisbelow" : True,
    "grid.color" : '#DDDDDD',
    "grid.linewidth" : "0.8",

    # Legend
    "legend.frameon" : True,
    "legend.framealpha" : 0.7,
    "legend.fancybox" : True,
    "legend.numpoints" : 1,

    # Set x axis
    "xtick.direction" : "in",
    "xtick.major.size" : 3,
    "xtick.major.width" : 0.5,
    "xtick.minor.size" : 1.5,
    "xtick.minor.width" : 0.5,
    "xtick.minor.visible" : True,
    "xtick.top" : True,

    # Set y axis
    "ytick.direction" : "in",
    "ytick.major.size" : 3,
    "ytick.major.width" : 0.5,
    "ytick.minor.size" : 1.5,
    "ytick.minor.width" : 0.5,
    "ytick.minor.visible" : True,
    "ytick.right" : True,

    # Set line widths
    "axes.linewidth" : 0.5,
    "grid.linewidth" : 0.5,
    "lines.linewidth" : 1.,

    # Always save as 'tight'
    "savefig.bbox" : "tight",
    "savefig.pad_inches" : 0.02,

    # Use serif fonts
    "font.serif" : "Computer Modern Serif, DejaVu Serif",
    "font.family" : "serif",
    "axes.formatter.use_mathtext" : True,
    "mathtext.fontset" : "cm",

    # Use LaTeX for text rendering if a LaTeX installation is available
    "text.usetex" : shutil.which("latex") is not None,
    'text.latex.preamble' : r'\usepackage{gensymb}'

}

plt.rcParams.update(tex_fonts)

width = 455.24411  # Text width of the LaTeX document in pt
plt.rcParams["figure.figsize"] = set_size(width, fraction=1)
