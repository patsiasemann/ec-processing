import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde
from matplotlib.patches import Ellipse
from numpy.lib.stride_tricks import sliding_window_view

cmap = matplotlib.colormaps["tab20"]

