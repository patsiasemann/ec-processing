

def plot_wind_rose(ax, directions, title, n_bins=16, color="blue"):
    """
    Plot an individual wind rose
    """
    dirs = pd.to_numeric(directions, errors='coerce').dropna() % 360
    bins = np.linspace(0, 360, n_bins + 1)
    counts, _ = np.histogram(dirs, bins=bins)
    widths = np.deg2rad(np.diff(bins))
    angles = np.deg2rad(bins[:-1]) + widths / 2
    freq = counts / counts.sum() * 100
    ax.bar(angles, freq, width=widths, bottom=0, align='center', color=color, edgecolor='k', linewidth=0.3, alpha=0.7)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title(title, pad=15)
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
    ax.set_yticks([5, 10, 15, 20, 25])
    ax.set_yticklabels([f"{int(t)}%" for t in ax.get_yticks()])
    ax.grid(True, alpha=0.3)

def compute_freq(directions, bins):
    dirs = pd.to_numeric(directions, errors='coerce').dropna() % 360
    counts, _ = np.histogram(dirs, bins=bins)
    return counts / counts.sum() * 100