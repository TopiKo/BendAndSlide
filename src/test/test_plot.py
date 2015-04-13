'''
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator

s = {'t': 1,
     'x': [1, 2, 3, 4, 5, 6, 7, 8],
     'T': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
     'D': [0.3, 0.5, 0.2, 0.3, 0.5, 0.5, 0.3, 0.4]}

width = 40

tot = np.repeat(s['D'],width).reshape(len(s['D']), width)
tot2 = np.repeat(s['T'],width).reshape(len(s['D']), width)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4)

fig.suptitle('Title of figure', fontsize=20)

# Line plots
ax1.set_title('Title of ax1')
ax1.plot(s['x'], s['T'])
ax1.set_ylim(0,1)

ax2.set_title('Title of ax2')
ax2.plot(s['x'], s['D'])
# Set locations of ticks on y-axis (at every multiple of 0.25)
ax2.yaxis.set_major_locator(MultipleLocator(0.25))
# Set locations of ticks on x-axis (at every multiple of 2)
ax2.xaxis.set_major_locator(MultipleLocator(2))
ax2.set_ylim(0,1)

ax3.set_title('Title of ax3')
# Display image, `aspect='auto'` makes it fill the whole `axes` (ax3)
im3 = ax3.imshow(tot, norm=LogNorm(vmin=0.001, vmax=1), aspect='auto')
# Create divider for existing axes instance
divider3 = make_axes_locatable(ax3)
# Append axes to the right of ax3, with 20% width of ax3
cax3 = divider3.append_axes("right", size="20%", pad=0.05)
# Create colorbar in the appended axes
# Tick locations can be set with the kwarg `ticks`
# and the format of the ticklabels with kwarg `format`
cbar3 = plt.colorbar(im3, cax=cax3, ticks=MultipleLocator(0.2), format="%.2f")
# Remove xticks from ax3
ax3.xaxis.set_visible(False)
# Manually set ticklocations
ax3.set_yticks([0.0, 2.5, 3.14, 4.0, 5.2, 7.0])

ax4.set_title('Title of ax4')
im4 = ax4.imshow(tot2, norm=LogNorm(vmin=0.001, vmax=1), aspect='auto')
divider4 = make_axes_locatable(ax4)
cax4 = divider4.append_axes("right", size="20%", pad=0.05)
cbar4 = plt.colorbar(im4, cax=cax4)
ax4.xaxis.set_visible(False)
# Manually set ticklabels (not ticklocations, they remain unchanged)
ax4.set_yticklabels([0, 50, 30, 'foo', 'bar', 'baz'])

plt.tight_layout()
# Make space for title
plt.subplots_adjust(top=0.85)
plt.show()
'''


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


c = mcolors.ColorConverter().to_rgb
#rvb = make_colormap(
#    [c('red'), c('violet'), 0.33, c('violet'), c('blue'), 0.66, c('blue')])
rvb         =   make_colormap([c('blue'), c('white'), 0.2, c('white'), c('green'), 0.4, c('green'), \
                                 c('red'), 0.6, c('red'), c('violet'), 0.8, c('violet'), c('black') ])

N = 1000
array_dg = np.random.uniform(0, 10, size=(N, 2))
colors = np.random.uniform(-2, 2, size=(N,))
plt.scatter(array_dg[:, 0], array_dg[:, 1], c=colors, cmap=rvb)
plt.colorbar()
plt.show()