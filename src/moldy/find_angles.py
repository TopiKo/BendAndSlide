'''
Created on 9.2.2015

@author: tohekorh
'''
import numpy as np
from aid.help import find_layers, make_colormap, shiftedColorMap
from scipy.optimize import curve_fit, fmin_l_bfgs_b
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib import gridspec
import matplotlib.colors as mcolors

s_f_size    =   (4,1.8)
ball_size   =   5

def kink_func(xs,*args):
    
    a2,b1,b2,c   =   args
    ys              =   np.zeros(len(xs))
    # print c, b1
    for i, x in enumerate(xs):
        
        if  x <= c: ys[i]   =    b1
        elif c < x: ys[i]   =    a2*x + b2    
    
    return ys 

def kink_func2(xs, *args):
    
    R, phi, z0, x0  =   args
    ys              =   np.zeros(len(xs))
    
    for i, x in enumerate(xs):
        
        if  x <= x0: 
            ys[i]   =    z0
        elif x0 < x <= x0 + R*np.sin(phi): 
            ys[i]   =    np.sqrt(R**2 - (x - x0)**2) + (z0 - R)    
        elif x0 + R*np.sin(phi) < x: 
            ys[i]   =    -np.tan(phi)*x + z0 - R*(1-np.cos(phi)) \
                                        + np.tan(phi)*(x0 + R*np.sin(phi))    
        
    return ys 

def kink_func3(popsis, *args):
    
    R, phi, z0, x0      =   popsis
    xs,y                =   args[0], args[1]
    ys                  =   np.zeros(len(xs))
    
    for i, x in enumerate(xs):
        
        if  x <= x0: 
            ys[i]   =    z0
        elif x0 < x <= x0 + R*np.sin(phi): 
            ys[i]   =    np.sqrt(R**2 - (x - x0)**2) + (z0 - R)    
        elif x0 + R*np.sin(phi) < x: 
            ys[i]   =    -np.tan(phi)*x + z0 - R*(1-np.cos(phi)) \
                                        + np.tan(phi)*(x0 + R*np.sin(phi))    
    
    Rsqr    =   0.
    for i in range(len(y)):
        Rsqr    +=  (ys[i] - y[i])**2   
        
    return Rsqr 


def plot_x_shift(ax, shift_table, edge, bond):
    
    xs      =   shift_table[:,0]
    shifts  =   shift_table[:,1]
    
    ax.scatter(xs, shifts)
    if edge == 'zz':
        ax.plot(xs, np.ones(len(xs))*np.sqrt(3)*bond, '-.', color = 'black')
        ax.plot(xs, np.ones(len(xs))*2*np.sqrt(3)*bond, '-.', color = 'black')
    elif edge == 'arm':
        ax.plot(xs, np.ones(len(xs))*2*bond, '-.', color = 'black')
        ax.plot(xs, np.ones(len(xs))*3*bond, '-.', color = 'black')
    
    plt.show()

def plot_KC_atoms(atoms, e_KC, layer_indices, edif_max, limits, path_to_fig):
    
    fig2, ax    =   plt.subplots(1, figsize = (8,5))

    xdata       =   atoms.positions[:,0]
    ydata       =   atoms.positions[:,2]
    
    s0          =   ball_size
    s           =   [s0 + e_KC[i]/edif_max*s0**np.sqrt(1.5) for i in range(len(xdata))]
    
    axf         =   ax.scatter(xdata, ydata, c=e_KC, s=s, cmap=mpl.cm.cool, \
                               vmin=-edif_max, vmax=edif_max, edgecolors='none')
    
    #cmaps: RdBu
    
    plt.colorbar(axf)
    ax.axis('equal') 
    plt.xlim(limits[0] - 10, limits[1] + 10)
    plt.ylim(limits[2], limits[3])
    
    plt.savefig(path_to_fig, dpi = 100)
    #plt.show()
    
def plot_atoms2(positions_t, angles_t, angles_av_t, Rads_t, \
               z0s_t, x0s_t, yav_t, ep, N, edge, z_t, bond, iz, \
               edif_min, edif_max, limits, line_limits, e_KC, shift_table, \
               pair_table, path_to_fig):
    
    
    fig2, axs    =   plt.subplots(4, figsize = (10,24))

    ax1, ax2, ax3, ax4   =   axs
    
    top_indices     =   find_layers(positions_t[0])[1][-2:]
    layer_indices   =   find_layers(positions_t[0])[1][:-2]
    
    positions       =   positions_t[iz]   
    xdata       =   positions[:,0]
    ydata       =   positions[:,2]
    zsets       =   np.zeros((1000, len(layer_indices)))
    xsets       =   np.zeros((1000, len(layer_indices)))

    #print iz, ydata
    

    n           =   len(layer_indices) 
    angles_av   =   0.
    
    angles  =   angles_t[iz]
    Rads    =   Rads_t[iz]
    z0s     =   z0s_t[iz]
    x0s     =   x0s_t[iz]
    z       =   z_t[iz]
    
    
    for i in range(n):
        x_min   =   np.min(xdata[layer_indices[i]])
        x_max   =   np.max(xdata[layer_indices[i]])
        
        xsets[:,i]      =   np.linspace(x_min, x_max, 1000)
        
        R, phi, z0, x0  =   Rads[i], angles[i]/360.*2*np.pi, z0s[i], x0s[i] #par_set[j, i]
        
        zsets[:,i]      =   kink_func2(xsets[:,i], R, phi, z0, x0)
        
        angles_av  +=   angles[i]/n   
    
    
    
    
    #
    # Using contourf to provide my colorbar info, then clearing the figure
    shift_cmap      =   plt.get_cmap('Reds') 
    Z           =   [[0,0],[0,0]]
    levels      =   np.linspace(line_limits[8] - .1, line_limits[9] + .1, 1000) 
    
    
    cNorm       =   colors.Normalize(vmin=-1, vmax=6)
    scalarMap   =   cmx.ScalarMappable(norm=cNorm, cmap=shift_cmap)
    shift_xs    =   shift_table[:,2]
    
    
    for idz in range(len(positions)):
        if idz not in top_indices[0] and idz not in layer_indices[-1]: 
            ri  =  positions[idz]
            if 1 in pair_table[idz]:
                j   =   np.where(pair_table[idz]==1)[0][0]     
                rj  =   positions[j]    
                ids =   np.where(shift_xs == idz)[0][0]
                colorVal = scalarMap.to_rgba(shift_table[ids,1])
                ax1.plot([ri[0], rj[0]], [ri[2], rj[2]], alpha = 1, lw = 1, color=colorVal)
                

    
    # COLOR PLOT
    s0          =   10
    #s           =   [s0 + e_KC[i]/edif_max*s0**np.sqrt(1.5) for i in range(len(xdata))]
    
    for i in range(n):
        ax1.plot(xsets[:,i], zsets[:,i], alpha = .3,  color = 'black')
        
    axf         =   ax1.scatter(xdata, ydata, c=e_KC, s=s0, cmap=mpl.cm.RdBu_r, \
                               vmin=edif_min, vmax=edif_max, edgecolors='none', zorder = 10000)
    
    
    
    
    ax1.set_ylim(limits[2], limits[3])
    ax1.set_aspect('equal') 
    divider3    =   make_axes_locatable(ax1)
    divider4    =   make_axes_locatable(ax1)
    
    cax3        =   divider3.append_axes("bottom", size="10%", pad=0.05)
    cax4        =   divider4.append_axes("right", size="5%", pad=0.5)
    
    cbar3       =   plt.colorbar(axf, cax=cax3, orientation = 'horizontal',\
                                  ticks = np.linspace(edif_min, edif_max, 5))
    
    axf2        =   plt.contourf(Z, levels, cmap=shift_cmap)
    cbar4       =   plt.colorbar(axf2, cax=cax4, \
                                 ticks = np.linspace(line_limits[8], line_limits[9], 5)) #CS3, cax=cax4, norm=cNorm, orientation='vertical')
    
    
    #cbar3.ax.set_xlabel('e_KC eV')
    ax1.set_title(r'KC-energy eV, edge=%s, N=%i' %(edge, N))
    
    y_scale =   limits[3] - limits[2]
    ax1.text(x_min, limits[2] + y_scale/10, r'angle = %.2f Deg' %angles_av, fontsize=12)
    ax1.text(x_min, limits[2] + y_scale/10*2, r'z = %.2f Angst' %z, fontsize=12)
    
    ax1.axis('off')
    
    
    
    ax2.plot(z_t[:iz], angles_av_t[:iz], color = 'black', label=r'angle')
    
    ax2a =   plt.twinx(ax2)
    ax2a.plot(z_t[:iz], ep[:iz], '-.', color = 'black', label = r'eP')
    
    ax2.legend(loc = 2, frameon = False)
    ax2a.legend(loc = 4, frameon = False)

    ax2.set_ylim(line_limits[4], line_limits[5])
    ax2.set_xlim(line_limits[0], line_limits[1])
    ax2a.set_ylim(line_limits[2], line_limits[3])
    
    ax2.set_ylabel(r'Angle deg')
    ax2a.set_ylabel(r'Pot E eV')
    
    for ai in range(len(yav_t[0])):
        ax3.plot(z_t[:iz], yav_t[:iz,ai], color = 'black', label = r'layer %i' %ai)
        
    ax3.set_xlim(line_limits[0], line_limits[1])
    ax3.set_ylim(line_limits[6], line_limits[7])
    
    ax3.set_ylabel(r'Shift y')
    ax3.set_xlabel(r'bend z')

    # CORRUGATION
    xs      =   shift_table[:,0]
    shifts  =   shift_table[:,1]
    ax4.scatter(xs, shifts, color = 'black', alpha = .6)
    if edge == 'zz':
        ax4.plot(xs, np.ones(len(xs))*np.sqrt(3)*bond, '-.', color = 'black')
        ax4.plot(xs, np.ones(len(xs))*2*np.sqrt(3)*bond, '-.', color = 'black')
    elif edge == 'arm':
        ax4.plot(xs, np.ones(len(xs))*1*bond, '-.', color = 'black')
        ax4.plot(xs, np.ones(len(xs))*2*bond, '-.', color = 'black')
        ax4.plot(xs, np.ones(len(xs))*3*bond, '-.', color = 'black')
    
        
    ax4.set_ylim(line_limits[8] - 1, line_limits[9] + 1)
    # END CORRUGATION

    
    plt.savefig(path_to_fig, dpi = 100)
    plt.clf()
    plt.close()
    
def plot_KC_single(positions_t, angles_t, angles_av_t, Rads_t, \
                   z0s_t, x0s_t, yav_t, strech_t, ep, N, edge, stack, z_t, bond, izs, \
                   edif_min, edif_max, limits, line_limits, e_KC, shift_table, \
                   path_to_fig):
    
    
    
    _, axs          =   plt.subplots(len(izs), figsize = (s_f_size[0], s_f_size[1]*len(izs)*1.3))
    
    for i, iz in enumerate(izs):
    
        layer_indices_f \
                    =   find_layers(positions_t[0])[1]
        positions   =   positions_t[iz]   
        angles      =   angles_t[iz]
        Rads        =   Rads_t[iz]
        z0s         =   z0s_t[iz]
        x0s         =   x0s_t[iz]
        z           =   z_t[iz]  
        
        if i == 0:
            title   =   'KC and corrugation %s, %s' %(edge, stack)
        else:
            title   =   '' 
        
        plot_KC(axs[i], N, positions, e_KC[iz] - e_KC[0], layer_indices_f, angles, Rads, z0s, x0s, z, \
                limits, line_limits, shift_table[iz], edif_min, edif_max, edge, bond, title)
    
    plt.savefig(path_to_fig, dpi = 125)
    
    plt.clf()
    plt.close()

def plot_Cor_single(positions_t, angles_t, angles_av_t, Rads_t, \
                   z0s_t, x0s_t, yav_t, strech_t, ep, N, edge, stack, z_t, bond, izs, \
                   edif_min, edif_max, limits, line_limits, shift_table, path_to_fig):
    
    _, axs          =   plt.subplots(len(izs), figsize = (s_f_size[0], s_f_size[1]*len(izs)))
    

    for i, iz in enumerate(izs):
        
        if i == 0:
            title   =   'Corrugation %s, %s' %(edge, stack)
        else:
            title   =   '' 
        
        plot_corrugation(axs[i], shift_table[iz], bond, limits, line_limits, edge, title)
    
    plt.savefig(path_to_fig, dpi = 125)
    
    plt.clf()
    plt.close()


def plot_il_single(positions_t, angles_t, angles_av_t, Rads_t, \
                   z0s_t, x0s_t, yav_t, strech_t, ep, N, edge, stack, z_t, bond, izs, \
                   limits, line_limits, il_dist, path_to_fig):
    
    _, axs          =   plt.subplots(len(izs), figsize = (s_f_size[0], s_f_size[1]*len(izs)))
    
    layer_indices_f \
                    =   find_layers(positions_t[0])[1]

    for i, iz in enumerate(izs):
        
        if i == 0:
            title   =   'interlayer dist %s, %s' %(edge, stack)
        else:
            title   =   '' 
            
        positions   =   positions_t[iz]   
        
        plot_il(axs[i], N, positions, layer_indices_f, limits, il_dist[iz], title)
    
    plt.savefig(path_to_fig, dpi = 125)
    
    plt.clf()
    plt.close()

def plot_strech_single(positions_t, angles_t, angles_av_t, Rads_t, \
                   z0s_t, x0s_t, yav_t, strech_t, ep, N, edge, stack, z_t, bond, izs, \
                   limits, line_limits, path_to_fig):
    
    _, axs          =   plt.subplots(len(izs), figsize = (s_f_size[0], s_f_size[1]*len(izs)))
    
    for i, iz in enumerate(izs):
        
        if i == 0:
            title   =   'Average bond lengths %s, %s' %(edge, stack)
        else:
            title   =   '' 
        plot_streches(axs[i], strech_t[iz], limits, title)
    
    plt.savefig(path_to_fig, dpi = 125)
    
    plt.clf()
    plt.close()

def plot_sY_single(positions_t, angles_t, angles_av_t, Rads_t, \
                   z0s_t, x0s_t, yav_t, strech_t, ep, N, edge, stack, z_t, bond, iz, \
                   edif_min, edif_max, limits, line_limits, e_KC, shift_table, \
                   il_dist, path_to_fig):
    
    _, ax1  =   plt.subplots(1, figsize = s_f_size)
    
    plot_shift_Y(ax1, z_t, yav_t, iz, line_limits)
    
    plt.savefig(path_to_fig, dpi = 125)
    
    plt.clf()
    plt.close()

def plot_aEp_single(positions_t, angles_t, angles_av_t, Rads_t, \
                   z0s_t, x0s_t, yav_t, strech_t, ep, N, edge, stack, z_t, bond, iz, \
                   edif_min, edif_max, limits, line_limits, e_KC, shift_table, \
                   il_dist, path_to_fig):
    
    _, ax1  =   plt.subplots(1, figsize = s_f_size)

    plot_angle_epot(ax1, ep, z_t, angles_av_t, iz, line_limits)
    
    plt.savefig(path_to_fig, dpi = 125)
    
    plt.clf()
    plt.close()

def plot_atoms3(positions_t, angles_t, angles_av_t, Rads_t, \
               z0s_t, x0s_t, yav_t, strech_t, ep, N, edge, z_t, bond, iz, \
               edif_min, edif_max, limits, line_limits, e_KC, shift_table, \
               il_dist, path_to_fig):
    
    
    _       =   plt.subplots(figsize = (6,14))
    
    gs      =   gridspec.GridSpec(5, 1, height_ratios=[2, 2, 2, 1, 1]) 
    #gs      =   gridspec.GridSpec(3, 1, height_ratios=[2, 2, 1]) 

    ax1     =   plt.subplot(gs[0])
    ax2     =   plt.subplot(gs[1])
    ax3     =   plt.subplot(gs[2])
    ax4     =   plt.subplot(gs[3])
    ax5     =   plt.subplot(gs[4])
    #ax6     =   plt.subplot(gs[5])
    
    layer_indices_f \
                =   find_layers(positions_t[0])[1]
    positions   =   positions_t[iz]   
    angles      =   angles_t[iz]
    Rads        =   Rads_t[iz]
    z0s         =   z0s_t[iz]
    x0s         =   x0s_t[iz]
    z           =   z_t[iz]   
    
   
    ########################
    plot_KC(ax1, N, positions, e_KC, layer_indices_f, angles, Rads, z0s, x0s, z, \
            limits, line_limits, shift_table, edif_min, edif_max, edge, bond, \
            'KC-energy eV, edge=%s, N=%i' %(edge, N))
    
    plot_il(ax2, N, positions, layer_indices_f, limits, il_dist, 'Interlayer distance')
    
    plot_streches(ax3, strech_t[iz], limits, 'Average bond lengths')
    
    plot_corrugation(ax4, shift_table, bond, limits, line_limits, edge)
    
    plot_shift_Y(ax5, z_t, yav_t, iz, line_limits)
    
    #plot_angle_epot(ax6, ep, z_t, angles_av_t, iz, line_limits)

        
    plt.savefig(path_to_fig, dpi = 125)
    
    plt.clf()
    plt.close()

def plot_corrugation(ax, shift_table, bond, limits, line_limits, edge, title = ''):
    
    xs      =   shift_table[:,1]
    shifts  =   shift_table[:,-2]
    ax.scatter(xs, shifts, color = 'black', alpha = .6)
    if edge == 'zz':
        ax.plot(xs, np.ones(len(xs))*np.sqrt(3)*bond, '-.', color = 'black')
        ax.plot(xs, np.ones(len(xs))*2*np.sqrt(3)*bond, '-.', color = 'black')
    elif edge == 'arm':
        ax.plot(xs, np.ones(len(xs))*1*bond, '-.', color = 'black')
        ax.plot(xs, np.ones(len(xs))*2*bond, '-.', color = 'black')
        ax.plot(xs, np.ones(len(xs))*3*bond, '-.', color = 'black')
    
        
    ax.set_ylim(line_limits[8] - 1, line_limits[9] + 1)
    ax.set_xlim(limits[0] - 1, limits[1] + 1)
    ax.set_title(title)
    # END CORRUGATION

def plot_shift_Y(ax, z_t, yav_t, iz, line_limits):
    
    for ai in range(len(yav_t[0])):
        ax.plot(z_t[:iz], yav_t[:iz,ai], color = 'black', label = r'layer %i' %ai)
    
    
    ax.set_xlim(line_limits[0], line_limits[1])
    ax.set_ylim(line_limits[6], line_limits[7])
    
    ax.set_ylabel(r'Shift y')
    ax.set_xlabel(r'bend z')

def plot_angle_epot(ax, ep, z_t, angles_av_t, iz, line_limits):

    ax.plot(z_t[:iz], angles_av_t[:iz], color = 'black', label=r'angle')
    ax2a =   plt.twinx(ax)
    ax2a.plot(z_t[:iz], ep[:iz], '-.', color = 'black', label = r'eP')
    
    ax.legend(loc = 2, frameon = False)
    ax2a.legend(loc = 4, frameon = False)

    ax.set_ylim(line_limits[4], line_limits[5])
    ax.set_xlim(line_limits[0], line_limits[1])
    ax2a.set_ylim(line_limits[2], line_limits[3])
    
    ax.set_ylabel(r'Angle deg')
    ax2a.set_ylabel(r'Pot E eV')

def plot_KC(ax1, N, positions, e_KC, layer_indices_f, angles, Rads, z0s, x0s, z, \
            limits, line_limits, shift_table, edif_min, edif_max, edge, bond, title = ''):
    
    
    layer_indices   =   layer_indices_f[:-2]

    n           =   len(layer_indices) 
    angles_av   =   0.
    
    xdata       =   positions[:,0]
    ydata       =   positions[:,2]
    zsets       =   np.zeros((1000, len(layer_indices)))
    xsets       =   np.zeros((1000, len(layer_indices)))

    
           
    for i in range(n):
        x_min   =   np.min(xdata[layer_indices[i]])
        x_max   =   np.max(xdata[layer_indices[i]])
        
        xsets[:,i]      =   np.linspace(x_min, x_max, 1000)
        
        R, phi, z0, x0  =   Rads[i], angles[i]/360.*2*np.pi, z0s[i], x0s[i] #par_set[j, i]
        
        zsets[:,i]      =   kink_func2(xsets[:,i], R, phi, z0, x0)
        
        angles_av  +=   angles[i]/n   
    
    
    smin        =   line_limits[8]
    
    if edge == 'arm':   
        smax_min    =   3*bond + 1
        ticks_shift =   [smin, 0., bond, 2*bond, 3*bond]
    elif edge == 'zz':  
        smax_min    =   np.sqrt(3)*bond*3 + 1
        ticks_shift =   [smin, 0., np.sqrt(3)*bond, 2*np.sqrt(3)*bond, 3*np.sqrt(3)*bond]
    
    smax        =   np.max([smax_min, line_limits[9]])
    s_range     =   smax - smin
    
    mid_p       =   np.abs(smin)/s_range
    
    c           =   mcolors.ColorConverter().to_rgb
    
    
    if edge == 'arm':
        b_r     =   bond/s_range
    elif edge == 'zz':
        b_r     =   np.sqrt(3)*bond/s_range
    
    cset        =   [mid_p, mid_p+b_r, mid_p+2*b_r, mid_p+3*b_r]
    
    shift_cmap  =   make_colormap([c('blue'), c('white'), cset[0], c('white'), c('green'), \
                                   cset[1], c('green'), c('red'), cset[2], c('red'), \
                                   c('violet'), cset[3], c('violet'), c('black')])
    
    #orig_map    =   mpl.cm.seismic
    #shift_cmap  =   shiftedColorMap(orig_map, start=0., midpoint=mid_p, stop=1., name='shrunk')    
    '''
    Z           =   [[0,0],[0,0]]
    levels      =   np.linspace(smin, smax, 1000) 
    
    
    cNorm       =   colors.Normalize(vmin=0, vmax=1)
    scalarMap   =   cmx.ScalarMappable(norm=cNorm, cmap=shift_cmap)
    shift_xs    =   shift_table #[:,2]
    
    
    
    
    for table in shift_xs:
        x0      =   [table[0], table[1]]
        y0      =   [table[2], table[3]]
        corr    =   table[4]
        
        val     =   (corr - smin)/s_range
        
        colorVal    =   scalarMap.to_rgba(val)
        ax1.plot(x0, y0, alpha = 1, lw = 1, color=colorVal)
    '''    
   
    # COLOR PLOT
    s0          =   ball_size
    #s           =   [s0 + e_KC[i]/edif_max*s0**np.sqrt(1.5) for i in range(len(xdata))]
    
    # Plot fits..
    #for i in range(n):
    #    ax1.plot(xsets[:,i], zsets[:,i], alpha = .3,  color = 'black')
        
    #
    x_av        =   (shift_table[:,0] + shift_table[:,1])/2.
    y_av        =   (shift_table[:,2] + shift_table[:,3])/2.
    corr        =   shift_table[:,4]
    
    
    axg         =   ax1.scatter(x_av, y_av, c = corr, s = s0, cmap = shift_cmap, \
                                vmin=smin, vmax = smax, edgecolors = 'none')
    #
    
    
    axf         =   ax1.scatter(xdata, ydata, c=e_KC, s=s0, cmap=mpl.cm.RdBu_r, \
                               vmin=edif_min, vmax=edif_max, edgecolors='none', zorder = 10000)
    
    
    
    
    ax1.set_ylim(limits[2], limits[3])
    ax1.set_aspect('equal') 
    divider3    =   make_axes_locatable(ax1)
    divider4    =   make_axes_locatable(ax1)
    
    cax3        =   divider3.append_axes("bottom", size="5%", pad=0.05)
    cax4        =   divider4.append_axes("right", size="5%", pad=0.5)
    
    cbar3       =   plt.colorbar(axf, cax=cax3, orientation = 'horizontal',\
                                  ticks = np.linspace(edif_min, edif_max, 5))
    
    #axf2        =   plt.contourf(Z, levels, cmap=shift_cmap)
    cbar4       =   plt.colorbar(axg, cax=cax4, \
                                 ticks = ticks_shift) #CS3, cax=cax4, norm=cNorm, orientation='vertical')
    cbar4.ax.set_yticklabels(['min', '0', '1. st', '2. nd', '3. th'])
    cbar3.ax.set_xticklabels(np.around(np.linspace(edif_min, edif_max, 5), 3))
    
    
    if title != '':
        ax1.set_title(title)
    
    y_scale =   limits[3] - limits[2]
    ax1.text(x_min, limits[2] + y_scale/10, r'angle = %.2f Deg' %angles_av, fontsize=12)
    ax1.text(x_min, limits[2] + y_scale/10*2, r'z = %.2f Angst' %z, fontsize=12)
    ax1.axis('off')


def plot_il(ax, N, positions, layer_indices_f, \
            limits, il_dist, title = ''):
    
    
    xdata       =   positions[:,0]
    ydata       =   positions[:,2]

    il_av           =   3.4
    il_min, il_max  =   np.max([limits[4], 3.2]), np.min([limits[5],3.6])
    
    
    xpoint  =   il_dist[:,0]
    ypoint  =   il_dist[:,1]
    ilh     =   il_dist[:,2]
    
    s0      =   ball_size
    ratio   =   (il_av - il_min)/(il_max - il_min)
    
    orig_map    =   mpl.cm.RdBu_r
    rvb = shiftedColorMap(orig_map, start=0., midpoint=ratio, stop=1., name='shrunk')    
    
    ax.scatter(xdata, ydata, s=s0, edgecolors='none', \
               color = 'black', alpha = .4, zorder = -1)
    
    ax2         =   ax.scatter(xpoint, ypoint, c=ilh, s=s0*1.1, cmap=rvb, \
                               vmin=il_min, vmax=il_max, edgecolors='none')
   
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect('equal') 
    
    divider3    =   make_axes_locatable(ax)
    
    cax3        =   divider3.append_axes("right", size="5%", pad=0.5)
    
    cbar3       =   plt.colorbar(ax2, cax=cax3, orientation = 'vertical',\
                                  ticks = np.linspace(il_min, il_max, 5))
    
    if title != '':
        ax.set_title(title)
    ax.axis('off')
    #################

def plot_streches(ax, streches, limits, title = ''):
    
    
    xdata       =   streches[:,0]
    ydata       =   streches[:,1]

    strech_av   =   np.zeros(len(streches))
    
    for i in range(len(streches)):
        n   =   0
        for k in range(3):
            if 1.2 < streches[i, 2 + k] < 1.8: 
                strech_av[i]  +=   streches[i, 2 + k]
                n   +=  1
        if n != 0:
            strech_av[i]    /= n   
            

    strech_avS  =   1.39695   #np.average(strech_av)
    strech_min, strech_max  =   strech_avS*0.995, strech_avS*1.005    

    
    #xpoint  =   il_dist[:,0]
    #ypoint  =   il_dist[:,1]
    #ilh     =   il_dist[:,2]
    
    s0      =   ball_size
    ratio   =   (strech_avS - strech_min)/(strech_max - strech_min)
    
    orig_map    =   mpl.cm.RdBu_r
    rvb         =   shiftedColorMap(orig_map, start=0., midpoint=ratio, stop=1., name='shrunk')    
    
    #ax.scatter(xdata, ydata, s=s0, edgecolors='none', \
    #           color = 'black', alpha = .4, zorder = -1)
    
    ax2         =   ax.scatter(xdata, ydata, c=strech_av, s=s0*1.5, cmap=rvb, \
                               vmin=strech_min, vmax=strech_max, edgecolors='none')
    
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect('equal') 
    
    divider3    =   make_axes_locatable(ax)
    
    cax3        =   divider3.append_axes("right", size="5%", pad=0.5)
    
    cbar3       =   plt.colorbar(ax2, cax=cax3, orientation = 'vertical',\
                                  ticks = np.linspace(strech_min, strech_max, 5))
    
    cbar3.ax.set_yticklabels(['-0.5%', '-0.25%', 'Gr', '+0.25%', '+0.5%'])
    if title != '':
        ax.set_title(title)
    ax.axis('off')
    #################
    

def plot_plotLogAtoms(positions, e_KC, layer_indices_f, angles, Rads, z0s, x0s, \
            limits, edif_min, edif_max, fig_size = (16,10)):
    
    _, ax1      =   plt.subplots(1, figsize = fig_size)
    layer_indices   =   layer_indices_f[:-2]

    n           =   len(layer_indices) 
    angles_av   =   0.
    xdata       =   positions[:,0]
    ydata       =   positions[:,2]
    zsets       =   np.zeros((1000, len(layer_indices)))
    xsets       =   np.zeros((1000, len(layer_indices)))
    
    for i in range(n):
        x_min   =   np.min(xdata[layer_indices[i]])
        x_max   =   np.max(xdata[layer_indices[i]])
        
        xsets[:,i]      =   np.linspace(x_min, x_max, 1000)
        
        R, phi, z0, x0  =   Rads[i], angles[i]/360.*2*np.pi, z0s[i], x0s[i] #par_set[j, i]
        
        zsets[:,i]      =   kink_func2(xsets[:,i], R, phi, z0, x0)
        
        angles_av  +=   angles[i]/n   
    
    
    s0      =   ball_size*5
    axf     =   ax1.scatter(xdata, ydata, c=e_KC, s=s0, cmap=mpl.cm.RdBu_r, \
                            vmin=edif_min, vmax=edif_max, edgecolors='none', zorder = -10000)

    for i in range(n):
        ax1.plot(xsets[:,i], zsets[:,i], alpha = 1.3,  color = 'black')
    
    #print xsets[:,i], zsets[:,i]
    
    ax1.set_ylim(limits[2], limits[3])
    ax1.set_aspect('equal') 
    divider3    =   make_axes_locatable(ax1)
    
    cax3        =   divider3.append_axes("right", size="5%", pad=0.05)
    
    cbar3       =   plt.colorbar(axf, cax=cax3, orientation = 'vertical',\
                                  ticks = np.linspace(edif_min, edif_max, 5))
    
    plt.show()

def plot_atoms(axs, traj_init, traj_c, angles_t, angles_av_t, Rads_t, \
               z0s_t, x0s_t, yav_t, ep, N, edge, z_t, bond, iz):
    
    
    ax1, ax2, ax3, ax4, ax5  =   axs
    layer_indices   =   find_layers(traj_init.positions)[1][:-2]
    positions       =   traj_c.positions   
    xdata       =   positions[:,0]
    ydata       =   positions[:,2]
    zsets       =   np.zeros((1000, len(layer_indices)))
    xsets       =   np.zeros((1000, len(layer_indices)))

    n           =   len(layer_indices) 
    angles_av   =   0.
    
    angles  =   angles_t[iz]
    Rads    =   Rads_t[iz]
    z0s     =   z0s_t[iz]
    x0s     =   x0s_t[iz]
    z       =   z_t[iz]
    
    #ax =   plt.gca()
    #plt.figure(1, figsize = (8,6))
    for i in range(n):
        x_min   =   np.min(xdata[layer_indices[i]])
        x_max   =   np.max(xdata[layer_indices[i]])
        
        xsets[:,i]      =   np.linspace(x_min, x_max, 1000)
        
        R, phi, z0, x0  =   Rads[i], angles[i]/360.*2*np.pi, z0s[i], x0s[i] #par_set[j, i]
        
        zsets[:,i]      =   kink_func2(xsets[:,i], R, phi, z0, x0)
        
        angles_av  +=   angles[i]/n   

    
    im  =   [ax1.scatter(xdata, ydata, alpha = 0.2, color = 'black')]
    for i in range(n):
        im.append(ax1.plot(xsets[:,i], zsets[:,i], color = 'black')[0])
    
    
    im.append(ax2.plot(z_t[:iz], angles_av_t[:iz], color = 'black')[0])
    im.append(ax5.plot(z_t[:iz], ep[:iz], '-.', color = 'black', label = 'eP')[0])
    
    im.append(ax2.text(np.max(z_t) - 15, 1.5*np.max(angles_av_t)/7., '- angle', fontsize=15))
    im.append(ax2.text(np.max(z_t) - 15, 2*np.max(angles_av_t)/7., '-. eP', fontsize=15))
    
    im.append(ax2.text(np.max(z_t) - 15, np.max(angles_av_t)/14., 'angle = %.2f Deg' %angles_av, fontsize=15))
    im.append(ax2.text(np.max(z_t) - 15, np.max(angles_av_t)/7., 'z = %.2f Angst' %z, fontsize=15))
    
    for ai in range(len(yav_t[0])):
        im.append(ax3.plot(z_t[:iz], yav_t[:iz,ai], color = 'black', label = 'layer %i' %ai)[0])
        
    for ai in range(len(Rads_t[0]) - 1):
        dif     =   Rads_t[:iz,ai  + 1]*angles_t[:iz,ai + 1]/360*2*np.pi \
                  - Rads_t[:iz,ai]*angles_t[:iz,ai]/360*2*np.pi   
        im.append(ax4.plot(z_t[:iz], dif, color = 'black', label = 'layer diff %i' %ai)[0])
        
        if edge == 'zz':
            im.append(ax4.plot(z_t, np.ones(len(z_t))*np.sqrt(3)*bond, '-.', color = 'black')[0])
            im.append(ax4.plot(z_t, np.ones(len(z_t))*2*np.sqrt(3)*bond, '-.', color = 'black')[0])

        elif edge == 'arm':
            im.append(ax4.plot(z_t, np.ones(len(z_t))*1*bond, '-.', color = 'black')[0])
            im.append(ax4.plot(z_t, np.ones(len(z_t))*2*bond, '-.', color = 'black')[0])
            im.append(ax4.plot(z_t, np.ones(len(z_t))*3*bond, '-.', color = 'black')[0])
        
    
        
        
    
    
    
    
    #
        
    #im     =   ax.scatter(xdata, ydata, alpha = 0.2, color = 'black')
    #ims.append(im)
        
    
    ax1.axis('equal') 
    
    '''
    path_bendF  =   '/space/tohekorh/BendAndSlide/pictures/bend/N=%i/%s/' %(N, edge)   
    
    if os.path.exists(path_bendF):
        plt.savefig(path_bendF + r'fig_h=%i.pdf' %iz)
    else:
        os.makedirs(path_bendF)
        plt.savefig(path_bendF + r'fig_h=%i.pdf' %iz)
    '''
    
    return im
        
def get_angle_set(positions_t, therm_idx, indent = 'fixTop', round_kink = True):
    
    
            
    if indent == 'fixTop':
        #layer_indices   =   find_layers(traj[0].positions)[1][:-2]
        layer_indices   =   find_layers(positions_t[0])[1][:-2]
    
    n               =   len(layer_indices)
    len_t           =   len(positions_t)
    par_set         =   np.empty((len_t, n), dtype = 'object')

    #M               =   int(len(traj)/4)
    angles          =   np.zeros((len_t, len(layer_indices)))
    Rads            =   np.zeros((len_t, len(layer_indices)))
    z0s             =   np.zeros((len_t, len(layer_indices)))
    x0s             =   np.zeros((len_t, len(layer_indices)))
    
    yav             =   np.zeros((len_t, len(layer_indices)))
    angles_av       =   np.zeros(len(angles))
    
    dg              =   [.01,.01,.01,.01,10]
    
    
   
        
    for j in range(len_t):
    
        #positions   =   traj[j].positions
        positions   =   positions_t[j]

        print j, len_t
        for i, lay_inds in enumerate(layer_indices):
            xdata    =   positions[lay_inds][:,0]
            ydata    =   positions[lay_inds][:,1]
            zdata    =   positions[lay_inds][:,2]
            
            if j > 0:
                yav[j, i]   =   np.average(ydata) - yav[0, i]
            elif j == 0:
                yav[j, i]   =   np.average(ydata)
            
            left_ind    =   np.where(xdata == np.min(xdata))[0][0]
            
            kink_ids    =   np.where((zdata - zdata[left_ind]) < -.3)[0]
            
            if len(kink_ids) == 0:
                kink    =   np.min(xdata) + (np.max(xdata) - np.min(xdata))*9./10.
            else:
                kink_id =   np.where(xdata == np.min(xdata[kink_ids]))
                kink    =   xdata[kink_id[0]]
            
            if not round_kink:
                ques        =   [0., zdata[left_ind], 0., kink]
                
                par_set[j,i]    =   curve_fit(kink_func, xdata, zdata, p0=ques, \
                                          factor = 1., diag=dg, epsfcn = .00001)[0]
            
                angles[j][i]   +=   -np.arctan(par_set[j,i][0])/(2*np.pi)*360/n             
            
            elif round_kink:
                
                hz              =   zdata[left_ind]
                Rmax            =   (max(xdata) - kink)*1.2
                
                
                if  j < therm_idx + len_t/40:
                    ques2       =   np.array([Rmax/2., 0., hz, kink])
                else:
                    ques2       =   np.array([Rads[j - 1, i], angles[j - 1, i]/360*2*np.pi, \
                                              z0s[j - 1, i], x0s[j - 1, i]])   
                
                                #ques2       =   np.array([Rmax/2., 0., hz, kink])
                
                bounds          =   [(0, Rmax), (0., np.pi/2.1), (hz-.3, hz+.3), (0., None)]
                par_set[j,i]    =   fmin_l_bfgs_b(kink_func3, ques2, args=(xdata, zdata), \
                                                  approx_grad = True, bounds = bounds)[0]
                
                
                Rads[j,i]       =   par_set[j,i][0]
                angles[j,i]     =   par_set[j,i][1]/(2*np.pi)*360
                z0s[j,i]        =   par_set[j,i][2]
                x0s[j,i]        =   par_set[j,i][3]
                
        for angle in angles[j]:
            angles_av[j]   +=   angle/len(angles[j])   
        
    plot_data   =   np.concatenate((angles, Rads, x0s, z0s, yav), axis=1)
        
    return angles_av, plot_data 
            