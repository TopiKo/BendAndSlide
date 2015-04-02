'''
Created on 25.3.2015

@author: tohekorh
'''

from aid.KC_parallel import KC_potential_p
import numpy as np

def get_KC(traj): 
    
    
    atoms               =   traj[0]
    
    params              =   {}
    params['ncores']    =   2
    params['positions'] =   atoms.positions.copy() 
    params['pbc']       =   atoms.get_pbc()
    params['cell']      =   atoms.get_cell().diagonal()
    params['ia_dist']   =   10
    params['chemical_symbols']  =   atoms.get_chemical_symbols()
    
    add_kc      =   KC_potential_p(params)
    e_KC        =   np.empty(len(traj), dtype = 'object')


    for i in range(len(traj)):
        print i, len(traj)
        atoms   =   traj[i]
        e_KC[i] =   np.zeros(len(atoms.positions))    
        e_KC[i] =   add_kc.energy_i(atoms.positions)/2.
        
    return e_KC

# Test for corrugation and for get_KC.
'''
from ase.io.trajectory import PickleTrajectory
from ase.visualize import view 
import matplotlib.pyplot as plt
import matplotlib as mpl
import os


path        =   '/space/tohekorh/BendAndSlide/test_KC/'
path_corr   =   path + 'trajectories/corrugation_trajectory_rebo_KC_iaS_p_arm.traj'

traj        =   PickleTrajectory(path_corr, 'r')
view(traj[0])
e_KC    =    get_KC(traj)

xmin    =   np.min(traj[0].positions[:,0])
xmax    =   np.max(traj[-1].positions[:,0])

print xmin, xmax
emin, emax  =   1000, -1000

#ne_KC   =   np.empty(len(e_KC), dtype = 'object')
#for i, e in enumerate(e_KC):
#    ne_KC[i]   =   e - e_KC[0]
#e_KC    =   ne_KC

for e in e_KC:
    
    emin_test   =   np.min(e)
    emax_test   =   np.max(e)
    if emin_test < emin:
        emin = emin_test
    if emax < emax_test:
        emax = emax_test    

for i, e in enumerate(e_KC):
    x,y =   traj[i].positions[:,0] ,traj[i].positions[:,2]
    ax  =   plt.scatter(x,y, s=100, c=e, cmap=mpl.cm.RdBu_r, vmin=emin, vmax=emax)
    ax  =   plt.scatter(x + 3*1.42,y, s=100, c=e, cmap=mpl.cm.RdBu_r, vmin=emin, vmax=emax)
    
    plt.ylim([3, 10])
    plt.xlim([xmin - 1, xmax + 1])
    cbar3       =   plt.colorbar(ax, orientation = 'horizontal',\
                                  ticks = np.linspace(emin, emax, 5))
    plt.savefig(path + 'pic_%04d' %i, dpi = 100)
    plt.clf()
    
os.system('mencoder "mf://%spic*.png" -mf type=png:fps=10  \
            -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o %svideo_corr.mpg' %(path, path)) 
os.system('rm -f %spic*.png' %path) 
''' 