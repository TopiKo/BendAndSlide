'''
Created on 12.2.2015

@author: tohekorh
'''
from ase.io.trajectory import PickleTrajectory
import numpy as np
from ase.units import *
import matplotlib.pyplot as plt


path_taito_ase_svn    =   '/space/tohekorh/BendAndSlide/files/test_lammps/taito_ase=svn/'
path_taito_ase_snp    =   '/space/tohekorh/BendAndSlide/files/test_lammps/taito_ase=snapshot/'
path_local_ase_svn    =   '/space/tohekorh/BendAndSlide/files/test_lammps/local_ase=svn/'
path_taito_ase_381    =   '/space/tohekorh/BendAndSlide/files/test_lammps/taito_ase=381/'
path_local_ase_old    =   '/space/tohekorh/BendAndSlide/files/test_lammps/local_ase=old/'

paths   =   [[path_local_ase_old, 'local_ase_old'], [path_local_ase_svn, 'local_ase_svn'], \
             [path_taito_ase_381, 'taito_ase_381'], [path_taito_ase_svn, 'taito_ase_svn'], \
             [path_taito_ase_snp, 'taito_ase_snp']]


f, [ax1, ax2] = plt.subplots(2)

ax1.set_title('Kinetic energies')
ax2.set_title('Potential energies')

for i, path in enumerate(paths):
    
    data    =   np.loadtxt(path[0] + 'md_KC.log')
    
    z   =   data[:,1]
    ep  =   data[:,2]
    ek  =   data[:,3]
    
    ax1.plot(z, ek + i*.001, lw = 2./(i+1), label = path[1])
    
    ax2.plot(z, ep + i*.001, lw = 2./(i+1), label = path[1])

ax1.legend(frameon = False)
ax2.legend(frameon = False)


plt.show()