'''
Created on 26.1.2015

@author: tohekorh
'''

import matplotlib.pyplot as plt
import numpy as np
from aid.help import get_fileName
from find_angles import get_angle_set
from ase.io.trajectory import PickleTrajectory
Ns  =   [7] #,6,7,8,9,10,11,12,13]
v    =  1.0

    

for N in Ns:
    plt.figure(1, figsize=(7,8))
    
    mdfile, mdLogFile   =   get_fileName(N, 'tear_E_rebo+KC_v', v)[:2]
    
    traj        =   PickleTrajectory(mdfile, 'r')
    ef          =   np.loadtxt(mdLogFile)
    fac         =   len(traj[0])
    
    if 1e3 < len(ef):
        x       =   max(int(len(ef)/1e3), 2)
        m       =   int(len(ef)/x)
        efn     =   np.zeros((m, len(ef[0])))
        ntraj   =   []
        for i in range(m):
            ntraj.append(traj[i*x]) 
            efn[i,:]    =   ef[i*x] 
        
        ef      =   efn
        traj    =   ntraj
        #ef      =   ef[::x]
    
    t           =   ef[:,0] 
    z           =   ef[:,1] 
    et          =   ef[:,4] - ef[0,4]
    ek          =   ef[:,3]
    ep          =   ef[:,2] - ef[0,2]
    #n=  len(traj[0])
    
    
    angles      =   get_angle_set(traj, plot = True)
    print N
    #print ep[::100]
    #print ek[::100]
    
    plt.subplot(211)
    plt.title('Energies, N=%i' %N)
    plt.plot(z, et*fac, '-', linewidth = 1, label = 'etot', color = 'black')
    plt.plot(z, ek*fac, '<', linewidth = 1, label = 'ekin', color = 'black') 
    plt.plot(z, ep*fac, '--', linewidth = 1, label = 'epot', color = 'black')
    plt.ylabel('E ev')
#    plt.xlabel('Bend heigh Angstrom')
    
    plt.legend(loc = 2, frameon = False)
    
    plt.subplot(212)
    plt.title('Potential energy ev')
    
    plt.plot(z, ep*fac, '--', linewidth = 1, label = 'epot', color = 'black')
    plt.ylabel('E_p ev')
    plt.legend(loc = 3, frameon = False)
    
    
    plt.twinx()
    plt.plot(z, angles, '-.', linewidth = 1, label = 'angle', color = 'black')
    plt.ylabel('Kink angle Deg')
    
    plt.xlabel('Bend heigh Angstrom')
    plt.legend(loc = 2, frameon = False)
    
    plt.savefig('/space/tohekorh/BendAndSlide/fig_N=%i.svg' %N)
    #plt.subplot(313)
    plt.show()
    plt.clf()

    