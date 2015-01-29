'''
Created on 26.1.2015

@author: tohekorh
'''

import matplotlib.pyplot as plt
import numpy as np
from aid.help import get_fileName

Ns  =   [5]


plt.figure(1, figsize=(6,8))
    

for N in Ns:
    mdLogFile   =   get_fileName(N, 'tear_E_rebo+KC')[1]
    ef          =   np.loadtxt(mdLogFile)
    
    z           =   ef[:,1] 
    et          =   ef[:,4]
    ek          =   ef[:,3]
    ep          =   ef[:,2]
    
    print ep[::100]
    print ek[::100]
    
    plt.subplot(211)
    plt.title('Energies ev/atom')
    plt.plot(z, et, label = 'etot')
    plt.plot(z, ek, label = 'ekin') 
    plt.plot(z, ep, label = 'epot')
    plt.ylabel('E ev/atom')
#    plt.xlabel('Bend heigh Angstrom')
    
    plt.legend()
    
    plt.subplot(212)
    plt.title('Potential energy ev/atom')
    plt.plot(z, ep, label = 'epot')
    plt.xlabel('Bend heigh Angstrom')
    plt.ylabel('E ev/atom')
    plt.legend(loc = 4)
    
    #plt.subplot(313)
    

plt.show()