'''
Created on 11.3.2015

@author: tohekorh
'''

import numpy as np
import matplotlib.pyplot as plt
from ase.structure import graphene_nanoribbon
from ase.visualize import view 
from scipy.optimize import fmin  
M       =   5e2

delta  =    0.578       # Angstrom   
C0     =    15.71*1e-3  # [meV]
C2     =    12.29*1e-3  # [meV]
C4     =    4.933*1e-3  # [meV]
C      =    3.030*1e-3  # [meV]
lamna  =    3.629       # 1/Angstrom
z0     =    3.34        # Angstrom
A      =    10.238*1e-3 # [meV]     

bond   =    1.39695 

AperAtom    =   np.sqrt(3)/4.*bond**2


atoms       =   graphene_nanoribbon(20, 20, type='zigzag', C_H=1.09, C_C=bond)
center      =   (atoms.positions[380] + atoms.positions[418])/2.
Rs          =   atoms.positions - center

def f(p):
    
    return np.exp(-(p/delta)**2)*(C0 + C2*(p/delta)**2 + C4*(p/delta)**4)

def Vh(h, *args):
    
    tilt_angle   =   args[0]/360*2*np.pi
    V   = 0.
    for plane_pos in Rs:
        R   =   np.linalg.norm(plane_pos)
        if np.sqrt(R**2 + h**2) < 10: 
            r   =   np.sqrt(R**2 + h**2)
            gamma   =   np.arctan(R/h)     
            
            pij =   R
            pji =   np.sqrt(r**2*(1 - np.cos(gamma - tilt_angle)**2))
            V  +=   np.exp(-lamna*(r - z0))*(C + f(pij) + f(pji)) - A*(r/z0)**(-6)
    return V


def get_e_adh(angle):
    return 0.044
    #return Vh(13.4, angle) - Vh(3.4, 0)
    
#print np.arctan(np.sqrt(3)/2.*bond/3.4)/(2*np.pi)*360   
'''
hs      =   np.linspace(3, 10, M)
angles  =   np.linspace(0,60, 1)

for angle in angles:
    V   =   np.zeros(M)
    for i, h in enumerate(hs):
        V[i]       +=   Vh(h, angle)

    hmin    =   fmin(Vh, [3.4], args=[angle])
    emin    =   Vh(hmin, angle) 
    print hmin ,emin
    plt.scatter(hmin, emin)
    plt.plot(hs, V, label = angle)

plt.legend(loc=4)


plt.show()
'''
