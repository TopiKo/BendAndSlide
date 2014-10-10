'''
Created on 8.10.2014

@author: tohekorh
'''
import numpy as np

class add_adhesion:
    
    def __init__(self, params, top_h):
        
        bond    =   params['bond']
        a       =   np.sqrt(3)*bond # 2.462
           
        self.n                  =   (a**2*np.sqrt(3)/4)**(-1)   
        self.ecc, self.sigcc    =   0.002843732471143, 3.4
        self.h          =   params['h'] + top_h
        
    def adjust_positions(self, oldpositions, newpositions):
        pass
        
    def adjust_forces(self, positions, forces):
        
        def get_force_rebo(h):
            # Integrate the rebo potential to yield E_adh/atom, see verification in 
            # calc_adhesion. Derivative of that is dE_adh/dh = -force/atom  
            return -8.*self.ecc*self.n*np.pi*(self.sigcc**12/h**11 - self.sigcc**6/h**5)

        
        for i, pos in enumerate(positions):
            z               =   pos[2]
            h               =   self.h - z
            f0              =   forces[i]
            forces[i]       =   [f0[0], f0[1], f0[2] + get_force_rebo(h)]
            
            


