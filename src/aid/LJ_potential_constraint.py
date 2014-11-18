import numpy as np
from help2 import map_seq, map_rj
from aid.help import find_layers


class add_adhesion:
    
    def __init__(self, params):
        
        bond    =   params['bond']
        posits  =   params['positions']
        a       =   np.sqrt(3)*bond # 2.462
           
        self.n                  =   (a**2*np.sqrt(3)/4)**(-1)   
        self.ecc, self.sigcc    =   0.002843732471143, 3.4
        layers, self.layer_inds =   find_layers(posits)
        top_h                   =   layers[-1]
        self.h                  =   params['h'] + top_h
        
        
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

class LJ_potential:
    
    def __init__(self, params):
        
        posits  =   params['positions']
        
        self.pbc     =   params['pbc']
        self.cell    =   params['cell']
        self.cutoff  =   params['ia_dist']
        
        self.n       =   0
        
        for i in range(3):
            if self.pbc[i]: self.n += 1
                   
        self.ecc, self.sigcc    =   0.002843732471143, 3.4
        
        self.layer_indices      =   find_layers(posits)[1]
         
    def adjust_positions(self, oldpositions, newpositions):
        pass
        
    def adjust_forces(self, positions, forces):
        pass

    

    def adjust_potential_energy(self, positions, energy):
        
        
        e_LJ           =   0
        
        def E(r):
            return 4*self.ecc*((self.sigcc/r)**12 - (self.sigcc/r)**6)
        
        for i, layer_ind in enumerate(self.layer_indices[:-1]):
            layer_ind_top   =   self.layer_indices[i + 1]
            for ind_bot in layer_ind:
                ri          =   positions[ind_bot]
                for ind_top in layer_ind_top:
                    rj          =   positions[ind_top]
                    m           =   0
                    out         =   False
                    while not out:
                        out     =   True
                        rj_im   =   map_rj(rj, map_seq(m, self.n), self.pbc, self.cell)
                        for mrj in rj_im:
                            r   =   np.linalg.norm(mrj - ri)
                            if r < self.cutoff:
                                e_LJ    +=  E(r)
                                if m > 0:
                                    out =  False
                            if m == 0: out = False
                                
                        m      +=   1
        
        return e_LJ
                    
        
           
        