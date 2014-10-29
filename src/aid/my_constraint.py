'''
Created on 8.10.2014

@author: tohekorh
'''
import numpy as np
from ase import Atoms
from ase.visualize import view
from help2 import extend_structure, nrst_neigh
from aid.help import find_layers

class add_adhesion:
    
    def __init__(self, params, layer_indices, top_h):
        
        bond    =   params['bond']
        a       =   np.sqrt(3)*bond # 2.462
           
        self.n                  =   (a**2*np.sqrt(3)/4)**(-1)   
        self.ecc, self.sigcc    =   0.002843732471143, 3.4
        self.h          =   params['h'] + top_h
        self.layer_inds  =  layer_indices
        
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
        self.ia_dist =   12
           
        self.ecc, self.sigcc    =   0.002843732471143, 3.4
        
        posits_ext              =   extend_structure(self.ia_dist, posits, self.pbc, self.cell)
        self.layer_neighbors    =   nrst_neigh(posits, posits_ext, 'layer')    
        self.layer_indices      =   find_layers(posits_ext)[1]
         
    def adjust_positions(self, oldpositions, newpositions):
        pass
        
    def adjust_forces(self, positions, forces):
        pass

    

    def adjust_potential_energy(self, positions, energy):
        
        posits_ext  =   extend_structure(self.ia_dist, positions.copy(), self.pbc, self.cell)
        self.interact_neighbors =   nrst_neigh(positions, posits_ext, 'interact', self.ia_dist, self.layer_indices)   
        e           =   0
        
        for i, ra in enumerate(positions):
            for j in self.interact_neighbors[i]:
                r_ia    =   posits_ext[j]
                r       =   np.linalg.norm(ra - r_ia)
                e      +=   4*self.ecc*((self.sigcc/r)**12 - (self.sigcc/r)**6)/2. # note factor .5 from
                # the fact that we are double counting each inteaction..
        
        #print e       
        return e
        
           
        
        
            
class KC_potential:
    
    # This is a constraint class to include the registry dependent 
    # interlayer potential to ase. 
    
    # NOT ABLE TO DO VERY INTENCE BENDING!! The issue related to the direction of the
    # surface normal is unsolved 
    
    def __init__(self, params):
        
        posits      =   params['positions']
        
        self.pbc    =   params['pbc']
        self.cell   =   params['cell']
        self.ia_dist=   params['ia_dist']
        # parameters as reported in KC original paper
        self.delta  =   0.578       # Angstrom   
        self.C0     =   15.71*1e-3  # [meV]
        self.C2     =   12.29*1e-3  # [meV]
        self.C4     =   4.933*1e-3  # [meV]
        self.C      =   3.030*1e-3  # [meV]
        self.lamna  =   3.629       # 1/Angstrom
        self.z0     =   3.34        # Angstrom
        self.A      =   10.238*1e-3 # [meV]     
           
        
        posits_ext              =   extend_structure(self.ia_dist, posits, self.pbc, self.cell)
        self.layer_neighbors    =   nrst_neigh(posits, posits_ext, 'layer')    
        self.layer_indices      =   find_layers(posits_ext)[1]
        #self.interact_neighbors =   nrst_neigh(posits, posits_ext, 'interact', self.ia_dist)    
        
         
    def adjust_positions(self, oldpositions, newpositions):
        pass
        
    def adjust_forces(self, positions, forces):
        pass

    

    def adjust_potential_energy(self, positions, energy):
        
        posits_ext      =   extend_structure(self.ia_dist, positions.copy(), self.pbc, self.cell)
        self.interact_neighbors     \
                        =   nrst_neigh(positions, posits_ext, 'interact', self.ia_dist, self.layer_indices)
        
        def local_normal(i, posits_ext):
            
            ri          =   posits_ext[i]
            
            tang_vec    =   np.zeros((len(self.layer_neighbors[i]), 3))
            
            for k, j in enumerate(self.layer_neighbors[i]):
                tang_vec[k]  =     posits_ext[j] - ri 
            
            if len(tang_vec) == 3:
                normal  =   np.cross(tang_vec[0], tang_vec[1])/np.linalg.norm(np.cross(tang_vec[0], tang_vec[1])) \
                        +   np.cross(tang_vec[2], tang_vec[0])/np.linalg.norm(np.cross(tang_vec[2], tang_vec[0])) \
                        +   np.cross(tang_vec[1], tang_vec[2])/np.linalg.norm(np.cross(tang_vec[1], tang_vec[2])) 
                
                normal  =   normal/3
                
                for l in self.layer_neighbors[i]:
                    if np.abs(np.linalg.norm(posits_ext[l] - posits_ext[i]) - 1.39695) > 0.1: print 'huu'
                
                return normal 
            
            else:
                raise
            
        def f(p):
            return np.exp(-1*(p/self.delta)**2)*(self.C0                    \
                                              +  self.C2*(p/self.delta)**2  \
                                              +  self.C4*(p/self.delta)**4)
        
        def V(pij, pji, rij):
            
            r   =   np.linalg.norm(rij)
            pij =   r**2 - np.dot(rij, ni)**2
            pji =   r**2 - np.dot(rij, nj)**2
            return np.exp(-self.lamna*(r - self.z0))*(self.C + f(pij) + f(pji)) - self.A*(r/self.z0)**(-6.)
            
        e                   =   0
        
        for i, ra in enumerate(positions):
            for j in self.interact_neighbors[i]:
                
                ni  =   local_normal(i, posits_ext)
                nj  =   local_normal(j, posits_ext)
                rij =   posits_ext[j] - ra
                
                e  +=   V(ni, nj, rij)/2 # divide by two due to double counting
    
        return e




