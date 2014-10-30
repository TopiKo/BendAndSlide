'''
Created on 8.10.2014

@author: tohekorh
'''
import numpy as np
from ase import Atoms
from ase.visualize import view
from help2 import extend_structure, nrst_neigh, map_seq, map_rj
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
                    
        
           
        
        
            
class KC_potential:
    
    # This is a constraint class to include the registry dependent 
    # interlayer potential to ase. 
    
    # NOT ABLE TO DO VERY INTENCE BENDING!! The issue related to the direction of the
    # surface normal is unsolved 
    
    def __init__(self, params):
        
        posits      =   params['positions']
        
        self.pbc    =   params['pbc']
        self.cell   =   params['cell']
        self.cutoff =   params['ia_dist']
        # parameters as reported in KC original paper
        self.delta  =   0.578       # Angstrom   
        self.C0     =   15.71*1e-3  # [meV]
        self.C2     =   12.29*1e-3  # [meV]
        self.C4     =   4.933*1e-3  # [meV]
        self.C      =   3.030*1e-3  # [meV]
        self.lamna  =   3.629       # 1/Angstrom
        self.z0     =   3.34        # Angstrom
        self.A      =   10.238*1e-3 # [meV]     
        
        self.n       =   0
        
        for i in range(3):
            if self.pbc[i]: self.n += 1   
        
        posits_ext              =   extend_structure(0., posits, self.pbc, self.cell)
        #print 'C'*len(posits_ext)
        #view_atoms = Atoms('C'*len(posits_ext), positions = posits_ext)
        #iew(view_atoms)
        self.layer_neighbors    =   nrst_neigh(posits, posits_ext, 'layer')    
        self.layer_indices      =   find_layers(posits)[1]
        #self.interact_neighbors =   nrst_neigh(posits, posits_ext, 'interact', self.ia_dist)    
        
         
    def adjust_positions(self, oldpositions, newpositions):
        pass
        
    def adjust_forces(self, positions, forces):
        
        dx,dy,dz    =   1e-4, 1e-4, 1e-4
        e           =   self.adjust_potential_energy(positions, 0)
            
        for i in range(len(positions)):
            positions_dx        =   positions.copy()
            positions_dx[i,0]   =   positions[i,0] + dx   
            positions_dy        =   positions.copy()
            positions_dy[i,1]   =   positions[i,1] + dy   
            positions_dz        =   positions.copy()
            positions_dz[i,2]   =   positions[i,2] + dz   
            edx = self.adjust_potential_energy(positions_dx, 0)
            edy = self.adjust_potential_energy(positions_dy, 0)
            edz = self.adjust_potential_energy(positions_dz, 0)
            
            forces[i,0] += -(edx - e)/dx
            forces[i,1] += -(edy - e)/dy
            forces[i,2] += -(edz - e)/dz
        
        
        

    def adjust_potential_energy(self, positions, energy):
        
        
        posits_ext      =   extend_structure(0., positions.copy(), self.pbc, self.cell)
        e_KC    =   0.

        
        def local_normal(i):
            
            ri          =   posits_ext[i]
            
            tang_vec    =   np.zeros((len(self.layer_neighbors[i]), 3))
            
            for k, j in enumerate(self.layer_neighbors[i]):
                tang_vec[k]  =     posits_ext[j] - ri 
            
            if len(tang_vec) == 3:
                normal  =   np.cross(tang_vec[0], tang_vec[1])/np.linalg.norm(np.cross(tang_vec[0], tang_vec[1])) \
                        +   np.cross(tang_vec[2], tang_vec[0])/np.linalg.norm(np.cross(tang_vec[2], tang_vec[0])) \
                        +   np.cross(tang_vec[1], tang_vec[2])/np.linalg.norm(np.cross(tang_vec[1], tang_vec[2])) 
                
                normal  =   normal/3
                #normal  =   normal/np.linalg.norm(normal)
                
                # tests!
                for l in self.layer_neighbors[i]:
                    if np.abs(np.linalg.norm(posits_ext[l] - posits_ext[i]) - 1.39695) > 0.1: print 'huu'
                if np.abs(np.linalg.norm(normal) - 1.) > 1e-8:
                    raise
                
                return normal 
            
            else:
                raise
        
        
        def f(p):
            return np.exp(-(p/self.delta)**2)*(self.C0                    \
                                              +  self.C2*(p/self.delta)**2  \
                                              +  self.C4*(p/self.delta)**4)
        
        def E_KC(rij, ni, nj):
            
            r   =   np.linalg.norm(rij)
            pij =   np.sqrt(r**2 - np.dot(rij, ni)**2)
            pji =   np.sqrt(r**2 - np.dot(rij, nj)**2)
            return np.exp(-self.lamna*(r - self.z0))*(self.C + f(pij) + f(pji)) - self.A*(r/self.z0)**(-6.)
            
        
        for i, layer_ind in enumerate(self.layer_indices[:-1]):
            layer_ind_top   =   self.layer_indices[i + 1]
            for ind_bot in layer_ind:
                ri          =   positions[ind_bot]
                for ind_top in layer_ind_top:
                    rj          =   positions[ind_top]
                    m           =   0
                    counting    =   True
                    while counting:
                        counting=   False
                        rj_im   =   map_rj(rj, map_seq(m, self.n), self.pbc, self.cell)
                        for mrj in rj_im:
                            rij =   mrj - ri
                            r   =   np.linalg.norm(mrj - ri)
                            if r < self.cutoff:
                                ni       = local_normal(ind_bot)
                                nj       = local_normal(ind_top)
                                e_KC    +=  E_KC(rij, ni, nj)
                                counting =  True
                                
                        m      +=   1
        
        return e_KC