'''
Created on 11.11.2014

@author: tohekorh
'''

import numpy as np
from help2 import extend_structure, nrst_neigh, map_seq, map_rj, local_normal, which_layer
from aid.help import find_layers
import scipy


        
            
class KC_potential:
    
    # This is a constraint class to include the registry dependent 
    # interlayer potential to ase. 
    
    # NOT ABLE TO DO VERY INTENCE BENDING!! The issue related to the direction of the
    # surface normal is unsolved 
    
    def __init__(self, params):
        
        posits      =   params['positions']
        
        self.chem_symbs =   params['chemical_symbols']
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
        
        self.n          =   0
        self.map_seqs   =  []
        for i in range(3):
            if self.pbc[i]: self.n += 1   
        
        posits_ext              =   extend_structure(0., posits, self.pbc, self.cell)
        self.layer_neighbors    =   nrst_neigh(posits, posits_ext, 'layer')    
        self.layer_indices      =   find_layers(posits)[1]
        
         
    def adjust_positions(self, oldpositions, newpositions):
        pass
    
    
    def adjust_forces(self, posits, forces):
        
        calcet =    np.zeros((len(posits), len(posits)))
        
        for i in range(len(forces)):
            if self.chem_symbs[i] == 'C':
                for j in self.get_neigh_layer_indices(i, only_C = True):
                    if calcet[i,j] == 0:
                        #print i, j
                        # Force due to atom j on atom i
                        fij         =   self.get_forces_ij(posits, i, j, test = False)
                        forces[i,:] +=  fij  
                        # Force due to i on j is minus force due to j on i
                        forces[j,:] += -fij  
                        calcet[i,j] = 1
                        calcet[j,i] = 1
                        

        
    def adjust_potential_energy(self, posits, energy):
        
        calcet          =   np.zeros((len(posits), len(posits)))
        e_KC            =   0.
        
        for i in range(len(posits)):
            if self.chem_symbs[i] == 'C':
                for j in self.get_neigh_layer_indices(i, only_C = True):
                    if calcet[i,j] == 0:
                        e_KC += self.get_potential_ij(posits, i, j) 
                        calcet[i,j] = 1
                        calcet[j,i] = 1
        return e_KC
     

    def get_neigh_layer_indices(self, i, only_C = True):
        
        lay_inds    =   self.layer_indices    
        layer       =   which_layer(i, self.layer_indices)[0]
        ret_lay_ind =   []
        
        if layer > 0:
            for j in lay_inds[layer - 1]:
                if self.chem_symbs[j] == 'C':
                    ret_lay_ind.append(j)
        if layer < len(lay_inds) - 1:
            for j in lay_inds[layer + 1]:
                if self.chem_symbs[j] == 'C':
                    ret_lay_ind.append(j)
        return ret_lay_ind    

        
    def grad_pot_ij(self,  posits, i, j):                                
        
        def e(ri, *args):
            k               =   args[0]
            posits_use      =   posits.copy()   
            posits_use[i,k] =   ri
            
            return self.get_potential_ij(posits_use, i, j)
                 
        
        de = np.zeros(3)
        for k in range(3):
            de[k]  =  scipy.misc.derivative(e, posits[i,k], dx=1e-6, n=1, args=[k], order = 3)    
        
        return de
        
             
    def f(self, p):
        return np.exp(-(p/self.delta)**2)*(self.C0 \
                     +  self.C2*(p/self.delta)**2  \
                     +  self.C4*(p/self.delta)**4)
    
    def get_forces_ij(self, positions, i, j, test = False):
        
        posits_ext      =   extend_structure(0., positions.copy(), self.pbc, self.cell)
        
        def gradN(ri, ind_bot):
            # Matrix
            #dn1/dx dn2/dx dn3/dx
            #dn1/dy dn2/dy dn3/dy
            #dn1/dz dn2/dz dn3/dz

            def nj(xi, *args):
                i, j                =   args[0], args[1]
                
                posits_use          =   positions.copy()
                posits_use[ind_bot, i] =  xi
                posits_ext_use      =   extend_structure(0., posits_use.copy(), self.pbc, self.cell)
                
                nj   =   local_normal(ind_bot, posits_ext_use, self.layer_neighbors)[j]
                return nj

            dni = np.zeros((3,3))
            for i in range(3):
                for j in range(3):
                    dni[i,j]   =  scipy.misc.derivative(nj, ri[i], dx=1e-3, n=1, args=[i, j], order = 3)
            
            return dni      
       
        
        def GradV_ij(rij, r, pij, pji, ni, nj, dni):
            
            # See notes    
            def g(p):
                return -2./self.delta**2*self.f(p) + np.exp(-(p/self.delta)**2)* \
                       (2.*self.C2/self.delta**2 + 4*self.C4*p**2/self.delta**4)
            
            GV      =   np.zeros(3)
            for k in range(3):
                GV[k] = (-self.lamna*np.exp(-self.lamna*(r - self.z0))*(self.C + self.f(pij) + self.f(pji)) \
                         + 6*self.A*self.z0**6/r**7)*(-rij[k]/r) \
                         + np.exp(-self.lamna*(r - self.z0))*g(pji)*(np.dot(nj, rij)*(nj[k]) - rij[k]) \
                         + np.exp(-self.lamna*(r - self.z0))*g(pij)*(np.dot(ni, rij)*(ni[k]  - np.dot(rij, dni[k])) - rij[k])
            
            return GV
        
        F               =   np.zeros(3)
        
        ri              =   positions[i]
        rj              =   positions[j]

        ni              =   local_normal(i, posits_ext, self.layer_neighbors)
        nj              =   local_normal(j, posits_ext, self.layer_neighbors)
        
        dni             =   gradN(ri, i)
        #dnj == 0
        
        m               =   0
        counting        =   True
        
        # sum over all images of j that are within cutoff
        while counting:
            counting=   False
            
            if len(self.map_seqs) < m + 1:
                self.map_seqs.append(map_seq(m, self.n))
            
            rj_im   =   map_rj(rj, self.map_seqs[m], self.pbc, self.cell)
            
            #rj_im   =   map_rj(rj, map_seq(m, self.n), self.pbc, self.cell)
            
            for mrj in rj_im:
                rij =   mrj - ri
                r   =   np.linalg.norm(mrj - ri)
                if r < self.cutoff:
                    
                    pij  =   np.sqrt(np.linalg.norm(rij)**2 - np.dot(ni, rij)**2)
                    pji  =   np.sqrt(np.linalg.norm(rij)**2 - np.dot(nj, rij)**2)
                    F   += - GradV_ij(rij, r, pij, pji, ni, nj, dni)
                    
                    counting =  True
            
            m      +=   1            
    
    
        # test that the force is -grad Vij. Very expensive!
        if test:
            f = -self.grad_pot_ij(positions.copy(), i, j)
            print 'forces from all images of %i to %i' %(j,i)
            for ll in range(3):
                print F[ll] - f[ll], F[ll]
                if 1e-9 < np.abs(f[ll]): 
                    if np.abs(F[ll] - f[ll])/np.abs(F[ll]) > 0.0001:
                        print i,j
                        print F
                        print f
                        raise
            print 
    
        return F #, f
    
    def get_potential_ij(self, positions, i, j):
        
        posits_ext      =   extend_structure(0., positions.copy(), self.pbc, self.cell)
        e_KC            =   0.

        
        def E_KC(rij, ni, nj, *args):
            
            r   =   np.linalg.norm(rij)
            pij =   np.sqrt(r**2 - np.dot(rij, ni)**2)
            pji =   np.sqrt(r**2 - np.dot(rij, nj)**2)
            
            return np.exp(-self.lamna*(r - self.z0))*(self.C + self.f(pij) + self.f(pji)) - self.A*(r/self.z0)**(-6.)

        ni          =   local_normal(i, posits_ext, self.layer_neighbors)
        ri          =   positions[i]
        rj          =   positions[j]
        
        m           =   0
        counting    =   True
        nj          =   local_normal(j, posits_ext, self.layer_neighbors)
        
        while counting:
            counting=   False
            
            if len(self.map_seqs) < m + 1:
                self.map_seqs.append(map_seq(m, self.n))
            
            rj_im   =   map_rj(rj, self.map_seqs[m], self.pbc, self.cell)
            
            #rj_im   =   map_rj(rj, map_seq(m, self.n), self.pbc, self.cell)
            
            for mrj in rj_im:
               
                rij =   mrj - ri
                r   =   np.linalg.norm(mrj - ri)
                if r < self.cutoff:
                    e_KC    +=  E_KC(rij, ni, nj) #/2.
                    
                    counting =  True
                    
            m      +=   1

            
        return e_KC

    