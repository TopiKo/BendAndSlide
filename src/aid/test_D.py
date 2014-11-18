'''
Created on 8.10.2014

@author: tohekorh
'''
import numpy as np
from ase import Atoms
from ase.visualize import view
from help2 import extend_structure, nrst_neigh, map_seq, map_rj, local_normal
from aid.help import find_layers
import scipy

            
class KC_potential_ij:
    
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
    
    
    def f(self, p):
        return np.exp(-(p/self.delta)**2)*(self.C0 \
                     +  self.C2*(p/self.delta)**2  \
                     +  self.C4*(p/self.delta)**4)
    
    
    def get_neigh_layer_indices(self, i, lay_inds):
            
        ret_lay_ind = []
        if i > 0:
            for j in lay_inds[i - 1]:
                ret_lay_ind.append(j)
        if i < len(lay_inds) - 1:
            for j in lay_inds[i + 1]:
                ret_lay_ind.append(j)
        return ret_lay_ind    
    
    def get_forces(self, positions, i, j):
        
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
                    dni[i,j]   =  scipy.misc.derivative(nj, ri[i], dx=1e-3, n=1, args=[i, j], order = 5)
            
            
            return dni      
            
            
        
        def GradV(rij, r, pij, pji, ni, nj, dni):
            
            
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
        
        ri              =   positions[i]
        F               =   np.zeros(3)
        
        ni              =   local_normal(i, posits_ext, self.layer_neighbors)
        dni             =   gradN(ri, i)
        
        neigh_indices   =   self.get_neigh_layer_indices(0, self.layer_indices)    
        
        for j in neigh_indices:
            rj              =   positions[j]
    
            m               =   0
            counting        =   True
            nj              =   local_normal(j, posits_ext, self.layer_neighbors)
            
            while counting:
                counting=   False
                rj_im   =   map_rj(rj, map_seq(m, self.n), self.pbc, self.cell)
                for mrj in rj_im:
                    rij =   mrj - ri
                    r   =   np.linalg.norm(mrj - ri)
                    if r < self.cutoff:
                        
                                
                        pij  =   np.sqrt(np.linalg.norm(rij)**2 - np.dot(ni, rij)**2)
                        pji  =   np.sqrt(np.linalg.norm(rij)**2 - np.dot(nj, rij)**2)
                        F   += - GradV(rij, r, pij, pji, ni, nj, dni)
                        
                        if m < 10:
                            counting =  True
                
                m      +=   1            
        
        f = -self.get_grad_pot(positions.copy(), i, j)
        
        print F[0] - f[0], F[1] - f[1], F[2] - f[2]
        print F
        print f
        print 
    
        return F, f
        
     
     
    def get_grad_pot(self, posits, i, j):                                
        
        def e(ri, *args):
            k               =   args[0]
            posits_use      =   posits.copy()   
            posits_use[i,k] =   ri
            
            return self.get_potential_energy(posits_use, i, j)
                 
        
        de = np.zeros(3)
        for k in range(3):
            de[k]  =  scipy.misc.derivative(e, posits[i,k], dx=1e-6, n=1, args=[k], order = 3)    
        
        return de
        
       
    def get_potential_energy(self, positions, i, j):
        
        
        posits_ext      =   extend_structure(0., positions.copy(), self.pbc, self.cell)
        e_KC            =   0.

        
        def E_KC(rij, ni, nj, *args):
            
            r   =   np.linalg.norm(rij)
            pij =   np.sqrt(r**2 - np.dot(rij, ni)**2)
            pji =   np.sqrt(r**2 - np.dot(rij, nj)**2)
            
            return np.exp(-self.lamna*(r - self.z0))*(self.C + self.f(pij) + self.f(pji)) - self.A*(r/self.z0)**(-6.)

        
        
        
        #for layer, layer_inds in enumerate(self.layer_indices):
            #layer  = 0
            #layer_inds = self.layer_indices[layer]
            
        #    for i in layer_inds:
        ni              =   local_normal(i, posits_ext, self.layer_neighbors)
        ri              =   positions[i]
        
        neigh_indices   =   self.get_neigh_layer_indices(0, self.layer_indices)       
        
        for j in neigh_indices:
            rj          =   positions[j]
            
            m           =   0
            counting    =   True
            nj          =   local_normal(j, posits_ext, self.layer_neighbors)
            
            while counting:
                counting=   False
                rj_im   =   map_rj(rj, map_seq(m, self.n), self.pbc, self.cell)
                for mrj in rj_im:
                   
                    rij =   mrj - ri
                    r   =   np.linalg.norm(mrj - ri)
                    if r < self.cutoff:
                        e_KC    +=  E_KC(rij, ni, nj) #/2.
                        if m < 10:
                            counting =  True
                        
                m      +=   1

            
        return e_KC

from help import make_graphene_slab    
from moldy.atom_groups import get_mask

bond        =   1.39695
a           =   np.sqrt(3)*bond # 2.462
h           =   3.38 

length      =   2*1             # slab length has to be integer*2
width       =   1               # slab width
N           =   2

atoms_init              =   make_graphene_slab(a,h,width,length,N, (True, True, False))[3]
lay_indices             =   find_layers(atoms_init.positions)[1]


atoms                   =   atoms_init.copy()
params                  =   {'bond':bond, 'a':a, 'h':h, 'acc':10}  
params['positions']     =   atoms_init.positions
params['cell']          =   atoms_init.get_cell().diagonal()
params['pbc']           =   atoms_init.get_pbc()
params['ia_dist']       =   14

top             =   get_mask(atoms_init.positions.copy(), 'top', 1, h)

new_pos     =   atoms_init.positions.copy()
for iat in range(len(atoms)):
    new_pos[iat][2] += new_pos[iat][0]*(-0.3) 
        
    if top[iat]:
        new_pos[iat][0] += bond/2 
atoms.positions =   new_pos

view(atoms)

kc_pot = KC_potential_ij(params)


i,j = 2,0

kc_forces = kc_pot.get_forces(new_pos, i, j)[0]
kc_pot_grad = kc_pot.get_grad_pot(new_pos, i, j)

print 'ERROR:'
print kc_forces + kc_pot_grad
print 'forces'
print kc_forces
