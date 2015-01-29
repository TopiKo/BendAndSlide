'''
Created on 11.11.2014

@author: tohekorh
'''

import numpy as np
from help2 import extend_structure, nrst_neigh, map_seq, map_rj, local_normal, which_layer
from aid.help import find_layers
import scipy
import time

        
            
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
        
        posits_ext              =   extend_structure(posits, self.pbc, self.cell)
        self.layer_neighbors    =   nrst_neigh(posits, posits_ext, 'layer')    
        self.layer_indices      =   find_layers(posits)[1]
        
         
    def adjust_positions(self, oldpositions, newpositions):
        pass
    
    
    def adjust_forces(self, posits, forces):
        
        take_t          =   True
        
        calcet          =   np.zeros((len(posits), len(posits)))
        if take_t:
            tot_t       =   0.   
            tot_grad    =   0.
            tot_loop    =   0.
            tot_lnormal =   0.
            tot_tnc     =   0.
            tot_tgln    =   0.                            

        
        for i in range(len(forces)):
            if self.chem_symbs[i] == 'C':
                for j in self.get_neigh_layer_indices_old(i, only_C = True):
                    if calcet[i,j] == 0:
                        
                        # Force due to atom j on atom i
                        
                        if take_t:
                            t1 = time.time()
                            fij, tg, tnc, tgln, tl, tln =   self.get_forces_ij(posits, i, j, test = False, take_t = True)
                            t2 = time.time()
                                               
                            tot_t +=    t2 - t1   
                            tot_grad    +=  tg
                            tot_loop    +=  tl
                            tot_lnormal +=  tln
                            tot_tnc     +=  tnc
                            tot_tgln    +=  tgln                            
                        else:
                            fij     =   self.get_forces_ij(posits, i, j, test = False)
                        # Force due to i on j is minus force due to j on i
                        forces[i,:] +=  fij  
                        forces[j,:] += -fij  
                        
                        calcet[i,j] = 1
                        calcet[j,i] = 1
        
        if take_t:
            print 'time grad = %.2f' %(tot_grad/tot_t*100) + '%'                 
            print '    coord change = %.2f' %(tot_tnc/tot_grad*100) + '%'                 
            print '    calc normal  = %.2f' %(tot_tgln/tot_grad*100) + '%'                 

            print 'time loop = %.2f' %(tot_loop/tot_t*100) + '%'                 
            print 'time norm = %.2f' %(tot_lnormal/tot_t*100) + '%'                 
                        

        
    def adjust_potential_energy(self, posits, energy):
        
        calcet          =   np.zeros((len(posits), len(posits)))
        e_KC            =   0.
        
        for i in range(len(posits)):
            if self.chem_symbs[i] == 'C':
                for j in self.get_neigh_layer_indices_old(i, only_C = True):
                    if calcet[i,j] == 0:
                        
                        e_KC += self.get_potential_ij(posits, i, j) 
                        
                        calcet[i,j] = 1
                        calcet[j,i] = 1
        return e_KC
     

    def get_neigh_layer_indices_old(self, i, only_C = True):
        
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
    
    def get_forces_ij(self, positions, i, j, test = False, take_t = False):
        
        # This module gives atoms j and all its 
        # images force on atom i provided that rij < cutoff.
        if take_t:
            self.tnc = 0
            self.tln = 0
        
        def gradN(ri, ind_bot):
            # Tama ulos, omaksi..
            # Matrix
            #dn1/dx dn2/dx dn3/dx
            #dn1/dy dn2/dy dn3/dy
            #dn1/dz dn2/dz dn3/dz

            def nj(xi, *args):
                i                       =   args[0]
                
                if take_t: tnc_1 =   time.time()
                posits_use              =   positions.copy()
                posits_use[ind_bot, i]  =   xi
                posits_ext_use          =   extend_structure(posits_use.copy(), self.pbc, self.cell)
                if take_t: 
                    tnc_2   =   time.time()
                    tln_1   =   tnc_2
                       
                nj                      =   local_normal(ind_bot, posits_ext_use, self.layer_neighbors)
                if take_t:
                    tln_2   =   time.time()
                    self.tnc    +=   tnc_2 - tnc_1  
                    self.tln    +=   tln_2 - tln_1                
                return nj
                
                
            dni = np.zeros((3,3))
            
            for i in range(3):
                dni[i,:]    =   scipy.misc.derivative(nj, ri[i], dx=1e-3, n=1, args=[i], order = 3)
                            
            return dni      
       
        
        def GradV_ij(rij, r, pij, pji, ni, nj, dni):
            # Tama ulos, omaksi..

            # Here we calculate the gradient of the KC- potential to obtain force:
            # See notes, for derivation.    
            
            # ulos... Selffiet poies...
            def g(p):
                return -2./self.delta**2*self.f(p) + np.exp(-(p/self.delta)**2)* \
                       (2.*self.C2/self.delta**2 + 4*self.C4*p**2/self.delta**4)
            
            gpij        =   g(pij)
            gpji        =   g(pji)
            
            njdotrij    =   nj[0]*rij[0] + nj[1]*rij[1] + nj[2]*rij[2]  # nj.rij
            nidotrij    =   ni[0]*rij[0] + ni[1]*rij[1] + ni[2]*rij[2]  # ni.rij
            
            bulkPreFac  =  -self.lamna*np.exp(-self.lamna*(r - self.z0))*(self.C + self.f(pij) + self.f(pji)) \
                        +   6*self.A*self.z0**6/r**7
            expPreFac   =   np.exp(-self.lamna*(r - self.z0))
                                    
            
            GV          =   np.zeros(3)
            for k in range(3):
                # np.dot(rij, dni[k]) matriisitulo?
                GV[k] =    bulkPreFac*(-rij[k]/r) \
                         + expPreFac*gpji*(njdotrij*(nj[k]) - rij[k]) \
                         + expPreFac*gpij*(nidotrij*(ni[k]  - np.dot(rij, dni[k])) - rij[k])

            return GV
        
        
        # tama ulos myos
        posits_ext      =   extend_structure(positions.copy(), self.pbc, self.cell)
        
        ri              =   positions[i]
        rj              =   positions[j]
        ##
        
        F               =   np.zeros(3)
        
        if take_t: tl_1 =   time.time()
        
        # tama ulos...
        ni              =   local_normal(i, posits_ext, self.layer_neighbors)
        nj              =   local_normal(j, posits_ext, self.layer_neighbors)
        ##
        
        if take_t: 
            tl_2 =   time.time()
            tg_1 =   tl_2
        
        # tamakin ulos..
        dni                 =   gradN(ri, i)
        
        
        if take_t: 
            tg_2            =   time.time()
            time_grad       =   tg_2 - tg_1
            time_lnormal    =   tl_2 - tl_1
            tl_1            =   tg_2
        
        # m counts the amount of translations Tx**n1 Ty**n2 rj = image_rj, n1 + n2 = m.
        m               =   0
        counting        =   True
        
        # sum over all images of j that are within cutoff
        # sqrt = math.sqrt
        while counting:
            counting    =   False
            
            # map_seqs defines all possible mappings Tx**n1 Ty**n2 with n1 + n2 = m
            # kenties kasaan ja arvaus?
            if len(self.map_seqs) < m + 1:
                self.map_seqs.append(map_seq(m, self.n))
            
            rj_im       =   map_rj(rj, self.map_seqs[m], self.pbc, self.cell)
            
            # np array rj_im..
            for mrj in rj_im:
                rij     =   mrj - ri
                r       =   np.sqrt(rij[0]**2 + rij[1]**2 + rij[2]**2) #np.linalg.norm(mrj - ri)
                
                if r < self.cutoff:
                    
                    # See notes:
                    niDotrij    =   ni[0]*rij[0] + ni[1]*rij[1] + ni[2]*rij[2] 
                    njDotrij    =   nj[0]*rij[0] + nj[1]*rij[1] + nj[2]*rij[2] 
                    
                    pij  =   np.sqrt(r**2 - niDotrij**2)
                    pji  =   np.sqrt(r**2 - njDotrij**2)
                    
                    F   += - GradV_ij(rij, r, pij, pji, ni, nj, dni)
                    
                    counting =  True
            
            m      +=   1     
                   
        if take_t: 
            tl_2        =   time.time()
            time_loop   =   tl_2 - tl_1
        
        
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
        
        if take_t: return F, time_grad, self.tnc, self.tln, time_loop, time_lnormal 
        else: return F
    
    def get_potential_ij(self, positions, i, j):
        
        posits_ext      =   extend_structure(positions.copy(), self.pbc, self.cell)
        e_KC            =   0.

        
        def E_KC(rij, ni, nj, *args):
            
            r   =   np.linalg.norm(rij)
            pij =   np.sqrt(r**2 - np.dot(rij, ni)**2)
            pji =   np.sqrt(r**2 - np.dot(rij, nj)**2)
            
            return np.exp(-self.lamna*(r - self.z0))*(self.C + self.f(pij) + self.f(pji)) - self.A*(r/self.z0)**(-6.)

        ri          =   positions[i]
        rj          =   positions[j]
        
        ni          =   local_normal(i, posits_ext, self.layer_neighbors)
        nj          =   local_normal(j, posits_ext, self.layer_neighbors)
        
        m           =   0
        counting    =   True
        
        while counting:
            counting=   False
            
            if len(self.map_seqs) < m + 1:
                self.map_seqs.append(map_seq(m, self.n))
            
            rj_im   =   map_rj(rj, self.map_seqs[m], self.pbc, self.cell)
            
            for mrj in rj_im:
                
                rij =   mrj - ri
                r   =   np.linalg.norm(mrj - ri)
                if r < self.cutoff:
                    
                    e_KC    +=  E_KC(rij, ni, nj)
                    counting =  True
                    
            m      +=   1

            
        return e_KC

    