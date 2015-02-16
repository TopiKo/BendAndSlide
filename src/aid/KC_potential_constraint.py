'''
Created on 11.11.2014

@author: tohekorh
'''

import numpy as np
from help2 import extend_structure, nrst_neigh, map_seq, map_rj, local_normal, which_layer #, map_rjs
from aid.help import find_layers
#from math import sqrt
from numpy import zeros, exp, dot, sqrt
import scipy
#import time


def gradN(ri, ind_bot, positions, pbc, cell, layer_neighbors, take_t = False):
    # Tama ulos, omaksi..
    # Matrix
    #dn1/dx dn2/dx dn3/dx
    #dn1/dy dn2/dy dn3/dy
    #dn1/dz dn2/dz dn3/dz
    
    def nj(xi, *args):
        i                       =   args[0]
        
        posits_use              =   positions.copy()
        posits_use[ind_bot, i]  =   xi
        posits_ext_use          =   extend_structure(posits_use.copy(), pbc, cell)
               
        nj                      =   local_normal(ind_bot, posits_ext_use, layer_neighbors)
                
        return nj
        
        
    dni = zeros((3,3))
    
    for i in range(3):
        dni[i,:]    =   scipy.misc.derivative(nj, ri[i], dx=1e-3, n=1, args=[i], order = 3)
                    
    return dni    

def GradV_ij(rij, r, pij, pji, ni, nj, dni, params):
    # Tama ulos, omaksi..
    
    # Here we calculate the gradient of the KC- potential to obtain force:
    # See notes, for derivation.    
    A, z0, lamna, delta, C, C0, C2, C4  =   params
    
    #tg1         =   time.time()
    gpij        =   -2./delta**2*f(pij, delta, C0, C2, C4) + exp(-(pij/delta)**2)* \
                    (2.*C2/delta**2 + 4*C4*pij**2/delta**4)
    gpji        =   -2./delta**2*f(pji, delta, C0, C2, C4) + exp(-(pji/delta)**2)* \
                    (2.*C2/delta**2 + 4*C4*pji**2/delta**4)
    #tg2         =   time.time()
    
    #tndotr1     =   time.time()    
    njdotrij    =   nj[0]*rij[0] + nj[1]*rij[1] + nj[2]*rij[2]  # nj.rij
    nidotrij    =   ni[0]*rij[0] + ni[1]*rij[1] + ni[2]*rij[2]  # ni.rij
    #tndotr2     =   time.time()    
    
    #tpre1       =   time.time()    

    bulkPreFac  =  -lamna*exp(-lamna*(r - z0))*(C + f(pij, delta, C0, C2, C4) + f(pji, delta, C0, C2, C4)) \
                +   6*A*z0**6/r**7
    expPreFac   =   exp(-lamna*(r - z0))

    #tpre2       =   time.time()      
    
    #tdotdn1     =   tpre2                      
    rijDotdni   =   dot(dni, rij)
    #tdotdn2     =   time.time()
    
    #tG1         =   tdotdn2
    GV          =    bulkPreFac*(-rij/r) \
                   + expPreFac*gpji*(njdotrij*nj - rij) \
                   + expPreFac*gpij*(nidotrij*(ni  - rijDotdni) - rij)
    #tG2         =   time.time()
    
    
    #print tg2 - tg1, tndotr2 - tndotr1, tpre2 - tpre1, tdotdn2 - tdotdn1, tG2 - tG1 
    
    return GV

def f(p, delta, C0, C2, C4):
    #A, z0, lamna, delta, C, C0, C2, C4 =   params
    
    return np.exp(-(p/delta)**2)*(C0 \
                 +  C2*(p/delta)**2  \
                 +  C4*(p/delta)**4)


def get_potential_ij(ri, rj, ni, nj, positions, i, j, params, \
                     pbc, cell, cutoff, n, map_seqs):
    
    A, z0, lamna, delta, C, C0, C2, C4 =   params
    
    e_KC            =   0.

    
    #def E_KC(r, pij, pji, *args):
        
        #r   =   np.linalg.norm(rij)
        #pij =   sqrt(r**2 - dot(rij, ni)**2)
        #pji =   sqrt(r**2 - dot(rij, nj)**2)
        
    #    return exp(-lamna*(r - z0))*(C + f(pij, delta, C0, C2, C4) + f(pji, delta, C0, C2, C4)) - A*(r/z0)**(-6.)

    
    
    m           =   0
    counting    =   True
    update_maps     =   False
    
    while counting:
        counting=   False
        
        if len(map_seqs) < m + 1:
            update_maps     =   True
            map_seqs.append(map_seq(m, n))
        
        rj_im   =   map_rj(rj, map_seqs[m], pbc, cell)
        
        for mrj in rj_im:
            
            rij =   mrj - ri
            r   =   sqrt(rij[0]**2 + rij[1]**2 + rij[2]**2)
            
            if r < cutoff:
                
                
                
                #rijDotni    =   rij[0]*ni[0] + rij[1]*ni[1] + rij[2]*ni[2]
                #rijDotnj    =   rij[0]*nj[0] + rij[1]*nj[1] + rij[2]*nj[2]
                
                niDotrij    =   ni[0]*rij[0] + ni[1]*rij[1] + ni[2]*rij[2] 
                njDotrij    =   nj[0]*rij[0] + nj[1]*rij[1] + nj[2]*rij[2] 
                
                if -1e-12 < r**2 - niDotrij**2 < 0.:
                    pij     =   0.
                else:
                    pij     =   sqrt(r**2 - niDotrij**2)
                    
                if -1e-12 < r**2 - njDotrij**2 < 0.:
                    pji     =   0.
                else:
                    pji     =   sqrt(r**2 - njDotrij**2)

                
                #pij         =   sqrt(r**2 - rijDotni**2)
                #pji         =   sqrt(r**2 - rijDotnj**2)
        
                
                
                e_KC       +=   exp(-lamna*(r - z0))*(C + f(pij, delta, C0, C2, C4) \
                                                        + f(pji, delta, C0, C2, C4)) \
                                                        - A*(r/z0)**(-6.)
                
                #e_KC    +=  E_KC(r, pij, pji)
                counting =  True
                
        m      +=   1

    if update_maps:
        return e_KC, map_seqs
    else:
        return e_KC, None

def grad_pot_ij(posits, i, j, params, \
                pbc, cell, cutoff, n, map_seqs, layer_neighbors):                                
    
    def e(rik, *args):
        k               =   args[0]
        posits_use      =   posits.copy()   
        posits_use[i,k] =   rik
        rj              =   posits_use[j]
        ri              =   posits_use[i]
        posits_ext      =   extend_structure(posits_use, pbc, cell)

        ni          =   local_normal(i, posits_ext, layer_neighbors)
        nj          =   local_normal(j, posits_ext, layer_neighbors)
        
        return get_potential_ij(ri, rj, ni, nj, posits_use, i, j, params, \
                                     pbc, cell, cutoff, n, map_seqs)[0]
             
        
    de          =   zeros(3)
    for k in range(3):
        de[k]   =   scipy.misc.derivative(e, posits[i,k], dx=1e-6, n=1, args=[k], order = 3)    
    
    return de

def get_forces_ij(ri, rj, ni, nj, dni, positions, posits_ext, i, j, params, \
                      pbc, cell, cutoff, n, map_seqs, layer_neighbors = None):
        
    # This module gives atoms j and all its 
    # images force on atom i provided that rij < cutoff.
     
    F               =   zeros(3)
    
    # m counts the amount of translations Tx**n1 Ty**n2 rj = image_rj, n1 + n2 = m.
    m               =   0
    counting        =   True
    update_maps     =   False
    
    #tm = 0
    #tg = 0
    #tr = 0
    
    while counting:
        counting    =   False
        
        #tm1         =   time.time()
        
        if len(map_seqs) < m + 1:
            update_maps     =   True
            map_seqs.append(map_seq(m, n))
        
        rj_im       =   map_rj(rj, map_seqs[m], pbc, cell)
        #tm2         =   time.time()
        
        #tm         +=   tm2 - tm1
              
        for mrj in rj_im:
            #tr1     =   time.time()
        
            rij     =   mrj - ri
            r       =   sqrt(rij[0]**2 + rij[1]**2 + rij[2]**2) 
            #tr2     =   time.time()
            
            #tr     +=  tr2 - tr1
            if r < cutoff:
                
                # See notes:
                #tg1         =   time.time()
                
                niDotrij    =   ni[0]*rij[0] + ni[1]*rij[1] + ni[2]*rij[2] 
                njDotrij    =   nj[0]*rij[0] + nj[1]*rij[1] + nj[2]*rij[2] 
                
                if -1e-12 < r**2 - niDotrij**2 < 0.:
                    pij     =   0.
                else:
                    pij     =   sqrt(r**2 - niDotrij**2)
                    
                if -1e-12 < r**2 - njDotrij**2 < 0.:
                    pji     =   0.
                else:
                    pji     =   sqrt(r**2 - njDotrij**2)
                
                F          += - GradV_ij(rij, r, pij, pji, ni, nj, dni, params)
                #tg2         =   time.time()
                
                #  tg         +=   tg2 - tg1             
                counting    =   True
        
        m      +=   1     
    
    
    # test that the force is -grad Vij. Very expensive!
    if layer_neighbors != None:
        f = -grad_pot_ij(positions.copy(), i, j, params, \
                              pbc, cell, cutoff, n, map_seqs, layer_neighbors)
        print 'forces from all images of %i to %i' %(j,i)
        for ll in range(3):
            print F[ll] - f[ll], F[ll]
            if 1e-6 < np.abs(f[ll]) or 1e-6 < np.abs(F[ll]): 
                if np.abs(F[ll] - f[ll])/np.abs(F[ll]) > 0.001:
                #if np.abs(F[ll] - f[ll]) > 1e-6:
                
                    print i,j
                    print F
                    print f
                    raise
        print 
    
    if update_maps:
        return F, map_seqs #, tm, tg, tr
    else:
        return F, None #, tm, tg, tr


def get_neigh_layer_indices(layer, layer_indices):
    
    neigh_lay   =   []
    
    if layer > 0:
        neigh_lay.append(layer - 1)
        
    if layer < len(layer_indices) - 1:
        neigh_lay.append(layer - 1)
    
    neigh_ind   =   []
    
    for j in neigh_lay:
        for k in layer_indices[j]:
            neigh_ind.append(k)
    
    return neigh_ind
   
class KC_potential:
    
    # This is a constraint class to include the registry dependent 
    # interlayer potential to ase. 
    
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
        
        self.params =   [self.A, self.z0, self.lamna, self.delta, self.C, self.C0, self.C2, self.C4]
        
        self.n          =   0
        self.map_seqs   =  []
        for i in range(3):
            if self.pbc[i]: self.n += 1   
        
        posits_ext              =   extend_structure(posits, self.pbc, self.cell)
        self.layer_neighbors    =   nrst_neigh(posits, posits_ext, 'layer')    
        self.layer_indices      =   find_layers(posits)[1]
        
        self.neighbor_layer_inds            =   np.empty(len(self.layer_indices), dtype = 'object')
        for i in range(len(self.layer_indices)):
            self.neighbor_layer_inds[i]     =   get_neigh_layer_indices(i, self.layer_indices)         
        
         
    def adjust_positions(self, oldpositions, newpositions):
        pass
    
    
    def adjust_forces(self, posits, forces):
        
        #tt1             =   time.time()
        calcet          =   zeros((len(posits), len(posits)))
        
        
        params          =   self.params
        pbc, cell, cutoff, n, map_seqs   \
                        =   self.pbc, self.cell, self.cutoff, self.n, self.map_seqs
        posits_ext      =   extend_structure(posits.copy(), self.pbc, self.cell)
        layer_indices   =   self.layer_indices
        layer_neighbors =   self.layer_neighbors
        chem_symbs      =   self.chem_symbs 
        neighbor_layer_inds   \
                        =   self.neighbor_layer_inds

        #tlf1            =   time.time()
        dnGreat         =   np.empty(len(forces), dtype = 'object')
        
        for i, r in enumerate(posits):
            if chem_symbs[i] == 'C':
                dnGreat[i]      =   gradN(r, i, posits, pbc, cell, layer_neighbors)
        
        #tlf2             =   time.time()
        
        #tlf =   tlf2 - tlf1        
        #tf = 0
        #tni = 0       
        #tnj = 0 
        #tmt =   0        
        #tgt =   0        
        #trt =   0
        for i in range(len(forces)):
            if chem_symbs[i] == 'C':
                ri              =   posits[i]
                #tni1            =   time.time()
                ni              =   local_normal(i, posits_ext, layer_neighbors)
                #tni2            =   time.time()
                
                #tni            +=   tni2 - tni1
                neigh_indices   =   neighbor_layer_inds[which_layer(i, layer_indices)[0]]
                for j in neigh_indices: 
                    
                    if calcet[i,j] == 0 and chem_symbs[j] == 'C':
                        rj              =   posits[j]
                        #tnj1             =   time.time()
                        nj              =   local_normal(j, posits_ext, layer_neighbors)
                        #tnj2             =   time.time()
                        # Force due to atom j on atom i
                        #tf1             =   time.time()
                        fij, new_maps   =   get_forces_ij(ri, rj, ni, nj, dnGreat[i], \
                                                       posits, posits_ext, i, j, params, \
                                                       pbc, cell, cutoff, n, map_seqs) #,
                                                       #layer_neighbors = layer_neighbors)
                        #tf2             =   time.time()
                        #tf             +=   tf2 - tf1
                        #tnj            +=   tnj2 - tnj1
                        #tmt            +=   tm
                        #tgt            +=   tg
                        #trt            +=   tr
                        if new_maps != None:    self.map_seqs   =   new_maps
                        
                        # Force due to i on j is minus force due to j on i
                        forces[i,:] +=  fij  
                        forces[j,:] += -fij  
                        
                        calcet[i,j] = 1
                        calcet[j,i] = 1
        #tt2             =   time.time()
        #tt  =   tt2 - tt1

        #print 'time grad = %.2f' %(tlf/tt*100) + '%'                 
        #print 'time forc = %.2f' %(tf/tt*100) + '%'                 
        #print '    time maps = %.2f' %(tmt/tt*100) + '%'                 
        #print '    time rad  = %.2f' %(trt/tt*100) + '%'                 
        #print '    time grad = %.2f' %(tgt/tt*100) + '%'                 
        #print 'time norm = %.2f' %((tni + tnj)/tt*100) + '%'                 
        #print 'time tot  = %.2fs' %(tt) 
        

        
    def adjust_potential_energy(self, posits, energy):
        
        calcet          =   zeros((len(posits), len(posits)))
        e_KC            =   0.
        params          =   self.params
        pbc, cell, cutoff, n, map_seqs    =   self.pbc, self.cell, self.cutoff, self.n, self.map_seqs
        posits_ext      =   extend_structure(posits.copy(), self.pbc, self.cell)
        layer_indices   =   self.layer_indices
        layer_neighbors =   self.layer_neighbors
        chem_symbs      =   self.chem_symbs
        neighbor_layer_inds     =   self.neighbor_layer_inds
        
        #et  = 0
        #tnj =   0
        #tni =   0
        #tt1             =   time.time()
        
        for i in range(len(posits)):
            
            if chem_symbs[i] == 'C':
                ri              =   posits[i]
                #tni1            =   time.time()
                ni              =   local_normal(i, posits_ext, layer_neighbors)
                #tni2            =   time.time()
                
                #tni            +=   tni2 - tni1
                
                
                neigh_indices   =   neighbor_layer_inds[which_layer(i, layer_indices)[0]]
                
                for j in neigh_indices: 
                    
                    if calcet[i,j] == 0 and chem_symbs[j] == 'C':
                        rj          =   posits[j]
                        
                        
                        #tnj1        =   time.time()
                        nj          =   local_normal(j, posits_ext, layer_neighbors)
                        #tnj2        =   time.time()
                        
                        #et1         =   time.time()
                        e, new_maps =   get_potential_ij(ri, rj, ni, nj, posits, i, j, params, \
                                                         pbc, cell, cutoff, n, map_seqs) 
                        #et2         =   time.time()
                        
                        #et         +=   et2 - et1
                        #tnj        +=   tnj2 - tnj1
                        
                        e_KC       +=  e
                        if new_maps != None:    self.map_seqs   =   new_maps
                        calcet[i,j] =  1
                        calcet[j,i] =  1
        
        #tt2             =   time.time()
        #tt              =   tt2 - tt1
        #print 'time e = %.2f' %(et/tt*100) + '%'                 
        #print 'time n = %.2f' %((tnj + tni)/tt*100) + '%'                 
        #print 'time tot, e = %.2fs' %tt     
        return e_KC
    