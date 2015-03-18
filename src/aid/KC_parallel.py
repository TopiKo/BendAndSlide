'''
        Created on 18.2.2015
        
        @author: tohekorh
'''

import numpy as np
from help2 import extend_structure, nrst_neigh, map_seq, map_rj, local_normal, which_layer #, map_rjs
from aid.help import find_layers
#from math import sqrt
from numpy import zeros, exp, dot, sqrt
from threading import Thread
from threading import Semaphore
import array
import scipy
import time
import multiprocessing

def gradN(ri, ind_bot, positions, pbc, cell, layer_neighbors, take_t = False):
    # Matrix
    #dn1/dx dn2/dx dn3/dx
    #dn1/dy dn2/dy dn3/dy
    #dn1/dz dn2/dz dn3/dz
    
    #return np.ones((3,3))
    
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
    


def GradV_ij(rij, r, pij, pji, ni, nj, dni, dnj, params):
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

    
    rijDotdni   =   dot(dni, rij)
    
    FIJ          =    -(bulkPreFac*(-rij/r) \
                   + expPreFac*gpji*(njdotrij*nj - rij) \
                   + expPreFac*gpij*(nidotrij*(ni  - rijDotdni) - rij))
    
    
    ###################
    # THIS TERMS IS CORRECTION BECAUSE IN KC FIJ != FJI, see deriv from notes...
    rijDotdnj   =   dot(dnj, rij)

    fji_add     =   exp(-lamna*(r - z0))*(gpij*(nidotrij)*rijDotdni + gpji*(njdotrij)*rijDotdnj)
    #Fij + Fji  =    fij_add
    
    ###################
    
    return FIJ, fji_add - FIJ

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

                
                e_KC       +=   exp(-lamna*(r - z0))*(C + f(pij, delta, C0, C2, C4) \
                                                        + f(pji, delta, C0, C2, C4)) \
                                                        - A*(r/z0)**(-6.)
                
                counting =  True
                
        m      +=   1

    if update_maps:
        return e_KC, map_seqs
    else:
        return e_KC, None

def grad_pot_ij(posits, i, j, params, \
                pbc, cell, cutoff, n, map_seqs, layer_neighbors, jtop = True):                                
    
    if jtop:
        njfac   =  -1
        nifac   =   1
    else:
        njfac   =   1
        nifac   =  -1

    def e(rik, *args):
        k               =   args[0]
        nifac, njfac    =   args[1]
        posits_use      =   posits.copy()   
        posits_use[i,k] =   rik
        rj              =   posits_use[j]
        ri              =   posits_use[i]
        posits_ext      =   extend_structure(posits_use, pbc, cell)

        ni          =   local_normal(i, posits_ext, layer_neighbors)*nifac
        nj          =   local_normal(j, posits_ext, layer_neighbors)*njfac
        
        
        
        return get_potential_ij(ri, rj, ni, nj, posits_use, i, j, params, \
                                     pbc, cell, cutoff, n, map_seqs)[0]
             
        
    de          =   zeros(3)
    
    for k in range(3):
        de[k]   =   scipy.misc.derivative(e, posits[i,k], dx=1e-6, n=1, args=[k, [nifac, njfac]], order = 3)    
        
    return de

def get_forces_ij(ri, rj, ni, nj, dni, dnj, positions, posits_ext, i, j, params, \
                      pbc, cell, cutoff, n, map_seqs, layer_neighbors = None, jtop = False):
        
    # This module gives atoms j and all its 
    # images force on atom i provided that rij < cutoff.
    Fij             =   zeros(3)
    Fji             =   zeros(3)
    
    
    # m counts the amount of translations Tx**n1 Ty**n2 rj = image_rj, n1 + n2 = m.
    m               =   0
    counting        =   True
    update_maps     =   False
    
    
    while counting:
        counting    =   False
        
        if len(map_seqs) < m + 1:
            update_maps     =   True
            map_seqs.append(map_seq(m, n))
        
        rj_im       =   map_rj(rj, map_seqs[m], pbc, cell)
              
        for mrj in rj_im:
            
            rij     =   mrj - ri
            r       =   sqrt(rij[0]**2 + rij[1]**2 + rij[2]**2) 
            
            if r < cutoff:
                
                # See notes:
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
                
                #F          += - GradV_ij(rij, r, pij, pji, ni, nj, dni, params)
                
                F1, F2      =   GradV_ij(rij, r, pij, pji, ni, nj, dni, dnj, params)
                Fij        +=   F1
                Fji        +=   F2 
                
                
                counting    =   True
        
        m      +=   1     
    
    
    # test that the force is -grad Vij. Very expensive!
    if layer_neighbors != None:
        
        fij = -grad_pot_ij(positions.copy(), i, j, params, \
                              pbc, cell, cutoff, n, map_seqs, layer_neighbors, jtop)
        fji= -grad_pot_ij(positions.copy(), j, i, params, \
                              pbc, cell, cutoff, n, map_seqs, layer_neighbors, not jtop)

        print 'forces from all images of %i to %i' %(j,i)
        for ll in range(3):
            print Fij[ll] - fij[ll], Fij[ll]
            if 1e-6 < np.abs(fij[ll]) or 1e-6 < np.abs(Fij[ll]): 
                if np.abs(Fij[ll] - fij[ll]) > 1e-6:
                #if np.abs(F[ll] - f[ll]) > 1e-6:
                
                    print i,j, jtop
                    print Fij
                    print f
                    raise
        
        print 'forces from all images of %i to %i' %(i,j)
        for ll in range(3):
            print Fji[ll] - fji[ll], Fji[ll]
            if 1e-6 < np.abs(fji[ll]) or 1e-6 < np.abs(Fji[ll]): 
                if np.abs(Fji[ll] - fji[ll]) > 1e-6:
                #if np.abs(F[ll] - f[ll]) > 1e-6:
                
                    print i,j, jtop
                    print Fij
                    print f
                    raise
        print 
        
    
    if update_maps:
        return Fij, Fji, map_seqs #, tm, tg, tr
    else:
        return Fij, Fji, None #, tm, tg, tr


def get_neigh_layer_indices(layer, layer_indices):
    
    neigh_ind  =   []
    
    if layer > 0:
        neigh_ind.append(['bottom', layer_indices[layer - 1]])
        
    if layer < len(layer_indices) - 1:
        neigh_ind.append(['top', layer_indices[layer + 1]])
        
        
    return neigh_ind   



   
class KC_potential_p:
    
    # This is a constraint class to include the registry dependent 
    # interlayer potential to ase. 
    
    def __init__(self, params):
        
        posits      =   params['positions']
        
        self.chem_symbs =   params['chemical_symbols']
        self.pbc    =   params['pbc']
        self.cell   =   params['cell']
        self.cutoff =   params['ia_dist']
        # parameters as reported in original KC paper
        self.delta  =   0.578       # Angstrom   
        self.C0     =   15.71*1e-3  # [meV]
        self.C2     =   12.29*1e-3  # [meV]
        self.C4     =   4.933*1e-3  # [meV]
        self.C      =   3.030*1e-3  # [meV]
        self.lamna  =   3.629       # 1/Angstrom
        self.z0     =   3.34        # Angstrom
        self.A      =   10.238*1e-3 # [meV]     
        
        self.params =   [self.A, self.z0, self.lamna, self.delta, self.C, self.C0, self.C2, self.C4]
        
        #self.threadMaster   =   ThreadMaster(6)
        self.cores          =   max((2, params['ncores']))
        self.n              =   0
        self.map_seqs       =  []
        for i in range(3):
            if self.pbc[i]: self.n += 1   
        
        posits_ext              =   extend_structure(posits, self.pbc, self.cell)
        self.layer_neighbors    =   nrst_neigh(posits, posits_ext, 'layer')    
        self.layer_indices      =   find_layers(posits)[1]
        
        self.neighbor_layer_inds           =   np.empty(len(self.layer_indices), dtype = 'object')

        for i in range(len(self.layer_indices)):
            self.neighbor_layer_inds[i]    =   get_neigh_layer_indices(i, self.layer_indices)       
        
         
    def adjust_positions(self, oldpositions, newpositions):
        pass
    
    
    def adjust_forces(self, posits, forces):
        
        #tt1             =   time.time()
        #calcet          =   zeros((len(posits), len(posits)))
        
        params          =   self.params
        pbc, cell, cutoff, n, map_seqs   \
                        =   self.pbc, self.cell, self.cutoff, self.n, self.map_seqs
        posits_ext      =   extend_structure(posits.copy(), self.pbc, self.cell)
        layer_indices   =   self.layer_indices
        layer_neighbors =   self.layer_neighbors
        chem_symbs      =   self.chem_symbs 
        
        neighbor_layer_inds2  \
                        =   self.neighbor_layer_inds
        
        dnGreat         =   np.empty(len(forces), dtype = 'object')
        
        
        # PARALLELL BEGIN
        
        # Divide the loop over alla atoms into split_len parts.
        
        if self.cores > 1:
            split_len       =   float(len(posits))/self.cores
            split_arr       =   [int((i + 1)*split_len) for i in range(self.cores - 1)]
        else:
            split_len       =   len(posits)
            split_arr       =   [0]
        split_posits    =   np.split(posits, split_arr)
        split_arr       =   np.insert(split_arr, 0, 0)
        
        
        # Arrays for paralell computing
        jobs        =   []
        ar11        =   multiprocessing.Array('d', np.zeros(len(forces)))
        ar12        =   multiprocessing.Array('d', np.zeros(len(forces)))
        ar13        =   multiprocessing.Array('d', np.zeros(len(forces)))
        ar21        =   multiprocessing.Array('d', np.zeros(len(forces)))
        ar22        =   multiprocessing.Array('d', np.zeros(len(forces)))
        ar23        =   multiprocessing.Array('d', np.zeros(len(forces)))
        ar31        =   multiprocessing.Array('d', np.zeros(len(forces)))
        ar32        =   multiprocessing.Array('d', np.zeros(len(forces)))
        ar33        =   multiprocessing.Array('d', np.zeros(len(forces)))
            
        
        for i, pos_arr in enumerate(split_posits):
            
            args        =   (i, split_arr[i], pos_arr,  ar11, ar12, ar13,  \
                                                        ar21, ar22, ar23,  \
                                                        ar31, ar32, ar33,  \
                                                        posits, pbc, cell, \
                                                        layer_neighbors, chem_symbs)
            
            process     =   multiprocessing.Process(target=self.pre_gradN, 
                                                    args=args)
            jobs.append(process)

        # Start the processes (i.e. calculate the random number lists)        
        for j in jobs:
            j.start()
        
        # Ensure all of the processes have finished
        for j in jobs:
            j.join()
        

        for i in range(len(forces)):
            dnGreat[i]      =   np.zeros((3,3))
            dnGreat[i][0,0] =   ar11[i]
            dnGreat[i][0,1] =   ar12[i]
            dnGreat[i][0,2] =   ar13[i]
            dnGreat[i][1,0] =   ar21[i]
            dnGreat[i][1,1] =   ar22[i]
            dnGreat[i][1,2] =   ar23[i]
            dnGreat[i][2,0] =   ar31[i]
            dnGreat[i][2,1] =   ar32[i]
            dnGreat[i][2,2] =   ar33[i]
        

        
        jobs        =   []
        
        af1         =   multiprocessing.Array('d', np.zeros(len(forces)))
        af2         =   multiprocessing.Array('d', np.zeros(len(forces)))
        af3         =   multiprocessing.Array('d', np.zeros(len(forces)))
        
        afj1        =   multiprocessing.Array('d', np.zeros(len(forces)))
        afj2        =   multiprocessing.Array('d', np.zeros(len(forces)))
        afj3        =   multiprocessing.Array('d', np.zeros(len(forces)))
        
        calcet      =   multiprocessing.Array('i', array.array('i', (0 for i in range(0,len(posits)**2))))
        
        for k, pos_arr in enumerate(split_posits):
            
            
            args        =   (split_arr[k], pos_arr,  af1, af2, af3, afj1, afj2, afj3, dnGreat,\
                             posits, pbc, cell, layer_neighbors, \
                             chem_symbs, posits_ext, neighbor_layer_inds2, \
                             layer_indices, params, cutoff, n, map_seqs, calcet, len(forces))
            
            process     =   multiprocessing.Process(target=self.forces_on_posArr, 
                                                    args=args)
            jobs.append(process)

        # Start the processes         
        for j in jobs:
            j.start()
        
        # Ensure all of the processes have finished
        for j in jobs:
            j.join()
        
        
        for i in range(len(forces)):
            forces[i,0]     +=  af1[i] + afj1[i]
            forces[i,1]     +=  af2[i] + afj2[i]
            forces[i,2]     +=  af3[i] + afj3[i]
            #print i, af1[i], af2[i], af3[i]
            
    def pre_gradN(self, i, split_beg, pos_arr,  ar11, ar12, ar13, \
                                                ar21, ar22, ar23, \
                                                ar31, ar32, ar33, \
                                                posits, pbc, cell, \
                                                layer_neighbors, chem_symbs):
        
        for j, r in enumerate(pos_arr):
            ind_atom    =   j  + split_beg
            if chem_symbs[ind_atom] == 'C':
                dni             =   gradN(r, ind_atom, posits, pbc, cell, layer_neighbors)
                ar11[ind_atom]  =   dni[0,0]
                ar12[ind_atom]  =   dni[0,1]
                ar13[ind_atom]  =   dni[0,2]
                ar21[ind_atom]  =   dni[1,0]
                ar22[ind_atom]  =   dni[1,1]
                ar23[ind_atom]  =   dni[1,2]
                ar31[ind_atom]  =   dni[2,0]
                ar32[ind_atom]  =   dni[2,1]
                ar33[ind_atom]  =   dni[2,2]
    
      
        
    def forces_on_posArr(self, split_beg, pos_arr,  af1, af2, af3, afj1, afj2, afj3, dnGreat, \
                             posits, pbc, cell, layer_neighbors, \
                             chem_symbs, posits_ext, neighbor_layer_inds2, \
                             layer_indices, params, cutoff, n, map_seqs, calcet, natoms):
        
            
        for l, ri in enumerate(pos_arr):
            i   =   l  + split_beg 
            if chem_symbs[i] == 'C':
                ni_f            =   local_normal(i, posits_ext, layer_neighbors)
                dni_f           =   dnGreat[i]
                    
                neigh           =   neighbor_layer_inds2[which_layer(i, layer_indices)[0]]
                
                for nset in neigh:
                    
                    
                    
                    if nset[0] == 'bottom':
                        norm_fac    =   1
                        ni          =   -1*ni_f
                        dni         =   -dni_f
                        jtop        =   False
                    elif nset[0] == 'top':
                        norm_fac    =   -1   
                        ni          =   1*ni_f
                        dni         =   dni_f
                        jtop        =   True
                    
                    neigh_indices   =   nset[1]
                
                    for j in neigh_indices: 
                        if chem_symbs[j] == 'C' and calcet[i*natoms + j] == 0:

                            calcet[i*natoms + j]    =  1
                            calcet[i + j*natoms]    =  1
                            

                            rj              =   posits[j]
                            nj              =   local_normal(j, posits_ext, layer_neighbors)*norm_fac
                            dnj             =   dnGreat[j]*norm_fac
                            
                            # Force due to atom j on atom i
                            fij, fji, new_maps   =   get_forces_ij(ri, rj, ni, nj, dni, dnj, \
                                                           posits, posits_ext, i, j, params, \
                                                           pbc, cell, cutoff, n, map_seqs) #,
                                                           #layer_neighbors = layer_neighbors, \
                                                           #jtop = jtop)
                    
                            if new_maps != None:    self.map_seqs   =   new_maps
                            
                            #print i, j, fij
                            
                            af1[i]  +=  fij[0]
                            af2[i]  +=  fij[1]
                            af3[i]  +=  fij[2]

                            afj1[j] +=  fji[0]
                            afj2[j] +=  fji[1]
                            afj3[j] +=  fji[2]
         
                            #af1[i]  +=  fij[0]
                            #af2[i]  +=  fij[1]
                            #af3[i]  +=  fij[2]
     
     
    def adjust_potential_energy(self, posits, energy):
        
        params          =   self.params
        pbc, cell, cutoff, n, map_seqs    \
                        =   self.pbc, self.cell, self.cutoff, self.n, self.map_seqs
        posits_ext      =   extend_structure(posits.copy(), self.pbc, self.cell)
        layer_indices   =   self.layer_indices
        layer_neighbors =   self.layer_neighbors
        chem_symbs      =   self.chem_symbs
        neighbor_layer_inds2  \
                        =   self.neighbor_layer_inds
        
        split_len       =   float(len(posits))/self.cores
        split_arr       =   [int((i + 1)*split_len) for i in range(self.cores - 1)]
        split_posits    =   np.split(posits, split_arr)
        split_arr       =   np.insert(split_arr, 0, 0)
        
        
        jobs            =   []
        #e_KCm           =   multiprocessing.Value('d', 0.)
        e_KCm           =   multiprocessing.Array('d', np.zeros(self.cores))
        calcet          =   multiprocessing.Array('i', array.array('i', (0 for i in range(0,len(posits)**2))))
        
        
        
        for k, pos_arr in enumerate(split_posits):
            
            args        =   (k, split_arr[k], pos_arr,  e_KCm, \
                             posits, pbc, cell, layer_neighbors, \
                             chem_symbs, posits_ext, neighbor_layer_inds2, \
                             layer_indices, params, cutoff, n, map_seqs, calcet, len(posits))
            
            process     =   multiprocessing.Process(target=self.energy_on_posArr, 
                                                    args=args)
            jobs.append(process)

        # Start the processes        
        for j in jobs:
            j.start()
        
        # Ensure all of the processes have finished
        for j in jobs:
            j.join()
        
        
        e_KC    =   0.
        for k in range(self.cores):
            e_KC    +=  e_KCm[k]
        
        return e_KC 
        
    
    def energy_on_posArr(self, k, split_len, pos_arr, e_KCm, \
                             posits, pbc, cell, layer_neighbors, 
                             chem_symbs, posits_ext, neighbor_layer_inds2, \
                             layer_indices, params, cutoff, n, map_seqs, calcet, natoms):
        
        for l, ri in enumerate(pos_arr):
            i   =   l  + split_len 
            if chem_symbs[i] == 'C':
                ni              =   local_normal(i, posits_ext, layer_neighbors)
                neigh           =   neighbor_layer_inds2[which_layer(i, layer_indices)[0]]
                
                for nset in neigh:
                    
                    if nset[0] == 'bottom':
                        norm_fac    =   1
                        ni          =   -1*ni
                    elif nset[0] == 'top':
                        norm_fac    =   -1   
                                         
                    
                    neigh_indices   =   nset[1]
                
                
                    for j in neigh_indices: 
                        if chem_symbs[j] == 'C' and calcet[i*natoms + j] == 0:
                            calcet[i*natoms + j]    =  1
                            calcet[i + j*natoms]    =  1
                            
                            rj              =   posits[j]
                            nj              =   local_normal(j, posits_ext, layer_neighbors)*norm_fac
                            
                            e, new_maps     =   get_potential_ij(ri, rj, ni, nj, posits, i, j, params, \
                                                         pbc, cell, cutoff, n, map_seqs) 
                    
                            if new_maps != None:    self.map_seqs   =   new_maps
                            
                            e_KCm[k]       +=  e
        
    