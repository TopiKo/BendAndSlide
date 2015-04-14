'''
Created on 30.3.2015

@author: tohekorh
'''
import numpy as np
from aid.help import find_layers, get_pairs, get_pairs2
from aid.help2 import extend_structure, local_normal, nrst_neigh

def get_shifts_old(traj):
    
    Rpix            =   np.array([[1,0,0], [0,1,-1],[0,1,1]])
    Rpiy            =   np.array([[1,0,1], [0,1,0], [-1,0,1]])
    
    pair_table      =   get_pairs(traj[0])
    posits_ext      =   extend_structure(traj[0].positions.copy(), \
                                             traj[0].get_pbc(), \
                                             traj[0].get_cell().diagonal())
    layer_neighbors =   nrst_neigh(traj[0].positions, posits_ext, 'layer')    
    layer_indices_f =   find_layers(traj[0].positions)[1]
    layer_indices   =   layer_indices_f[:-3]
    
    
    x_shift         =   np.empty(len(traj), dtype = 'object')
    il_dist_t       =   np.empty(len(traj), dtype = 'object')
    
    for ti, atoms in enumerate(traj):
        pav         =   []
        il_dist     =   []
        positions   =   atoms.positions
        posits_ext  =   extend_structure(positions.copy(), atoms.get_pbc(), atoms.get_cell().diagonal())

        for ilay, layer_ind in enumerate(layer_indices):
            for i in layer_ind:
                if 1 in pair_table[i]:
                    j   =   np.where(pair_table[i] == 1)[0][0]
                    ri  =   positions[i]
                    rij =   positions[j] - ri 
                    ni  =   local_normal(i, posits_ext, layer_neighbors)
                    nj  =   local_normal(j, posits_ext, layer_neighbors)
                    
                    rijdotni    =   np.dot(rij,ni)
                    rijdotnj    =   np.dot(rij,nj)
                    pij =   np.sqrt(np.linalg.norm(rij)**2 - rijdotni**2)
                    pji =   np.sqrt(np.linalg.norm(rij)**2 - rijdotnj**2)
                    
                    lr_k=   (ni[0]*rij[2] - ni[2]*rij[0])/np.abs(ni[0]*rij[2] - ni[2]*rij[0])
                    
                    pav.append([traj[0].positions[i][0], lr_k*(pij + pji)/2., i])
                    
                    
                    minD    =   10000.
                    take_k  =   None
                    for ik in layer_indices_f[ilay + 1]:
                        if atoms[ik].number == 6:
                            rk  =   positions[ik]
                            if np.linalg.norm(rk - ri) < minD:
                                take_k  =   ik    
                                minD    =   np.linalg.norm(rk - ri)    
                    rk      =   positions[take_k]
                    point   =   (rk + ri)/2.
                    
                    nk      =   -local_normal(take_k, posits_ext, layer_neighbors)
                    t1, t2  =   np.dot(Rpix, nk), np.dot(Rpiy, nk)
                    
                    lay     =   np.array([[ni[0], -t1[0], -t2[0]], \
                                          [ni[1], -t1[1], -t2[1]], \
                                          [ni[2], -t1[2], -t2[2]]])      
                    
                    h       =   np.linalg.solve(lay, rk - ri)[0]
                    
                    il_dist.append([point[0], point[2], h])
                    
                    #ia_sep  =      
                        
                        
        pav_n       =   np.zeros((len(pav), 3))
        il_dist_n   =   np.zeros((len(il_dist), 3))

        for i in range(len(pav)):
            pav_n[i,0]  =   pav[i][0] 
            pav_n[i,1]  =   pav[i][1]
            pav_n[i,2]  =   pav[i][2]

            il_dist_n[i,0]  =   il_dist[i][0] 
            il_dist_n[i,1]  =   il_dist[i][1]
            il_dist_n[i,2]  =   il_dist[i][2]

        
        x_shift[ti]     =   pav_n
        il_dist_t[ti]   =   il_dist_n
        
    return x_shift, il_dist_t, pair_table
       


def get_shifts(traj, positions_t):
    
    pair_table      =   get_pairs2(traj[0])[0]
    posits_ext      =   extend_structure(traj[0].positions.copy(), \
                                         traj[0].get_pbc(), \
                                         traj[0].get_cell().diagonal())
    layer_neighbors =   nrst_neigh(traj[0].positions, posits_ext, 'layer')    
    layer_indices_f =   find_layers(traj[0].positions)[1]
    #layer_indices   =   layer_indices_f[:-3]
    
    # minL, maxL      =   1000., -1 
    x_shift         =   np.empty(len(traj), dtype = 'object')
    il_dist_t       =   np.empty(len(traj), dtype = 'object')
    
    for ti, atoms in enumerate(traj):
        pav         =   []
        il_dist     =   []
        positions   =   positions_t[ti]
        posits_ext  =   extend_structure(positions.copy(), atoms.get_pbc(), atoms.get_cell().diagonal())
        print ti
        
        for ilay, layer_ind in enumerate(layer_indices_f):
            for i in layer_ind:
                #print ilay
                if 1 in pair_table[i] and atoms[i].number == 6:
                    j   =   np.where(pair_table[i] == 1)[0][0]
                    #print i,j 
                    
                    ri  =   positions[i]
                    rj  =   positions[j]
                    rij =   rj - ri 
                    ni  =   -local_normal(i, posits_ext, layer_neighbors)
                    nj  =   local_normal(j, posits_ext, layer_neighbors)
                    
                    rijdotni    =   np.dot(rij,ni)
                    rijdotnj    =   np.dot(rij,nj)
                    pij =   np.sqrt(np.linalg.norm(rij)**2 - rijdotni**2)
                    pji =   np.sqrt(np.linalg.norm(rij)**2 - rijdotnj**2)
                    
                    lr_k=   (ni[0]*rij[2] - ni[2]*rij[0])/np.abs(ni[0]*rij[2] - ni[2]*rij[0])
                    
                    
                    #if ilay != len(layer_indices_f) - 2: 
                    #    pav.append([ri[0], rj[0], ri[2], rj[2], lr_k*(pij + pji)/2., i])
                    
                    
                    
                    minD    =   10000.
                    take_k  =   None
                    
                    for ik in layer_indices_f[ilay - 1]:
                        if atoms[ik].number == 6:
                            rk  =   positions[ik]
                            if np.linalg.norm(rk - ri) < minD:
                                take_k  =   ik    
                                minD    =   np.linalg.norm(rk - ri)    
                    
                    rk      =   positions[take_k]
                    point   =   (rk + ri)/2.
                    
                    nk      =   local_normal(take_k, posits_ext, layer_neighbors)
                    # tangent vectors
                    t1      =   np.cross([1,1,0], nk)/np.linalg.norm(np.cross([1,1,0], nk))
                    t2      =   np.cross(t1, nk)
                    
                    lay     =   np.array([[ni[0], -t1[0], -t2[0]], \
                                          [ni[1], -t1[1], -t2[1]], \
                                          [ni[2], -t1[2], -t2[2]]])      
                    
                    h       =   np.linalg.solve(lay, rk - ri)[0]
                    
                    
                    if h < 4.2:
                        pav.append([ri[0], rj[0], ri[2], rj[2], lr_k*(pij + pji)/2., i])
                        il_dist.append([point[0], point[2], h])
                    
                                    
        pav_n       =   np.zeros((len(pav), 6))
        il_dist_n   =   np.zeros((len(il_dist), 3))

        for i in range(len(pav)):
            pav_n[i,0]  =   pav[i][0] 
            pav_n[i,1]  =   pav[i][1]
            pav_n[i,2]  =   pav[i][2]
            pav_n[i,3]  =   pav[i][3] 
            pav_n[i,4]  =   pav[i][4]
            pav_n[i,5]  =   pav[i][5]

        for i in range(len(il_dist)):

            il_dist_n[i,0]  =   il_dist[i][0] 
            il_dist_n[i,1]  =   il_dist[i][1]
            il_dist_n[i,2]  =   il_dist[i][2]

        
        #if maxL < np.max(il_dist_n[:,2]):
        #    maxL    =   np.max(il_dist_n[:,2])
        #if minL > np.min(il_dist_n[:,2]):
        #    minL    =   np.min(il_dist_n[:,2])
            
            
        x_shift[ti]     =   pav_n
        il_dist_t[ti]   =   il_dist_n
        
    return x_shift, il_dist_t #,  [minL, maxL]
       
def get_streches(traj, positions_t): 
    
    strech_t        =   np.empty(len(traj), dtype = 'object')
    
    posits_ext      =   extend_structure(traj[0].positions.copy(), \
                                         traj[0].get_pbc(), \
                                         traj[0].get_cell().diagonal())
    layer_neighbors =   nrst_neigh(traj[0].positions, posits_ext, 'layer')    
    
    
    pbc, cell       =   traj[0].get_pbc(), traj[0].get_cell().diagonal()
    
    for i, positions in enumerate(positions_t):
        strech_t[i] =   np.zeros((len(positions), 5))
        posits_ext  =   extend_structure(positions.copy(), pbc, cell)
        
        for ir, r in enumerate(positions):
            tang_vec    =   posits_ext[layer_neighbors[ir]] - r
            strech_t[i][ir, 0]  =   r[0] # x
            strech_t[i][ir, 1]  =   r[2] # 'y'
            
            for k, tv in enumerate(tang_vec):
                strech_t[i][ir, k + 2]  =   np.linalg.norm(tv)
                
    return strech_t