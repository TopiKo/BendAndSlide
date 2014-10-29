'''
Created on 28.10.2014

@author: tohekorh
'''
import numpy as np
from aid.help import find_layers

def extend_structure(ia_length, posits, pbc, cell):
            
    posits_ext  =   posits.copy()
    for i in range(3):  
        if pbc[i]:
            n           =   int(1 + (ia_length + 5)/cell[i]) 
            
            add_posits  =   np.zeros((2*n*len(posits_ext), 3))
            m           =   0 
            kset        =   np.concatenate((range(-n,0), range(1,n + 1)))
            for k in kset:
                for pos in posits_ext:
                    add_posits[m]       =   pos
                    add_posits[m][i]   +=   k*cell[i]
                    m += 1
                
            posits_ext = np.concatenate((posits_ext, add_posits))
            
    return posits_ext   

def which_layer(i, layer_inds):
    
    for j, inds in enumerate(layer_inds):
        if i in inds: return j, layer_inds[j]
    else: raise
    
def nrst_neigh(posits, posits_ext, key, *args):
    
    if key == 'layer':
        d               =   1.5
        n               =   len(posits_ext)
        layer_inds_ext  =   find_layers(posits_ext)[1]
        neighbours      =   np.empty(n, dtype = object)
        for i in range(n):
            
            inds            =   which_layer(i, layer_inds_ext)[1]
            r               =   posits_ext[i]
            neighbours[i]   =   []
            for ind in inds:
                if ind != i:
                    if np.linalg.norm(r - posits_ext[ind]) < d:
                        neighbours[i].append(ind)
        
        return neighbours

    elif key == 'interact':
        
        d               =   args[0]
        n               =   len(posits)
        #layer_inds_ext  =   find_layers(posits_ext)[1]
        layer_inds_ext  =   args[1]
        neighbours_ia   =   np.empty(n, dtype = object)
        
        for i in range(n):
            
            layer       =   which_layer(i, layer_inds_ext)[0]
            r           =   posits_ext[i]
            neighbours_ia[i]   =   []
            
            if layer + 1 < len(layer_inds_ext):
                inds    =   layer_inds_ext[layer + 1]
                for ind in inds:
                    if np.linalg.norm(r - posits_ext[ind]) < d:
                        neighbours_ia[i].append(ind)
            if 0 <= layer - 1:
                inds    =   layer_inds_ext[layer - 1]
                for ind in inds:
                    if np.linalg.norm(r - posits_ext[ind]) < d:
                        neighbours_ia[i].append(ind)

        
        return neighbours_ia

    
        