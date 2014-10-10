'''
Created on 24.9.2014

@author: tohekorh
'''
import numpy as np
from aid.help import find_layers

def get_mask(positions, group, *args):
    
    r   =   positions
    
    if group == 'top':
        nlayers, h = args[0], args[1] 
        top        = r[:,2] > r[:,2].max() - nlayers*h + 1
        return top
    elif group == 'bottom':
        bottom    = r[:,2] < r[:,2].max() - 1.
        return bottom
    elif group == 'left':
        nl, bond    = args[0], args[1]
        left        = r[:,0] < r[:,0].min() + nl*bond + .1
        return left 
    elif group == 'touch':
        one_top       = np.logical_and(r[:,2] > r[:,2].max() - 1., r[:,0] < 1. )
        return one_top
    else:
        raise
    
def get_ind(positions, ind, *args):
    
    if ind == 'rend':
        chem_symb   =   args[0]
        zset = find_layers(positions.copy())[0]
        rend = [] 
        
        r0max = 0
        ind_m = None
        for ir, r in enumerate(positions):
            if chem_symb[ir] == 'C' and np.abs(r[2] - zset[-1]) < 0.1:
                if r[0] > r0max: 
                    r0max = r[0]
                    ind_m = ir
        if ind_m != None: rend.append(ind_m)
    
        return rend
    
    elif ind == 'arend':
        zset, lay_ind   =   find_layers(positions.copy())
        arend           =   [i for i in range(len(zset))]
        
        
        for i, inds in enumerate(lay_ind):
            r0max   = 0
            ind_m   = None
            for j, r in enumerate(positions[inds]):        
                if r[0] > r0max: 
                    r0max = r[0]
                    ind_m = inds[j] 
            if ind_m != None: arend[i] = ind_m
            else: raise
            
        return arend
    
def get_type_mask(top):
    
    tmask   =   np.zeros(len(top))
    
    for ia in range(len(top)): 
        if top[ia]: tmask[ia]   =   1 
        else:       tmask[ia]   =   2 
    
    return tmask
