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
        left_ind    = []
        nl, bond    = args[0], args[1]
        left        = r[:,0] < r[:,0].min() + nl*bond + .1
        for i in range(len(left)):
            if left[i]: left_ind.append(i)
        return left, left_ind 
    elif group == 'touch':
        one_top       = np.logical_and(r[:,2] > r[:,2].max() - 1., r[:,0] < 1. )
        return one_top
    else:
        raise
    
def get_ind(positions, ind, *args):
    
    if ind == 'rend':
        chem_symb   =   args[0]
        top_d       =   args[1]
        lay_ind     =   find_layers(positions)[1][-top_d - 1]
        rend        =   [] 
        
        r0max       = 0
        ind_m       = None
        
        for ind in lay_ind:
            r = positions[ind]
            if r[0] > r0max and chem_symb[ind] == 'C': 
                r0max = r[0]
                ind_m = ind
        if ind_m != None: rend.append(ind_m)
    
        return rend
    
    elif ind == 'h':
        
        chem_symbs  =   args[0]
        hs          =   []
        for i, symb in enumerate(chem_symbs):
            if symb == 'H': hs.append(i)
        
        return hs
    
    elif ind == 'hrend':
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
    
    elif ind == 'left':
        r   =   positions
        left_ind    = []
        nl, bond    = args[0], args[1]
        left        = r[:,0] < r[:,0].min() + nl*bond + .1
        for i in range(len(left)):
            if left[i]: left_ind.append(i)
        return left_ind 
    
    elif ind == 'weak_rend': 
        
        chem_symb   =   args[0]
        false_inds  =   flatten(args[1])
        length      =   args[2]
        lay_ind     =   find_layers(positions.copy())[1]
        arend       =   [] 
        
        low_lim     =   positions[lay_ind[0], 0].max() - length    
        for inds in lay_ind:
            r0max   =   0
            ind_m   =   None
            for j in inds:
                r   =   positions[j]        
                if low_lim  <   r[0] and chem_symb[j] == 'C' and j not in false_inds: 
                    arend.append(j)
            
        return arend
    
    elif ind == 'bottom':
        chem_symb   =   args[0]
        false_inds  =   flatten(args[1])
        lay_ind     =   find_layers(positions.copy())[1]
        bottom      =   [] 
        
        for i, inds in enumerate(lay_ind):
            bot_lay =   []
            for j in inds:
                if chem_symb[j] == 'C' and j not in false_inds: 
                    bot_lay.append(j)
            if bot_lay != []:
                bottom.append(bot_lay)
            
        return bottom

    elif ind == 'top':
        top_d       =   args[0]
        left        =   args[1]
        ind_top     =   find_layers(positions.copy())[1][-top_d:]
        flattened   =   [val for sublist in ind_top for val in sublist]
        
        ind_top     =   strip(flattened, left)
        
        return ind_top

def flatten(arr):
    
    return [val for sublist in arr for val in sublist]

def strip(arr, strip_arr):
    
    new_arr = []
    for val in arr:
        if len(strip_arr) != 0:
            dont_add    =   False
            for nthis in strip_arr:
                if val == nthis: 
                    dont_add = True
        if not dont_add:
            new_arr.append(val)
        #else:
        #    new_arr.append(val)
    return new_arr

def take_inds(arr, inds, inverse = False):

    new_arr = []

    if inverse:
        for i, val in enumerate(arr):
            if i not in inds: new_arr.append(val)
        return new_arr

    else:
        for i, val in enumerate(arr):
            if i in inds: new_arr.append(val)
        return new_arr

    
def get_type_mask(top):
    
    tmask   =   np.zeros(len(top))
    
    for ia in range(len(top)): 
        if top[ia]: tmask[ia]   =   1 
        else:       tmask[ia]   =   2 
    
    return tmask
