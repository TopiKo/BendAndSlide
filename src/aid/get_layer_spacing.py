'''
Created on 27.10.2014

@author: tohekorh
'''
'''
Created on 8.10.2014

@author: tohekorh
'''
from ase.calculators.lammpsrun import LAMMPS
from ase.optimize import BFGS
from ase.visualize import view

from KC_potential_constraint import KC_potential
from KC_parallel import KC_potential_p

from help import make_graphene_slab
from scipy.optimize import fmin
import numpy as np

# fixed parameters
bond        =   1.39695
a           =   np.sqrt(3)*bond # 2.462
h           =   3.37 

length      =   2*1
width       =   1               # slab width
N           =   3
cell_h      =   h*N + 5

path        =   '/space/tohekorh/BendAndSlide/test_KC/'

edge        =   'zz'
pbc         =   (True, True, False)
#pbc         =   (False, True, False)

cell_add_x  =   0.
if not pbc[0]:
    cell_add_x  =   12 

if edge == 'zz':
    diag_cell   =   [a*length + cell_add_x, 3*bond, cell_h]
elif edge == 'arm':
    length      =   length*2
    diag_cell   =   [3*bond*length/2 + cell_add_x, a, cell_h]

def scale_epot(fac, *args):
    
    atoms   =   args[0]
    cell    =   atoms.get_cell()
    cell_hf =   fac*cell[2,2]
    
    atoms.set_cell([cell[0,0], cell[1,1], cell_hf], scale_atoms = True)

    ep      =   atoms.get_potential_energy()
    
    atoms.set_cell(cell, scale_atoms = True)
    return ep
        
def corrugationAndAdhesion(params):
    
    
    bond        =   params['bond']
    a           =   np.sqrt(3)*bond # 2.462
    h           =   params['h']
    width       =   params['width']
    length      =   params['length'] 
       
    
    # GRAPHENE SLAP
    atoms              =   make_graphene_slab(a, h, width, length, N, \
                                                   edge_type = edge, h_pass = not pbc[0])[3]
    
    
    atoms.set_pbc(pbc)
    atoms.set_cell(diag_cell)
    atoms.center()
    
    view(atoms)
    
    params['ncores']        =   2
    params['positions']     =   atoms.positions
    params['pbc']           =   atoms.get_pbc()
    params['cell']          =   atoms.get_cell().diagonal()
    params['ia_dist']       =   10.
    params['chemical_symbols']  =   atoms.get_chemical_symbols()
    
    add_kc               =   KC_potential_p(params)
    
    
    # CALCULATOR LAMMPS 
    if not pbc[0]:
        params_rebo      =   {'pair_style':'rebo',
                             'pair_coeff':['* * CH.airebo C H'],
                             'mass'      :['1 12.0', '2 1.0'],
                             'units'     :'metal', 
                             'boundary'  :'f p f'}    
    else:
        params_rebo      =   {'pair_style':'rebo',
                             'pair_coeff':['* * CH.airebo C'],
                             'mass'      :['* 12.01'],
                             'units'     :'metal', 
                             'boundary'  :'p p f'}    
    
    calc        =   LAMMPS(parameters=params_rebo) 
    atoms.set_calculator(calc)
    # END CALCULATOR
    
    atoms.set_constraint([add_kc])
       
    
    
    f_opm   =   fmin(scale_epot, 1.1, args=[atoms])[0]
    h_opm   =   f_opm*h
    print 'optimal height = %.5f' %h_opm
    
    dyn         =   BFGS(atoms, trajectory=path + '/bfgs_bilayer.traj')
    dyn.run(fmax=0.05)
    view(atoms)



params      =   {'bond': bond, 'a':a, 'h':h, 'width': width, 'length':length}  
    
corrugationAndAdhesion(params) 
 
