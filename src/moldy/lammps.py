'''
Created on 16.10.2014

@author: tohekorh
'''
# Make lammps input..

from aid.help import passivate, make_atoms_file 
from aid.help import make_graphene_slab, remove_atoms_from_bot_right_end
from atom_groups import get_mask, get_ind
from ase.visualize import view
import numpy as np

# fixed parameters
bond        =   1.39695
a           =   np.sqrt(3)*bond # 2.462
h           =   3.38 
dt          =   2               #units: fs
eps_dict    =   {1:[2.82, ''], 2:[45.44, '_Topi']}
idx_epsCC   =   1 
length      =   2*20            # slab length has to be integer*2
width       =   1               # slab width
N           =   2
fix_top     =   1

# SIMULATION PARAMS
M           =   int(1e4)        # number of moldy steps
d           =   3*h            # maximum separation
dz          =   d/M
dt          =   2               # fs
T           =   0.              # temperature
interval    =   10              # interval for writing stuff down
epsCC, eps_str \
            = eps_dict[idx_epsCC]  
    
# Make atoms
atoms       =   make_graphene_slab(a,h,width,length,N)[3]

top_ind     =   get_ind(atoms.positions.copy(), 'top', fix_top, [])  
rm_rend     =   get_ind(atoms.positions.copy(), 'weak_rend', ['C' for i in range(len(atoms))], [top_ind], 7)
atoms       =   remove_atoms_from_bot_right_end(atoms, fix_top, rm_rend)
atoms       =   passivate(atoms, 'rend')
# Atoms ready

# Define atom groups 
left_ind    =   get_mask(atoms.positions.copy(), 'left', 2, bond)[1]
top_ind     =   get_ind(atoms.positions.copy(), 'top', fix_top, left_ind)  
rend_ind    =   get_ind(atoms.positions.copy(), 'rend', atoms.get_chemical_symbols(), fix_top)
weak_rend_ind \
            =   get_ind(atoms.positions.copy(), 'weak_rend', atoms.get_chemical_symbols(), [top_ind, rend_ind], 4)
bottom_ind  =   get_ind(atoms.positions.copy(), 'bottom', atoms.get_chemical_symbols(), [top_ind, rend_ind, left_ind])
h_ind       =   get_ind(atoms.positions.copy(), 'h', atoms.get_chemical_symbols())

ind_dict            =   {}
for n in range(len(bottom_ind)):
    ind_dict['bottom%i'%n]  =   [n + 1, bottom_ind[n]]


ind_dict['top']     =   [n + 2, top_ind]
ind_dict['left']    =   [n + 3, left_ind]
ind_dict['rend']    =   [n + 4, rend_ind]
ind_dict['wrend']   =   [n + 5, weak_rend_ind]
ind_dict['h']       =   [n + 6, h_ind]

# Groups ready
view(atoms)

potential_airebo= "/home/tohekorh/workspace/LAMMPS/lammps-28Jun14/potentials/CH.airebo"

dump_file       =  "/space/tohekorh/BendAndSlide/lammps/bend_slab/fix_top/lammps.md"
data_file       =  "/space/tohekorh/BendAndSlide/lammps/bend_slab/fix_top/lammps.data"
atoms_data      =  "/space/tohekorh/BendAndSlide/lammps/bend_slab/fix_top/atoms.data"
dump_xyz        =  "/space/tohekorh/BendAndSlide/lammps/bend_slab/fix_top/lammps.xyz" 
lammps_in       =  "/space/tohekorh/BendAndSlide/lammps/bend_slab/fix_top/lammps.in"
group_symbs, groups_string =  make_atoms_file(atoms, atoms_data, ind_dict)

o=open(lammps_in,'w')
print>>o,"""
# (XXX)
clear
variable dump_file string %(dump_file)s
variable data_file string %(data_file)s
units metal 
boundary f p f 

read_data %(atoms_data)s

#pair_style     hybrid/overlay airebo 10. 1 0 lj/cut 10
#pair_coeff     * * airebo %(potential_airebo)s %(group_symbs)s
#pair_coeff     1 2 lj/cut .0028  3.4 
pair_style     airebo 10. 1 0 
pair_coeff     * * airebo %(potential_airebo)s %(group_symbs)s


mass * 12.01 
mass 6 1.0

### run
dump dump_xyz all xyz 100 %(dump_xyz)s

%(groups_string)s

timestep 0.002 

fix 1 all store/state 0 x y z          # fix to store initial x and z positions
fix 2 top setforce 0. 0. 0.            # Set force to zero
fix 3 top momentum 1 linear 1 1 1      # put momentum of all atoms with iD 1
                                       # to zero every 1th time step
fix 22 left setforce 0. 0. 0.          # Set force to zero
fix 32 left momentum 1 linear 1 1 1    # put momentum of all atoms with iD 1
                                       # to zero every 1th time step


variable zmiddle equal 103.380000
variable xleft equal 3.376613
# make variables initial x and z positions (just for notational comfort)
variable x0 atom f_1[1]
variable z0 atom f_1[3]


#
#   continuous bending with MD
#

fix fix_nve bottom0 nve
fix 7 bottom0 langevin 10.000000 10.000000 1.0 819 zero yes


# auxiliary variables

variable d  equal 5*ramp(0,1)

# calculate displacements for bending 
variable dz equal  -v_d 
variable dd equal  0.0


fix 8 rend move variable v_dd v_dd v_dz v_dd v_dd v_dd

 
thermo_style custom step temp v_d
thermo_modify flush yes
thermo 10

dump        2 all image 200 image.*.ppm type type &
        zoom 1.6 adiam 1.5 view 90 90
dump_modify    2 pad 5

run 10000 
""" %vars()
o.close()