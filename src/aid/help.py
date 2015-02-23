'''
Created on 24.9.2014

@author: tohekorh
'''
from ase.lattice.hexagonal import Graphite
from ase.io.trajectory import PickleTrajectory
from numpy import *
from ase import *
from ase.visualize import view
from scipy.optimize import fmin_l_bfgs_b #, fmin, minimize
import sys

def saveAndPrint(atoms, traj = None, print_dat = False):    
    epot = atoms.get_potential_energy() #/len(atoms)
    ekin = atoms.get_kinetic_energy()   #/len(atoms)
    save_atoms  = get_save_atoms(atoms)
    if traj != None: traj.write(save_atoms)
    if print_dat:
        print ("Total Energy : Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  Etot = %.3feV" %
           (epot, ekin, ekin/(1.5*units.kB), epot+ekin) )
    return epot, ekin, epot + ekin, save_atoms

def get_save_atoms(atoms):
    positions = atoms.positions.copy()
    symbs     = atoms.get_chemical_symbols()
        
    new_atoms = Atoms()
    for i, pos in enumerate(positions):
        new_atoms += Atom(symbs[i], position = pos)    
    new_atoms.set_cell(atoms.get_cell())
    pbc =  atoms.get_pbc()
    new_atoms.set_pbc((pbc[0], pbc[1], pbc[2]))   
    return new_atoms    

def passivate(atoms, ind):

    from moldy.atom_groups import get_ind
    
    if ind == 'rend':
        ch_bond = 1.1
        rend_ind    = get_ind(atoms.positions.copy(), 'hrend')
        h_posits    = atoms.positions[rend_ind].copy()
        h_posits[:,0] += ch_bond
        for posit in h_posits:
            atoms += Atom('H', position = posit)
        return atoms

    if ind == 'lend':
        ch_bond = 1.1
        lend_ind    = get_ind(atoms.positions.copy(), 'hlend')
        h_posits    = atoms.positions[lend_ind].copy()
        h_posits[:,0] -= ch_bond
        for posit in h_posits:
            atoms += Atom('H', position = posit)
        return atoms

        
def remove_atoms_from_bot_right_end(atoms, fix_top, rm_ind):
    
    positions   =   atoms.positions.copy()
    symbs       =   atoms.get_chemical_symbols()
        
    new_atoms   =   Atoms()
    for i, pos in enumerate(positions):
        if i not in rm_ind:
            new_atoms += Atom(symbs[i], position = pos)    
    new_atoms.set_cell(atoms.get_cell())
    pbc     =  atoms.get_pbc()
    new_atoms.set_pbc(pbc)   
    return new_atoms    

def make_atoms_file(atoms, address, inds_dict):
    
    
    def which_group(i):
        for group in inds_dict:
            
            for j in inds_dict[group][1]:
                if i == j:
                    return inds_dict[group][0]
        raise 
    
    def groups():
        
        groups_string       =   ''
        for group in inds_dict:
            groups_string   +=   'group ' + group + ' type ' + str(inds_dict[group][0]) + ' \n'
        return groups_string
        
    m   =   len(inds_dict)
    
    f = open(address, 'w')
    
    l = atoms.get_cell().diagonal()
    
    print>>f, '\n\n\n'  
    print>>f, len(atoms), 'atoms'
    print>>f, m,'atom types'
    print>>f, '0 %.9f xlo xhi' %l[0]
    print>>f, '0 %.9f ylo yhi' %l[1]
    print>>f, '0 %.9f zlo zhi' %l[2]
    print>>f, '\n\nAtoms\n'
    
    for i, a in enumerate(atoms):
        r = a.position
        print>>f,i+1, int(which_group(i)), r[0], r[1], r[2]
   
    
    group_symbs = (m- 1)*'C ' + 'H'
    group_string =  groups()
    return group_symbs, group_string
    

def make_graphene_slab(a,h,x1,x2,N, \
                       edge_type = 'arm', h_pass = False):
    
    
    # make ML graphene slab
    if edge_type == 'arm':
        width   =   x1
        length  =   x2
        pbc     =   (False, True, False)
    elif edge_type == 'zz':
        width   =   x2
        length  =   x1*2
        pbc     =   (True, False, False)
    else:
        raise
    atoms       =   Graphite('C',latticeconstant=(a,2*h),size=(width,length,N) )
    
    l1,l2,l3    =   atoms.get_cell().diagonal()
    atoms.set_cell(atoms.get_cell().diagonal())
    atoms.set_scaled_positions(atoms.get_scaled_positions())
    
    
    atoms.rotate('z',pi/2)
     
    z = atoms.get_positions()[:,2]
    del atoms[ list(arange(len(z))[z>z.min()+(N-1)*h+h/2]) ]    
    
    atoms.set_cell( (l2,l1,l3) )
    atoms.center()
    atoms.translate((a/4, 0., 0.))
    
    atoms.set_scaled_positions( atoms.get_scaled_positions() )
    atoms.set_pbc(pbc)
    
    
    if edge_type == 'arm':
        W   =   atoms.get_cell().diagonal()[1]
        H   =   (N-1)*h
        L   =   atoms.get_positions()[:,0].ptp()/2
        atoms.center(vacuum=L,axis=0)

    elif edge_type == 'zz':
        atoms.rotate('z', pi/2)
        atoms.set_cell( (l1,l2,l3) )
        atoms.center()
        
        W   =   atoms.get_cell().diagonal()[1]
        H   =   (N-1)*h
        L   =   atoms.get_positions()[:,0].ptp()/2
        atoms.center(vacuum=L,axis=0)
    
    atoms.center(vacuum=L,axis=2)
    print 'structure ready'
    
    if h_pass:
        
        atoms = passivate(atoms, 'rend')
        atoms = passivate(atoms, 'lend')

        #ch_bond = 1.1
        #from moldy.atom_groups import get_ind
        #rend_ind    = get_ind(atoms.positions.copy(), 'hrend')
        #h_posits    = atoms.positions[rend_ind].copy()
        #h_posits[:,0] += ch_bond
        #for posit in h_posits:
        #    atoms += Atom('H', position = posit)
        
    return H,W,L,atoms

def deform(positions, top, dist, indent):
    z_pos  = positions[top == False][:,2]
    xunder = z_pos.max()
    newpos = positions.copy()
    
    if indent == 'adh':
        for i, r in enumerate(positions):
            if top[i]: newpos[i][2] = xunder + dist
    elif indent == 'corr':
        for i, r in enumerate(positions):
            if top[i]: newpos[i][0] = positions[i][0] + dist
    return newpos 

def get_nearest_ind(dat, val):
    sep_min = 1E10
    for i, dat_val in enumerate(dat):
        if abs(dat_val - val) < sep_min: 
            sep_min = abs(dat_val - val)
            ind_min = i
    return ind_min, dat[ind_min], sep_min   

def x2(x, *params):
    a,b,c = params
    return a*x**2 + b*x + c

def dx2(x, *params):
    a,b = params
    return 2*a*x + b

def der(x,y):
    der = zeros(len(x))
    for i in range(1, len(x)):
        dx = x[i] - x[i-1]   
        dy = y[i] - y[i-1]   
        der[i] = dy/dx
    return der


def move_atoms(atoms, scale, indent):
    
    if indent == 'space':
        posits = atoms.positions.copy()
        zmin = min(posits[:,2])
        for i, r in enumerate(posits):
            posits[i][2] = (r[2] - zmin)*scale + zmin
        atoms.set_positions(posits)
        return atoms
    elif indent == 'width':
        posits = atoms.positions.copy()
        wmin = min(posits[:,1])
        l1,l2,l3 = atoms.get_cell().diagonal()
        atoms.set_cell((l1, l2*scale, l3))
          
        for i, r in enumerate(posits):
            posits[i][1] = (r[1] - wmin)*scale + wmin
        atoms.set_positions(posits)
        return atoms
    

def optimize_width_and_spacing(atoms):
    
    def e(scales, *args):
        [scale_s, scale_w] = scales
        atoms = args[0]
        init_pos, init_cell = atoms.positions.copy(), atoms.get_cell().copy()
        ep    = atoms.get_potential_energy()
        atoms = move_atoms(atoms, scale_s, 'space')
        atoms = move_atoms(atoms, scale_w, 'width')
        ep    = atoms.get_potential_energy()
        atoms.set_cell(init_cell) 
        atoms.set_positions(init_pos)
        return ep
    
    [scale_s, scale_w] = fmin_l_bfgs_b(e, [1., 1.], approx_grad=True, epsilon = 1E-6, \
              args=(atoms, None), bounds = [(0.9, 1.1), (0.9, 1.1)])[0]
    
    atoms = move_atoms(atoms, scale_s, 'space')
    atoms = move_atoms(atoms, scale_w, 'width')
    
    return atoms

def optimize_spacing(atoms_orig):
    pos_atoms_orig = atoms_orig.positions.copy()
    l1,l2,l3  = atoms_orig.get_cell().diagonal()
 
    def e(scale):
        zmin = min(pos_atoms_orig[:,2])
        pos_atoms = pos_atoms_orig.copy()
        for i, r in enumerate(pos_atoms):
            pos_atoms[i][2] = (r[2] - zmin)*scale + zmin
        atoms_orig.set_positions(pos_atoms)
        ep = atoms_orig.get_potential_energy()
        return ep

    sc = fmin(e, 1., disp = False)[0]
    #view(atoms_orig)
    for i, r in enumerate(pos_atoms_orig):
        atoms_orig.positions[i][2] = r[2]*sc
    
    return atoms_orig

def find_layers(positions):
    
    zs = positions[:,2]
    layers = [[1, zs[0]]]
    for z in zs:
        add = True
        for layer in layers:
            if abs(z - layer[1]) < .5:
                add = False
                l_idx   = layers.index(layer)
                n       = layer[0]
                layers[l_idx][1] = (n*layer[1] + z)/(n + 1)
                layers[l_idx][0] = n + 1
                
        if add: layers.append([1, z])
    
    layers_new = []
    for layer in layers:
        layers_new.append(layer[1])
    
    layers = layers_new    
    layers.sort()
    inds_L  = []
    for layer in layers:
        inds = []
        for ir, r in enumerate(positions):    
            z   =   r[2]
            if abs(z - layer) < 0.1:
                inds.append(ir)
        inds_L.append(inds)
                
    return layers, inds_L

def get_fileName(N, indent, *args):
    
    if indent == 'tear_E':
        potential   =   args[0]
        epsCC       =   args[1]
        cont        =   args[2]
        
        path_f      =   '/space/tohekorh/BendAndSlide/files/%s/' %potential
        
        mdfile      =   path_f + 'md_N=%i_epsCC=%.2f%s.traj'    %(N, epsCC, cont)
        mdlogfile   =   path_f + 'md_N=%i_epsCC=%.2f%s.log'     %(N, epsCC, cont)
        return mdfile, mdlogfile
    
    if indent == 'tear_E_rebo+KC':
        cont        = ''
        if len(args) == 1:
            cont        =   '_' + args[0]
    
        path_f      =   '/space/tohekorh/BendAndSlide/files/%s/' %('rebo+KC')
    
        mdrelax     =   path_f + 'BFGS_init=%i%s.traj'  %(N, cont)
        mdfile      =   path_f + 'md_N=%i%s.traj'       %(N, cont)
        mdlogfile   =   path_f + 'md_N=%i%s.log'        %(N, cont)
        return mdfile, mdlogfile, mdrelax
    
    if indent == 'tear_E_rebo+KC_v':
        cont        = ''
        v           =   args[0]
        edge        =   '_' + args[1]

        if len(args) == 3:
            cont    =   '_' + args[2]
           
        path_f      =   '/space/tohekorh/BendAndSlide/files/%s/' %('rebo+KC')   
         
        mdrelax     =   path_f + 'BFGS_init=%i_v=%.2f%s%s.traj'  %(N, v, edge, cont)
        mdfile      =   path_f + 'md_N=%i_v=%.2f%s%s.traj'       %(N, v, edge, cont)
        mdlogfile   =   path_f + 'md_N=%i_v=%.2f%s%s.log'        %(N, v, edge, cont)
        return mdfile, mdlogfile, mdrelax
    
    
    else:
        raise
    
def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is one of "yes" or "no".
    """
    valid = {"yes":True,   "y":True,  "ye":True,
             "no":False,     "n":False}
    if default == None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "\
                             "(or 'y' or 'n').\n")

def collect_files(dyn, Nset = None):
    
    if dyn == 'traj': idx = 0
    elif dyn == 'log': idx = 1
    else: raise
    
    data = []
    data_take = []
   
    for N in range(1, 20):
        fname = get_fileName(N, 2.82, 'tear_E')[idx]
        try:
            if idx == 1: data.append([N, loadtxt(fname)])
            elif idx == 0: data.append([N, PickleTrajectory(fname)])

        except IOError as e:
            continue
    
    for dat in data:
        if Nset == None:
            question = 'Take data? N = %i' %dat[0]
            if query_yes_no(question, 'no'): data_take.append(dat)
        elif Nset != None:
            if dat[0] in Nset: data_take.append(dat)
    
    data_take.sort()
    
    
    return data_take
            
