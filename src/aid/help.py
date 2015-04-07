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
import numpy as np
import sys

import os.path

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

def passivate(atoms, ind, edge_type):

    from moldy.atom_groups import get_ind
    
    if ind == 'rend':
        ch_bond = 1.1
        rend_ind    = get_ind(atoms.positions.copy(), 'hrend')
        h_posits    = atoms.positions[rend_ind].copy()
        
        if edge_type == 'arm':
            h_posits[:,0] += ch_bond
        elif edge_type == 'zz':
            add_r           =   np.zeros((len(h_posits), 3))
            for i, r in enumerate(h_posits):
                for r2 in h_posits:
                    if abs(r[2] - r2[2]) < .1:
                        if abs(r[1] - r2[1]) < 1.6:
                            if r[1] < r2[1]:
                                add_r[i,:]    =   [ch_bond*np.cos(np.pi/6), -ch_bond*np.sin(np.pi/6), 0]      
                            elif r[1] > r2[1]:
                                add_r[i,:]    =   [ch_bond*np.cos(np.pi/6), +ch_bond*np.sin(np.pi/6), 0]   
                        elif abs(r[1] - r2[1]) > 1.6:
                            if r[1] < r2[1]:
                                add_r[i,:]    =   [ch_bond*np.cos(np.pi/6), ch_bond*np.sin(np.pi/6), 0]      
                            elif r[1] > r2[1]:
                                add_r[i,:]    =   [ch_bond*np.cos(np.pi/6), -ch_bond*np.sin(np.pi/6), 0]   
            h_posits   +=   add_r    
            
        for posit in h_posits:
            atoms += Atom('H', position = posit)
        return atoms

    if ind == 'lend':
        ch_bond = 1.1
        lend_ind    = get_ind(atoms.positions.copy(), 'hlend')
        h_posits    = atoms.positions[lend_ind].copy()
        
        if edge_type == 'arm':
            h_posits[:,0] -= ch_bond
        elif edge_type == 'zz':
            add_r           =   np.zeros((len(h_posits), 3))
            for i, r in enumerate(h_posits):
                for r2 in h_posits:
                    if abs(r[2] - r2[2]) < .1:
                        if abs(r[1] - r2[1]) < 1.6:
                            if r[1] < r2[1]:
                                add_r[i,:]    =   [-ch_bond*np.cos(np.pi/6), -ch_bond*np.sin(np.pi/6), 0]      
                            elif r[1] > r2[1]:
                                add_r[i,:]    =   [-ch_bond*np.cos(np.pi/6), +ch_bond*np.sin(np.pi/6), 0]   
                        elif abs(r[1] - r2[1]) > 1.6:
                            if r[1] < r2[1]:
                                add_r[i,:]    =   [-ch_bond*np.cos(np.pi/6), ch_bond*np.sin(np.pi/6), 0]      
                            elif r[1] > r2[1]:
                                add_r[i,:]    =   [-ch_bond*np.cos(np.pi/6), -ch_bond*np.sin(np.pi/6), 0]   
            h_posits   +=   add_r    
        
        
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
        pbc     =   (False, True, False)
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
        
        atoms = passivate(atoms, 'rend', edge_type)
        atoms = passivate(atoms, 'lend', edge_type)

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

def get_fileName(N, indent, taito, *args):
    
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
           
        if not taito:
            path_f      =   '/space/tohekorh/BendAndSlide/files/%s/' %('rebo+KC')   
        elif taito:
            path_f      =   ''
        
        mdrelax     =   path_f + 'BFGS_init=%i_v=%.2f%s%s.traj'  %(N, v, edge, cont)
        mdfile      =   path_f + 'md_N=%i_v=%.2f%s%s.traj'       %(N, v, edge, cont)
        mdlogfile   =   path_f + 'md_N=%i_v=%.2f%s%s.log'        %(N, v, edge, cont)
        return mdfile, mdlogfile, mdrelax

    if indent == 'fixTop' or indent == 'fixTop_T=10':
        cont        = ''
        v           =   args[0]
        edge        =   '_' + args[1]

        if len(args) == 3:
            cont    =   '_' + args[2]
           

        if not taito:
            path_f      =   '/space/tohekorh/BendAndSlide/files/taito/%s/N=%i_v=%i/%s/' %(indent, N, v, args[1])   
        elif taito:
            path_f      =   ''

         
        mdrelax     =   path_f + 'BFGS_init=%i_v=%.2f%s%s.traj'  %(N, v, edge, cont)
        mdfile      =   path_f + 'md_N=%i_v=%.2f%s%s.traj'       %(N, v, edge, cont)
        mdlogfile   =   path_f + 'md_N=%i_v=%.2f%s%s.log'        %(N, v, edge, cont)
        plotlogfile =   path_f + 'plot_log_N=%i_v=%.2f%s%s.log'  %(N, v, edge, cont)
        plotKClog   =   path_f + 'KC_log_N=%i_v=%.2f%s%s'        %(N, v, edge, cont)
        plotShiftlog=   path_f + 'Shift_log_N=%i_v=%.2f%s%s'     %(N, v, edge, cont)
        plotIlDistlog=  path_f + 'Il_dist_log_N=%i_v=%.2f%s%s'   %(N, v, edge, cont)
        
        return mdfile, mdlogfile, plotlogfile, plotKClog, plotShiftlog, plotIlDistlog, mdrelax
    
    
    else:
        raise
    

def get_traj_and_ef(mdfile, mdLogFile, cmdfile, cmdLogFile):
    
    traj        =   PickleTrajectory(mdfile, 'r')
    ef          =   np.loadtxt(mdLogFile)
    conc        =   False
    
    posits  =   np.empty(len(traj), dtype = 'object')
    for i in range(len(traj)):
        posits[i]   =   traj[i].positions
    
    if os.path.isfile(cmdLogFile):
        cef     =   np.loadtxt(cmdLogFile)
        cef[:,0]   +=   ef[-1,0]
        cef[:,1]   +=   ef[-1,1]
        conc    =   True
        print 'we concatenate the extented bending!'
        ef      =   np.concatenate((ef, cef))
        
    if os.path.isfile(cmdfile):
        ctraj   =   PickleTrajectory(cmdfile, 'r')
        posits  =   np.empty(len(ctraj) + len(traj), dtype = 'object')
        
        delta_z =   ctraj[0].positions[0][2] - traj[-1].positions[0][2] 
        
        ntraj   =   []
        for i in range(len(traj)):
            ntraj.append(traj[i])
            posits[i]   =   traj[i].positions
        for i in range(len(ctraj)):
            ntraj.append(ctraj[i])
            posits[len(traj) + i]       =   ctraj[i].positions # -   delta_z   
            posits[len(traj) + i][:,2] -=   delta_z
        traj    =   ntraj
    
    return traj, ef, conc, posits
    
    
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

def get_pairs(atoms):
    
    layer_indices   =   find_layers(atoms.positions)[1]
    positions       =   atoms.positions
    tops            =   np.zeros((len(positions), len(positions)))
    for i in range(len(layer_indices) -1):
        layer   =   layer_indices[i]
        for atom in layer:
            if atoms[atom].number == 6:
                ri          =   positions[atom]
                layer_up    =   layer_indices[i + 1]
                for atom_up in layer_up:
                    rj      =   positions[atom_up]
                    if np.linalg.norm(rj - ri) < 3.5:
                        tops[atom, atom_up] =   1
                        #tops[atom_up, atom] =   1
    return tops             

def get_pairs2(atoms):
    
    layer_indices   =   find_layers(atoms.positions)[1]
    positions       =   atoms.positions
    tops            =   np.zeros((len(positions), len(positions)))
    bots            =   np.zeros((len(positions), len(positions)))
    
    for i in range(len(layer_indices) - 1):
        layer   =   layer_indices[i]
        for atom in layer:
            if atoms[atom].number == 6:
                ri          =   positions[atom]
                layer_up    =   layer_indices[i + 1]
                for atom_up in layer_up:
                    rj      =   positions[atom_up]
                    if np.linalg.norm(rj - ri) < 3.5:
                        tops[atom, atom_up] =   1
                        bots[atom_up, atom] =   1
    return bots, tops    

def make_colormap(seq, alphas):
    import matplotlib.colors as mcolors
    
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq         = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    
    cdict = {'red': [], 'green': [], 'blue': [], 'alpha':[]}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    
        
    for item in alphas:
        r, alpha    =   item[0], item[1]
        cdict['alpha'].append([r, alpha, alpha])
    
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)   



def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    import matplotlib
    import matplotlib.pyplot as plt
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap      

