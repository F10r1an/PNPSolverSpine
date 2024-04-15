import numpy as np
import gmpy2
from gmpy2 import mpfr
from math_mpfr import mpfr_abs
import figures as fgs
import time

#############################################################################

def solve_domain_with_fixed_dPhi(domain, dPhiIn, dPhiExt):
    # set potential
    domain.set_delta_phi_in(dPhiIn)
    domain.set_delta_phi_ext(dPhiExt)
    domain.solve_domain()
    return domain

def find_max_index_finite_element(domain, quantity, position=None):    
    """
    find the element of a list of mpfr-numbers that has the highest index but is not inf or nan
    quantity: sets the list to be analyzed to be cumulative charge or el. potential
    """
    # last point is default
    # TODO: add options that only intracellular space can be analyzed
    if position == None:
        position = domain.res_in + domain.res_mem + domain.res_out - 3
    
    if quantity == 'q':
        x = domain.cumulative[:position+1]
    elif quantity == 'v':
        x = domain.electric_potential[:position+1]
    
    # find element in list with highest index that is still finite             
    max_value = 0
    for i, x_i in enumerate(x):
        if gmpy2.is_finite(x_i):
            max_value = x_i
        else:
            break 
    
    if quantity == 'v':   
        return max_value + domain.membrane_potential
    else:
        return max_value


def get_new_boundaries(d_phi_new, f_new, d_phi_bnds_prev, f_prev, domain):
    """
    compute interval for next step in bisection algorithm
    """
    if f_prev[0] * f_new < 0:
        b_vals = [f_prev[0], f_new]
        d_phi_bnds = [d_phi_bnds_prev[0], d_phi_new]
    elif f_prev[1] * f_new < 0:
        b_vals = [f_new, f_prev[1]]
        d_phi_bnds = [d_phi_new, d_phi_bnds_prev[1]]
    else:
        raise Exception('Error in method get_new_boundaries() \n run ID: ', domain.experiment_id)
        
    return d_phi_bnds, b_vals

def get_initial_biscetion_interval(domain, where, quantity, d_phi_in_0=None, max_exp=200):
    """
    find a suitable starting interval for the bisection optimization
    domain: instance of spine_neck or spine_head
    where: 'in' or 'ext' (string variable); states whether d_phi_in or d_phi_ext boundaries should be found
    d_phi_in: in case that "where"-argument is set to "ext" then the value d_phi_in hast to be specified additionally
    quantity: 'v' or 'q' depending whether the voltage or cumulative charge is target for optimization
    
    
    """
    
    # print('test method', d_phi_in_0)
    
    if type(d_phi_in_0) != type(mpfr('0')):
        if d_phi_in_0 != None:
            raise ValueError('d_phi_in_0 should be of type mpfr')
    
    if where == 'ext':
        assert d_phi_in_0 != None, 'd_phi_in is not set'
    elif where != 'in':
        raise ValueError('where argument can be "in" or "ext"')
    
    if where == 'in':
        # first index of extracell space right after membrane
        pos_i = domain.res_in + domain.res_mem - 1
    elif where == 'ext':
        pos_i = domain.res_in + domain.res_mem + domain.res_out - 3  # last point of extracell space
    
    # find a numerical solution that exists i.e. the potential does not diverge to inifinity 
    
    # interval for d_phi_in
    # when d_phi_in is set to zero the solution does not diverge in the intracellular space or the 
    # membrane
    # but d_phi_in=0 is an upper boundary for d_phi_in because
    # there has to be an excess of positive charges in the intracellular space to compensate the 
    # strongly negative surface charge.
    # search algoriithm for boundaries for d_phi_in
    # 1. set d_phi_in = 0 as upper boundary and evaluate sign of voltage v or cumulative q
    # 2. approach d_phi_in = 0 starting from d_phi_in = -1 by increasing the exponent n in -10^-n
    #    increase n until the sign of v/q has the same sign at the exterior membrane boundary as 
    #    for d_phi_in = 0 -> n_same_sign
    # 3. check if for n_same_sign - 1 the q/v change their sign (this is expected) 
    # 4. return interval
    
    if where == 'in':
        # 1. get reference sign for d-phi_in = 0
        sign_ref = 'NaN'
        d_phi_in = mpfr('0')
        d_phi_ext = mpfr('0')  # has no effect and can be set to zero
        # get solution when d_phi_in = 0
        # and record sign of quantity at ext. membrane surface
        domain = solve_domain_with_fixed_dPhi(domain, dPhiIn=d_phi_in, dPhiExt=d_phi_ext)
        if quantity == 'q':
            q = domain.cumulative[pos_i]
        elif quantity == 'v':
            q = domain.electric_potential[pos_i]
        sign_ref = gmpy2.sign(q)
        exp_ref = '-inf'  # -10^-inf = 0
        
        
        # 2. approach d_phi_in = 0 from -1
        exp_neg_side = 'NaN'        
        for e in range(max_exp): # limit range of search
            # set d_phi and solve
            d_phi_in = mpfr('-1') * gmpy2.exp10(-e)  # approach from -1 towards 0
            d_phi_ext = mpfr('0')
            domain = solve_domain_with_fixed_dPhi(domain, dPhiIn=d_phi_in, dPhiExt=d_phi_ext)
            # evaluate at membrane or exterior boundary
            if quantity == 'q':
                q = domain.cumulative[pos_i]
            elif quantity == 'v':
                q = domain.electric_potential[pos_i]
            # test if finite
            if gmpy2.is_finite(q):
                # print(np.float(q), gmpy2.sign(q))
                # print('solution found for 10^-', e)
                #3. check if q has same sign as q for d_phi_in =0
                sign_neg_side = gmpy2.sign(q)

                if sign_ref * sign_neg_side == 1:    
                    exp_neg_side = e - 1 # decrease e, in that case the sign is different again
                    break
            else:
                # print('no solution for 10^-', e)
                pass
    
        # finally check found d_phi_in
        d_phi_in = mpfr('-1') * gmpy2.exp10(-exp_neg_side)  # previously found exponent
        d_phi_ext = mpfr('0')
        domain = solve_domain_with_fixed_dPhi(domain, dPhiIn=d_phi_in, dPhiExt=d_phi_ext)
        # evaluate at membrane or exterior boundary
        if quantity == 'q':
            q = domain.cumulative[pos_i]
        elif quantity == 'v':
            q = domain.electric_potential[pos_i]

        sign_neg_side = gmpy2.sign(q)
        assert sign_ref * sign_neg_side == -1, 'starting points for bisection can not be found in this interval'
        
        print('Initial boundaries for bisection successfully found. (where, quantity, id)=', where, quantity, domain.get_experiment_id())
                
        return (d_phi_in, mpfr('0'))
    
    
    # approach from both sides to zero starting at +1 and -1 -> Increase the exponents e +-10^-e
    # search through list of d_phi_ext
    # find the two neighbour elements in the list where sign of voltage or charge changes
    # from positive to negative or the other way round
    if where == 'ext':
        # create lists of d_phi_in and d_phi_ext
        # d_phi_in is fixed
        # d_phi_ext gets varied for search
        signs = [mpfr('-1')] * max_exp + ['NaN'] + [mpfr('1')] * max_exp
        exponents = list(range(-1, -1*max_exp-1, -1)) + ['-inf'] + list(range(-1*max_exp, 0, 1))
        evals = ['NaN'] * (2*max_exp+1)
        
        d_phi_ext_list = ['NaN'] * (2*max_exp+1)
        d_phi_in_list = ['NaN'] * (2*max_exp+1)
        # fill lists
        for i in range(max_exp*2+1):

                d_phi_in_list[i] = d_phi_in_0
                if i == max_exp:
                    d_phi_ext_list[i] = mpfr('0')
                else:
                    d_phi_ext_list[i] = signs[i] * gmpy2.exp10(exponents[i])      
        # compute sign if soulution is finite        
        for i in range(max_exp * 2 + 1):
            d_phi_in = d_phi_in_list[i]
            d_phi_ext = d_phi_ext_list[i]
        
            solve_domain_with_fixed_dPhi(domain, dPhiIn=d_phi_in, dPhiExt=d_phi_ext)
            
            f = find_max_index_finite_element(domain, quantity, position=pos_i)
            
            is_finite = gmpy2.is_finite(f)
            if is_finite == True:
                evals[i] = gmpy2.sign(f)
            else: evals[i] = 'NaN'
        
        #print('################')
        #print(evals)
        
        
        # find inicies in evals where sign changes
        evals=np.array(evals)
        ref_first = evals[0]
        ref_last = evals[-1]
        print('ref', ref_first, ref_last)
        i_min = np.max(np.where(evals==ref_first))
        i_max = np.min(np.where(evals==ref_last))
        
        assert i_max - i_min == 1, 'error in (where, quantity, id) ='+where+quantity+str( domain.get_experiment_id())
        
        d_phi_ext_min = d_phi_ext_list[i_min]
        d_phi_ext_max = d_phi_ext_list[i_max]
        
        return (d_phi_ext_min, d_phi_ext_max)

        

   
        
        

def get_bisection_values(domain, d_phi_in_1, d_phi_in_2, d_phi_ext_1, d_phi_ext_2, quantity, position):
    """
    evaluate function at initial bisection interval
    """
    domain = solve_domain_with_fixed_dPhi(domain, d_phi_in_1, d_phi_ext_1)
    f_1 = find_max_index_finite_element(domain, quantity, position)

    
    domain = solve_domain_with_fixed_dPhi(domain, d_phi_in_2, d_phi_ext_2)
    f_2 = find_max_index_finite_element(domain, quantity, position)

   
    if f_1 * f_2 < 0:
        return [f_1, f_2] 
    else:
        print('bad bisection boundaries')
        print(f_1, f_2)
        raise Exception('Error in method get_bisection_values() \n run ID: ', domain.experiment_id)

def shoot(domain, d_phi_in_A, d_phi_in_B, d_phi_ext_A, d_phi_ext_B, max_repetitions, report_modulo=10, logfile=None):
   
    v_mem = domain.membrane_potential
    
    
    # function values for bisection algorithm
    # [v_A, v_B]
    domain = solve_domain_with_fixed_dPhi(domain, d_phi_in_A, d_phi_ext_A)
    v_A = domain.electric_potential[-1] + v_mem
    domain = solve_domain_with_fixed_dPhi(domain, d_phi_in_B, d_phi_ext_B)
    v_B = domain.electric_potential[-1] + v_mem
    v_vals = [v_A, v_B]
    
    d_phi_ext_bnds = [mpfr('-0.2'), mpfr('0.2')]
    #d_phi_ext_bnds = [d_phi_ext_A, d_phi_ext_B]
    d_phi_in_bnds = [d_phi_in_A, d_phi_in_B]
    
    for rep in range(max_repetitions):
        if rep % report_modulo == 0:
            report = 'shooting optimization: round ' +  str(rep + 1) + ' of ' + str(max_repetitions) + ' in run ID: '  + str(domain.experiment_id)
            if logfile == None:
                print(report)
            else:
                global start_time
                print(report)
                logfile = open('logfile.txt', 'a')
                logfile.write(report)
                proc_time = (time.time() - start_time) / 60 ## minutes
                logfile.write('\n time: ' + str(proc_time)) 
                logfile.write('\n \n')
                logfile.close()
        
        d_phi_in_new = (d_phi_in_bnds[0] + d_phi_in_bnds[1]) / mpfr('2')
        
        d_phi_ext_new = find_root_exterior_boundary(domain, quantity='q', 
                                                    d_phi_in=d_phi_in_new, 
                                                    d_phi_ext_boundaries=d_phi_ext_bnds, 
                                                    max_repetitions=max_repetitions,
                                                    print_output=False)
        
        domain = solve_domain_with_fixed_dPhi(domain, d_phi_in_new, d_phi_ext_new)
        
        v_new = find_max_index_finite_element(domain, quantity='v') 
        
        d_phi_in_bnds, v_vals = get_new_boundaries(d_phi_in_new, v_new, d_phi_in_bnds, v_vals, domain)
        
        
        #print('###############')
        #print(float(d_phi_in_bnds[0]), float(d_phi_in_bnds[1]))
        #print(float(v_vals[0]), float(v_vals[1]))
        #print('###############')

        

    d_phi_in = (d_phi_in_bnds[0]  + d_phi_in_bnds[1] ) / mpfr('2')
    d_phi_ext= find_root_exterior_boundary(domain, quantity='q', 
                                                    d_phi_in=d_phi_in, 
                                                    d_phi_ext_boundaries=d_phi_ext_bnds, 
                                                    max_repetitions=max_repetitions,
                                                    print_output=False)
    print('>>>')
    print('shooting optimization successful in run (ID): ', domain.experiment_id)
    print('d_phi_in: ', float(d_phi_in))
    print('d_phi_ext: ', float(d_phi_ext))
    print('<<<')
    return d_phi_in, d_phi_ext

    
def find_root_intracell_boundary(domain, quantity, d_phi_in_boundaries, d_phi_ext=mpfr('0'), max_repetitions=1000, offset=mpfr('0')):
    
    i_pos = domain.res_in
    print(i_pos)
    #TODO

def find_root_after_membrane(domain, quantity, d_phi_in_boundaries, d_phi_ext=mpfr('0'), max_repetitions=1000, 
                             offset=mpfr('0'), print_output=True):
    
    # first index of extracell space right after membrane
    i_pos = domain.res_in + domain.res_mem - 1
    
    # compute function values for both starting points of bisection algorithm
    # and check if starting points are suitable for optimization
    # f = [f1, f2]
    b_vals = get_bisection_values(domain, 
                                  d_phi_in_1=d_phi_in_boundaries[0], 
                                  d_phi_in_2=d_phi_in_boundaries[1], 
                                  d_phi_ext_1=d_phi_ext, 
                                  d_phi_ext_2=d_phi_ext, 
                                  quantity=quantity, 
                                  position=i_pos)
    # bisection interval for function arguments
    d_phi_in_bnds = d_phi_in_boundaries
    
    # bisection optimization
    for rep in range(max_repetitions):
        d_phi_in_new = (d_phi_in_bnds[0] + d_phi_in_bnds[1]) / mpfr('2')
        
        domain = solve_domain_with_fixed_dPhi(domain, d_phi_in_new, d_phi_ext)        
        b_new = find_max_index_finite_element(domain, quantity, i_pos)
        
        d_phi_in_bnds, b_vals = get_new_boundaries(d_phi_in_new, b_new, d_phi_in_bnds, b_vals, domain)
        
    d_phi_in = (d_phi_in_bnds[0]  + d_phi_in_bnds[1] ) / mpfr('2')
    b_val = (b_vals[0]  + b_vals[1] ) / mpfr('2')
    if print_output== True:
        print('>>>')
        print('d_phi_in successfully optimized at exterior boundary of membrane in run (ID): ', domain.experiment_id)
        print('d_phi_in: ', float(d_phi_in))
        print('optimisation quantity ', quantity, ' = ', float(b_val))
        print('<<<')
    return d_phi_in


def find_root_exterior_boundary(domain, quantity, d_phi_in, d_phi_ext_boundaries, max_repetitions=1000, 
                                offset=mpfr('0'), print_output=True):
    
    i_pos = domain.res_in + domain.res_mem + domain.res_out - 3

    # compute function values for both starting points of bisection algorithm
    # and check if starting points are suitable for optimisation
    # f = [f1, f2]
    b_vals = get_bisection_values(domain, d_phi_in_1=d_phi_in, d_phi_in_2=d_phi_in, 
                         d_phi_ext_1=d_phi_ext_boundaries[0], 
                         d_phi_ext_2=d_phi_ext_boundaries[1], 
                         quantity=quantity, position=i_pos)
    # bisection interval for function arguments
    d_phi_ext_bnds = d_phi_ext_boundaries
    
    for rep in range(max_repetitions):
        d_phi_ext_new = (d_phi_ext_bnds[0] + d_phi_ext_bnds[1]) / mpfr('2')
        
        domain = solve_domain_with_fixed_dPhi(domain, d_phi_in, d_phi_ext_new)        
        b_new = find_max_index_finite_element(domain, quantity, i_pos)
    
        d_phi_ext_bnds, b_vals = get_new_boundaries(d_phi_ext_new, b_new, d_phi_ext_bnds, b_vals, domain)
        
    d_phi_ext = (d_phi_ext_bnds[0]  + d_phi_ext_bnds[1] ) / mpfr('2')
    b_val = (b_vals[0]  + b_vals[1] ) / mpfr('2')
    if print_output == True:
        print('>>>')
        print('d_phi_ext successfully optimized at exterior boundary in run (ID): ', domain.experiment_id)
        print('d_phi_ext: ', float(d_phi_ext))
        print('optimization quantity ', quantity, ' = ', float(b_val))
        print('<<<')
    return d_phi_ext

###########
# note in the four methods below:
# A -> point P1 in figure
# B -> point P2 in figure
############

def find_d_phi_in_A(domain, max_repetitions,):
    # worked well for head r = 20,40,80 & 160 nm
    # problems with r = 320
    # d_phi_in_boundaries = [mpfr('-1e-7'), mpfr('1.e-7')]
    # bnds for r = 320
    #d_phi_in_boundaries = [mpfr('-1e-4'), mpfr('1.e-4')]
    
    d_phi_in_boundaries = get_initial_biscetion_interval(domain, where='in', quantity='q')
    
    
    d_phi_in = find_root_after_membrane(domain, quantity='q', 
                             d_phi_in_boundaries=d_phi_in_boundaries,
                             max_repetitions=max_repetitions)
    return d_phi_in
    
def find_d_phi_ext_A(domain, d_phi_in, max_repetitions):
    #d_phi_ext_boundaries = [mpfr('-0.2'), mpfr('0.2')]
    d_phi_ext_boundaries = get_initial_biscetion_interval(domain, where='ext', quantity='q',
                                                          d_phi_in_0=d_phi_in)
    d_phi_ext = find_root_exterior_boundary(domain, quantity='q', 
                                            d_phi_in=d_phi_in,
                                            d_phi_ext_boundaries=d_phi_ext_boundaries)
    return d_phi_ext

def find_d_phi_in_B(domain, max_repetitions):
    # worked well for head r = 20,40,80 & 160 nm
    # problems with r = 320
    # d_phi_in_boundaries = [mpfr('-1e-7'), mpfr('1.e-7')]
    # bnds for r = 320
    # d_phi_in_boundaries = [mpfr('-1e-4'), mpfr('1.e-4')]
    d_phi_in_boundaries = get_initial_biscetion_interval(domain, where='in', quantity='v')
    d_phi_in = find_root_after_membrane(domain, quantity='v', 
                             d_phi_in_boundaries=d_phi_in_boundaries,
                             max_repetitions=max_repetitions)
    return d_phi_in

def find_d_phi_ext_B(domain, d_phi_in, max_repetitions):
    # d_phi_ext_boundaries = [mpfr('-0.2'), mpfr('0.2')]
    d_phi_ext_boundaries = get_initial_biscetion_interval(domain, where='ext', quantity='q',
                                                          d_phi_in_0=d_phi_in)
    d_phi_ext = find_root_exterior_boundary(domain, quantity='q', 
                                            d_phi_in=d_phi_in,
                                            d_phi_ext_boundaries=d_phi_ext_boundaries,
                                            max_repetitions=max_repetitions)
    return d_phi_ext


def test_optimization_boundaries(domain, d_phi_in_bnds, d_phi_ext_bnds):

    eps_q = mpfr('1.e-5')  # tolerance for cumulative charge
    v_mem = domain.membrane_potential
    
    # references A denotes point P1 in figure
    d_phi_in_A = d_phi_in_bnds[0]
    d_phi_in_B = d_phi_in_bnds[1]
    d_phi_ext_A = d_phi_ext_bnds[0]
    d_phi_ext_B = d_phi_ext_bnds[1]
    
    # solve at point A
    domain.set_delta_phi_in(d_phi_in_A)
    domain.set_delta_phi_ext(d_phi_ext_A)
    domain.solve_domain()
    phi_A = domain.electric_potential[-1] + v_mem
    q_A = domain.cumulative[-1]
    
    # potential has to be smaller than 0
    t1 = (phi_A < mpfr('0'))
    if t1 == False:
        print('volage at A is not smaller than zero')
        print('V = ', phi_A)
    # cumulative charge is zero
    t2 = (mpfr_abs(q_A) < eps_q)
    if t2 == False:
        print('cumulative charge at A is not zero')
        print('Q = ', q_A)
    
    # solve at point B denotes point P2 in figure
    domain.set_delta_phi_in(d_phi_in_B)
    domain.set_delta_phi_ext(d_phi_ext_B)
    domain.solve_domain()
    phi_B = domain.electric_potential[-1] + v_mem
    q_B = domain.cumulative[-1]
    
    # potential has to change sign and therefor is larger than zero here
    t3 = (phi_B > mpfr('0'))
    if t3 == False:
        print('volatge at B is not larger than zero')
        print('V = ', phi_B)
    # cumulative is zero
    t4 = (mpfr_abs(q_B) < eps_q)
    if t4 == False:
        print('cumulative charge at B is not zero')
        print('Q = ', q_B)
    
    # check if all tests are true
    if np.all([t1, t2, t3, t4]) != True:
        print([t1, t2, t3, t4])
        raise AssertionError('boundary conditions can not be optimized within those boundaries')
    else:
        print('optimization buondaries found in domain ', domain.experiment_id)

    # Check additional assumption drawn in methods figure that shows null-clines of v and q
    # in 2d d_phi_in, d_phi_ext space
    # assumptinos are
    # q < 0 at upper right corner
    # q > 0 at lower left corner
    domain.set_delta_phi_in(d_phi_in_A)
    domain.set_delta_phi_ext(d_phi_ext_B)
    domain.solve_domain()
    # phi_AB = domain.electric_potential[-1] + v_mem
    q_AB = find_max_index_finite_element(domain, quantity='q', position=None) 
    
    domain.set_delta_phi_in(d_phi_in_B)
    domain.set_delta_phi_ext(d_phi_ext_A)
    domain.solve_domain()
    # phi_BA = domain.electric_potential[-1] + v_mem
    q_BA =  find_max_index_finite_element(domain, quantity='q', position=None)
    t5 = (q_AB > mpfr('0'))
    t6 = (q_BA < mpfr('0'))
    # check if all tests are true
    if np.all([t5, t6]) != True:
        print('WARNING: cumulative charge has different sign as expected')
        print(t5, t6, q_AB, q_BA)
    else:
        pass
        
#######################################################################################
def main():
    """
    this script follows all steps described in the optimize_perturbation_parameters.ipynb
    to find the correct boundary conditions
    
    run code (example with 12 processes):
    mpiexec -n 12 ipython -m mpi4py solve_optimize.py 3
    use -m mpi4py option to prevent deadlocks
    last number "3" sets run variable to control parameter subset
    see: https://mpi4py.readthedocs.io/en/stable/mpi4py.run.html
    """
    
    import sys
    from mpi4py import MPI
    import copy
    import pandas as pd
    import pickle
    import time
    
    import spine_head
    import spine_neck
    # make reference to domain classes
    SpineHead = spine_head.SpineHead
    SpineNeck = spine_neck.SpineNeck
    from constants import constants
    import file_io as fio

    # set preciosion of mpfr variables
    gmpy2.get_context().precision = constants['precision']
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # test numper of processes
    #n_processes = 1
    n_processes = 12
    if not n_processes == size:
        raise ValueError('bad number of processes') 
    run= eval(sys.argv[1])
    max_runs = 11
    if run > max_runs:
        raise ValueError('bad run variable argv[1]')
        
    # crate empty log-file 
    global start_time
    start_time = time.time()
    if rank == 0:
        logfile = open('logfile.txt', 'a')    
        logfile.write('... solver started ...\n')
        logfile.close()
    
    
    # set up parameters for simulations
    param_collection = []
    # results 
    results_collection = []
    ###########################################################################
    # run indices 0 - 9
    for domain_type in ['head', 'neck']:
        for radius in [20, 40, 80, 160, 320]:
            for Vm in [-70., -35., 0., 35.,]:          
                for dV in [-1, 0, 1]:
                    
                    max_repetitions = int(radius * 3)
                    
                    # set parameters of spine domain           
                    d_mem = 5e-9          
                    
                    n_digits = 9  # round to nm
                    
                    r = radius * 1e-9
                    r_in = str(round(r, n_digits))
                    r_mem = str(round(r+d_mem,n_digits))
                    r_out =  str(round(2*r+d_mem, n_digits))
                    
                    
                    eps_in = '50'
                    eps_mem = '5'
                    eps_out = '60'           
                    
                    # particel concentrations
                    c_pos_in = '165.0'
                    c_neg_in = '10.0'
                    c_back_in = '155.0'
                    
                    
                    c_pos_out = '155.00'
                    c_neg_out = '125.00'
                    c_back_out = '30.00'
                    
                    # set surface charge density to -0.02 C/m^2
                    rho_surf_in = '-0.02'
                    rho_surf_out = '-0.02'
                    
                    # membra^
                    v_mem =  str(round((Vm + dV) * 0.001,3))  
                    
                    # r_in, r_mem, r_out, eps_in, eps_mem, eps_out, c_pos_in, c_neg_in, c_back_in, c_pos_out, c_neg_out, c_back_out, rho_surf_in, rho_surf_out, v_mem
                    # provide all parameters as str-object except res_in, res_mem, res_ext - use int instead
                    param_set = [r_in, r_mem, r_out, # 0 1 2
                                 eps_in, eps_mem, eps_out, # 3 4 5
                                 c_pos_in, c_neg_in, c_back_in,  # 6 7 8 
                                 c_pos_out, c_neg_out, c_back_out,  # 9 10 11
                                 rho_surf_in, rho_surf_out, v_mem,  # 12 13 14
                                 domain_type]  # 15
                    param_collection.append(param_set)
                    
                    
    # run indices 10 - 11
    for domain_type in ['head', 'neck']:
        for radius in [10]:
            for Vm in [-70., -35., 0., 35.,]:          
                for dV in [-1, 0, 1]:
                    
                    max_repetitions = int(radius * 3)
                    
                    # set parameters of spine domain           
                    d_mem = 5e-9          
                    
                    n_digits = 9  # round to nm
                    
                    r = radius * 1e-9
                    r_in = str(round(r, n_digits))
                    r_mem = str(round(r+d_mem,n_digits))
                    r_out =  str(round(2*r+d_mem, n_digits))
                    
                    
                    eps_in = '50'
                    eps_mem = '5'
                    eps_out = '60'           
                    
                    # particel concentrations
                    c_pos_in = '165.0'
                    c_neg_in = '10.0'
                    c_back_in = '155.0'
                    
                    
                    c_pos_out = '155.00'
                    c_neg_out = '125.00'
                    c_back_out = '30.00'
                    
                    # set surface charge density to -0.02 C/m^2
                    rho_surf_in = '-0.02'
                    rho_surf_out = '-0.02'
                    
                    # membra^
                    v_mem =  str(round((Vm + dV) * 0.001,3))  
                    
                    # r_in, r_mem, r_out, eps_in, eps_mem, eps_out, c_pos_in, c_neg_in, c_back_in, c_pos_out, c_neg_out, c_back_out, rho_surf_in, rho_surf_out, v_mem
                    # provide all parameters as str-object except res_in, res_mem, res_ext - use int instead
                    param_set = [r_in, r_mem, r_out, # 0 1 2
                                 eps_in, eps_mem, eps_out, # 3 4 5
                                 c_pos_in, c_neg_in, c_back_in,  # 6 7 8 
                                 c_pos_out, c_neg_out, c_back_out,  # 9 10 11
                                 rho_surf_in, rho_surf_out, v_mem,  # 12 13 14
                                 domain_type]  # 15
                    param_collection.append(param_set)

    # set of parameters used in this run
    params = param_collection[12 * run : 12*(run+1)]
    
    # send all simulation parameters to their respective processes
    if rank == 0:
        for r in range(1, size):  # loop over all processes where i denotes their ranks
            comm.send(params[r], dest=r, tag=100+r)
        params_r = params[0]
    # 
    elif rank != 0:
        params_r = comm.recv(source=0, tag=100+rank)
        
    print(12 * run + rank, params_r)
    
    


    domain_type = params_r[15]    
    if domain_type == 'head':
        domain = SpineHead(params_r[0], params_r[3], params_r[1],params_r[4], params_r[2], params_r[5],)
    elif domain_type == 'neck':
        domain = SpineNeck(params_r[0], params_r[3], params_r[1],params_r[4], params_r[2], params_r[5],)
    domain.set_intracellular_concentrations(params_r[6], params_r[7], params_r[8],)
    domain.set_surface_concentrations(params_r[12], params_r[13])
    domain.set_extracellular_concentrations(params_r[9], params_r[10], params_r[11],)
    domain.set_membrane_potential(params_r[14])
    
    domain.set_id(12 * run + rank)
    
    # find boundaries of perturbation parameters
    # the solution of the equation is within those boundaries 
    cpy_dm1, cpy_dm2, cpy_dm3, cpy_dm4 = copy.deepcopy(domain), copy.deepcopy(domain), copy.deepcopy(domain), copy.deepcopy(domain)
    
    max_repetitions = int(domain.get_r_in() * 1e9 * 3)
    
    

    d_phi_in_A = find_d_phi_in_A(cpy_dm1, max_repetitions=max_repetitions)
    
    
    d_phi_ext_A = find_d_phi_ext_A(cpy_dm2, d_phi_in=d_phi_in_A, max_repetitions=max_repetitions)
    d_phi_in_B = find_d_phi_in_B(cpy_dm3, max_repetitions)
    d_phi_ext_B = find_d_phi_ext_B(cpy_dm4, d_phi_in_B, max_repetitions)  
    

    d_phi_in_bnds = [d_phi_in_A, d_phi_in_B]
    d_phi_ext_bnds = [d_phi_ext_A, d_phi_ext_B]
    
    # is solution within those boundaries
    test_optimization_boundaries(domain, d_phi_in_bnds, d_phi_ext_bnds)

    # optimize the boundary conditions of the initial value problem until the boundary conditions
    # of the original problem are fullfilled
    
    d_phi_in, d_phi_ext = shoot(domain, d_phi_in_A, d_phi_in_B, d_phi_ext_A, d_phi_ext_B, max_repetitions, logfile=True)

    # read all relevant paramters and main results in a pandas dataframe
    df_r = fio.main_results_to_pandas(spine_domain=domain, 
                                    optimization_boundaries=[d_phi_in_A, d_phi_in_B, d_phi_ext_A, d_phi_ext_B]) 
                                    

    # send_results back
    # send data back to 0
    if rank != 0: 
        df_r = pickle.dumps(df_r)
        comm.send(df_r, dest=0, tag=200+rank)
        
    elif rank == 0:
        results_collection.append(df_r)
        for r in range(1, size):
            results_r = comm.recv(source=r, tag=200+r)
            results_r = pickle.loads(results_r)
            results_collection.append(results_r)
    
    
    # collect all results in pandas dataframe and save
    if rank == 0:
        df_tmp = results_collection[0]
        for r in range(1, size):
            new_row_df = results_collection[r]
            new_row_df.index = [df_tmp.shape[0]]
            df_tmp = pd.concat([df_tmp, new_row_df], axis=0)
        
        print(df_tmp)
        pickle.dump(df_tmp, open('./../../results/results_test_run'+str(run)+'.pcl', 'wb'))

if __name__ == '__main__':
    
    main()
