#!/usr/bin/python

import numpy as np
import gmpy2
from gmpy2 import mpfr
from .constants import constants
from .math_mpfr import *  # cube, d1r, d2r

# set preciosion of mpfr variables
gmpy2.get_context().precision = constants['precision']

# TODO add all methods for PNP terms

class GenericSpineDomain:
    
    def __init__(self, r1, eps1, r2, eps2, r3, eps3, grid_spacing='log'):
        # type of spine domain: can be base and head and neck for derived classes
        self.domain = 'base'
        
        # used to save results and print messages
        self.experiment_id = 0
        
        
        
        # set precision of mpfr-variables
        self.precision = constants['precision']
        gmpy2.get_context().precision=self.precision
        
        # check if all numbers are provied as stting-variable
        # only string variables can be cerverted to mpfr without precision loss
        for number in [r1, eps1, r2, eps2, r3, eps3,]:
            if not (type(number)==str):
                raise TypeError('r1, eps1, r2, eps2, r3 and eps3 input variables must be of type string for exact conversion to mpfr')

        # convert inputs and set class variables
        self.radius_in = mpfr(r1)
        self.radius_mem = mpfr(r2)
        self.radius_out = mpfr(r3)
        
        self.eps_r_in = mpfr(eps1)
        self.eps_r_mem = mpfr(eps2)
        self.eps_r_out = mpfr(eps3)
        
        
        # load constants  TODO : can be removed from __init__
        self.CONST_N_A = constants['N_A']  # Avogadro
        self.CONST_e = constants['e']  # elementary charge
        self.CONST_k_B = constants['k_B']  # Boltzmann
        self.CONST_T = constants['T']  # Temperature
        self.CONST_eps_0 = constants['eps_0']  # vacuum permittivity  
        self.CONST_PI = constants['pi']  # pi
        self.CONST_EBT = self.CONST_e / self.CONST_k_B / self.CONST_T  # constant coefficient in solver
        
        # Potential variables
        # delta_phi_in and delta_phi_ext are set externally by class method
        self.membrane_potential = mpfr('-0.07')
        self.delta_phi_in = mpfr('0.0') # difference to bulk electrochemical potential at r=0
        self.delta_phi_ext = mpfr('0.0')  # difference to bulk electrochemical potential at r=r_ext
        
        # initialize charge density variables
        # c: [particle number / m^3]
        # sigma: surface density [coulomb / m^2]
        # exact values are set by class method
        self.c_pos_0_in = mpfr('0')
        self.c_neg_0_in = mpfr('0')
        self.c_back_0_in = mpfr('0')
        self.sigma_surf_in = mpfr('0')
        self.sigma_surf_out = mpfr('0')
        self.c_pos_0_out = mpfr('0')
        self.c_neg_0_out = mpfr('0')
        self.c_back_0_out = mpfr('0')
        # charge numbers (can be used to check inputs and may be modified for later versions of the code with variable charge numbers))
        self.charge_number_c_pos = mpfr('1')
        self.charge_number_c_neg = mpfr('-1')
        self.charge_number_c_back = mpfr('-1')

        
        # the distance between the grid points right at the membrane should be smaller that the debye length
        # the desired resolution variable sets the max grid interval between the two gridpoints adjacent to the membrane
        # in the intracellular and extracellular space
        self.grid_precision = mpfr('0.1e-9')  # [m]
        self.grid_spacing = 'log'
        self.res_in = 0
        self.res_mem = 0
        self.res_out = 0
        self.compute_grid_resolution()

        # define lists to save results and initialize all values to zero
        self.grid_points = [mpfr('0') for i in range(self.res_in + self.res_out + self.res_mem - 2)]
        self.eps_r = [mpfr('0') for i in range(self.res_in + self.res_out + self.res_mem - 2)]  
        
        self.concentration_positive = [mpfr('0') for i in range(self.res_in + self.res_out + self.res_mem - 2)]  
        self.concentration_negative = [mpfr('0') for i in range(self.res_in + self.res_out + self.res_mem - 2)]
        self.concentration_background = [mpfr('0') for i in range(self.res_in + self.res_out + self.res_mem - 2)]
        
        self.cumulative_positive = [mpfr('0') for i in range(self.res_in + self.res_out + self.res_mem - 2)]
        self.cumulative_negative = [mpfr('0') for i in range(self.res_in + self.res_out + self.res_mem - 2)]
        self.cumulative_background = [mpfr('0') for i in range(self.res_in + self.res_out + self.res_mem - 2)]    
        self.cumulative_mem_in = [mpfr('0') for i in range(self.res_in + self.res_out + self.res_mem - 2)]
        self.cumulative_mem_out = [mpfr('0') for i in range(self.res_in + self.res_out + self.res_mem - 2)]
        self.cumulative = [mpfr('0') for i in range(self.res_in + self.res_out + self.res_mem - 2)]
        
        self.electric_potential = [mpfr('0') for i in range(self.res_in + self.res_out + self.res_mem - 2)]
        self.electric_potential_pos = [mpfr('0') for i in range(self.res_in + self.res_out + self.res_mem - 2)]
        self.electric_potential_neg = [mpfr('0') for i in range(self.res_in + self.res_out + self.res_mem - 2)]
        self.electric_potential_back = [mpfr('0') for i in range(self.res_in + self.res_out + self.res_mem - 2)]
        self.electric_potential_mem_in = [mpfr('0') for i in range(self.res_in + self.res_out + self.res_mem - 2)]
        self.electric_potential_mem_out = [mpfr('0') for i in range(self.res_in + self.res_out + self.res_mem - 2)]
        
        self.electric_field = [mpfr('0') for i in range(self.res_in + self.res_out + self.res_mem - 2)]
        self.electric_field_pos = [mpfr('0') for i in range(self.res_in + self.res_out + self.res_mem - 2)]
        self.electric_field_neg = [mpfr('0') for i in range(self.res_in + self.res_out + self.res_mem - 2)]
        self.electric_field_back = [mpfr('0') for i in range(self.res_in + self.res_out + self.res_mem - 2)]
        self.electric_field_mem_in = [mpfr('0') for i in range(self.res_in + self.res_out + self.res_mem - 2)]
        self.electric_field_mem_out = [mpfr('0') for i in range(self.res_in + self.res_out + self.res_mem - 2)]
        
        self.chemical_potential_pos = [mpfr('0') for i in range(self.res_in + self.res_out + self.res_mem - 2)]
        self.chemical_potential_neg = [mpfr('0') for i in range(self.res_in + self.res_out + self.res_mem - 2)]
        self.chemical_potential_back = [mpfr('0') for i in range(self.res_in + self.res_out + self.res_mem - 2)]
        
        self.chemical_field_pos = [mpfr('0') for i in range(self.res_in + self.res_out + self.res_mem - 2)]
        self.chemical_field_neg = [mpfr('0') for i in range(self.res_in + self.res_out + self.res_mem - 2)]
        self.chemical_field_back = [mpfr('0') for i in range(self.res_in + self.res_out + self.res_mem - 2)]

        self.pnp_lhs = [mpfr('0') for i in range(self.res_in + self.res_out + self.res_mem - 2)]
        self.pnp_rhs = [mpfr('0') for i in range(self.res_in + self.res_out + self.res_mem - 2)]        
        
        
        self.compute_grid_points()
        self.set_permittivity_vector()
         
    def set_id(self, exId):
        self.experiment_id = exId
    
    def set_permittivity_vector(self):
        for i in range(1,self.res_in):
            self.eps_r[i] = self.eps_r_in
        for i in range(0, self.res_mem-1):
            self.eps_r[self.res_in+i] = self.eps_r_mem
        for i in range(0, self.res_out-1):
            self.eps_r[self.res_in+self.res_mem-1+i] = self.eps_r_out
    
    def compute_grid_resolution(self):
        """
        compute grid resolution
        intervals at membrane have to be smaller than self.grid_precision
        """
        if self.grid_spacing == 'lin':
            
            self.res_in = int(gmpy2.ceil(self.radius_in / self.grid_precision)) + 1
            d_mem = self.radius_mem-self.radius_in
            elf.res_mem = int(gmpy2.ceil(d_mem / self.grid_precision)) + 1
            d_ext = self.radius_out - self.radius_mem
            self.res_out = int(gmpy2.ceil(d_ext / self.grid_precision)) + 2  # extracell space + 2, otherwise there is one grid point less than intracell.
            
        elif self.grid_spacing == 'log':
            # if scale parameter is higher grid points are moved towards the membrane
            # larger domains should have a higher scale because at the 
            # center concentrations are almost constant and the resulution can
            # be reduced there
            # for extremely small domains the debye layer increases
            # therefore a higher resulution at the center is needed
            scale_in = self.radius_in * mpfr('1.e9')
            scale_ext = (self.radius_out - self.radius_mem) * mpfr('1.e9')
            d = self.grid_precision  # shorter reference
            
            # intracellular space
            R  = self.radius_in
            
            tmp = (mpfr('1')-d/R)*gmpy2.log(scale_in+mpfr('1'))
            N= mpfr('1') / (mpfr('1')-(gmpy2.exp(tmp)-mpfr('1')) / scale_in)
            
            self.res_in = int(gmpy2.ceil(N)) + 1
            
            # membrane linear grid
            d_mem = self.radius_mem-self.radius_in
            self.res_mem = int(gmpy2.ceil(d_mem / self.grid_precision)) 
            
            # extracellular space
            R = self.radius_out - self.radius_mem
            tmp = (mpfr('1')-d/R)*gmpy2.log(scale_ext+mpfr('1'))
            N= mpfr('1') / (mpfr('1')-(gmpy2.exp(tmp)-mpfr('1')) / scale_ext)
            self.res_out = int(gmpy2.ceil(N)) + 2 # extracell space + 2, otherwise there is one grid point less than intracell.
    
    def transform_lin_to_log_spacing(self):
        """
        method to transform a linearly spaced grid into an grid with decreases interval size near the membrane
        in the intracellular space
        """
        R_in = self.grid_points[self.res_in-1]#self.radius_in
        R_ext = self.grid_points[self.res_in+ self.res_mem+ self.res_out-3] - self.grid_points[self.res_in + self.res_mem -1]  # self.radius_out - self.radius_mem

        scale_in = R_in * mpfr('1.e9')
        scale_ext = R_ext * mpfr('1.e9')
        
        for i in range(self.res_in):        
            r_i = self.grid_points[i] / R_in
            r_new = R_in * gmpy2.log(scale_in * r_i + mpfr('1')) / gmpy2.log(scale_in + mpfr('1'))
            self.grid_points[i] = r_new
        
        for j in range(0, self.res_out-1):
            i = self.res_in+self.res_mem-1+j
            r_i = mpfr('1') - (self.grid_points[i] - self.grid_points[self.res_in+self.res_mem-1])  / R_ext
                     
            r_new = R_ext - (R_ext * gmpy2.log(scale_ext * r_i + mpfr('1'))
                             / gmpy2.log(scale_ext + mpfr('1')) ) + self.grid_points[self.res_in+self.res_mem-1]
            self.grid_points[i] = r_new
            

    
    def check_grid_precision(self):
        """
        test if grid precision is sufficient
        intervals adjacent to membrane (intraclluar and extracell.) should be smaller
        than self.grid_precision

        Returns
        -------
        None.

        """
    
        interval_in = self.grid_points[self.res_in - 1] - self.grid_points[self.res_in - 2]
        interval_ext = self.grid_points[self.res_in + self.res_mem] - self.grid_points[self.res_in + self.res_mem -1]
        if interval_in > self.grid_precision or interval_ext > self.grid_precision:
            print('interval_in: ', interval_in)
            print('interval_ext: ', interval_ext)
            raise Exception('resolution of grid too low')
        
    
    def compute_grid_points(self): 

        # subtraction of two interger is exact and can be computed beofore converting to mpfr
        delta_in = self.radius_in / mpfr(str(self.res_in-1))  
        delta_mem = (self.radius_mem-self.radius_in)/mpfr(str(self.res_mem-2));
        delta_out = (self.radius_out-self.radius_mem)/mpfr(str(self.res_out-2));
        
        # compute linearly spaced grid
        for i in range(1,self.res_in):
            self.grid_points[i] = mpfr(str(i)) * delta_in
        for i in range(0, self.res_mem-1):
            self.grid_points[self.res_in+i] = self.radius_in + mpfr(str(i)) * delta_mem
        for i in range(0, self.res_out-1):
            self.grid_points[self.res_in+self.res_mem-1+i] = self.radius_mem + mpfr(str(i)) * delta_out     
        
        
        if self.grid_spacing == 'linear':
            pass
              
        elif self.grid_spacing == 'log':

            # TODO eventually non-linarly space grid points can make problems with the d2r method of math_mpfr.py
            # but this method is only used in the compute_pnp methods and is therefore not essential for the solution
            # intracellular space and extracelluar space should have same size in r-direction
            self.transform_lin_to_log_spacing()
            pass
        else:
            raise Exception('unknown grid spacing')
        
        self.check_grid_precision() 
        
    # set functions  
    
    def set_membrane_potential(self, v_mem):
        """
        the default value is: self.membrane_potential = mpfr('-0.07')
        to study depolarized states the membrane potential can be changed with this method
        method arguments:
        v_mem: has to be of type str
        """
        if type(v_mem)==str:
            self.membrane_potential = mpfr(v_mem)
        else:
            raise TypeError('v_mem input variable must be of type str for exact conversion to mpfr')
           
    def set_delta_phi_in(self, d_phi):       
        if type(d_phi)==mpfr:
            self.delta_phi_in = mpfr(d_phi)
        else:
            raise TypeError('d_phi input variable must be of type mpfr')
        
    def set_delta_phi_ext(self, d_phi):    
        if type(d_phi)==mpfr:
            self.delta_phi_ext = mpfr(d_phi)
        else:
            raise TypeError('d_phi input variable must be of type mpfr')
          
    def set_intracellular_concentrations(self, c_pos, c_neg, c_back):
        """
        unit of function arguments for concentration is mmol
        """
        # check if inputs are str
        for number in [c_pos, c_neg, c_back]:
            if not (type(number)==str):
                raise TypeError('c_pos, c_neg, c_back input variables must be of type string for exact conversion to mpfr')     
        # set variables, inside class particle concentrations are used    
        self.c_pos_0_in = mpfr(c_pos) * self.CONST_N_A
        self.c_neg_0_in = mpfr(c_neg) * self.CONST_N_A
        self.c_back_0_in = mpfr(c_back) * self.CONST_N_A
           
    def set_surface_concentrations(self, sigma_in, sigma_out):
        """
        unit of function arguments for concentration is coulomb
        """
        # check if inputs are str
        for number in [sigma_in, sigma_out]:
            if not (type(number)==str):
                raise TypeError('sigma_in, sigma_out input variables must be of type string for exact conversion to mpfr')     
        # set variables
        self.sigma_surf_in = mpfr(sigma_in)
        self.sigma_surf_out = mpfr(sigma_out)
                    
    def set_extracellular_concentrations(self, c_pos, c_neg, c_back):
        """
        unit of function arguments for concentration is mmol
        """
        # check if inputs are str
        for number in [c_pos, c_neg, c_back]:
            if not (type(number)==str):
                raise TypeError('c_pos, c_neg, c_back input variables must be of type string for exact conversion to mpfr')     
        # set variables, inside class particle concentrations are used
        self.c_pos_0_out = mpfr(c_pos) * self.CONST_N_A
        self.c_neg_0_out = mpfr(c_neg) * self.CONST_N_A
        self.c_back_0_out = mpfr(c_back) * self.CONST_N_A
        
    def compute_concentrations_with_rk4(self, i):
        """
        implementation of the runge-kutta type rk4 algorithm
        uses rk4 algorithm to compute concentrations at r_i
        i: index that refers to position r_i
        """
        # step width between r_i and r_{i-1}
        h = self.grid_points[i] - self.grid_points[i-1]
        # make further references
        j = i - 1  # previous point before i where d/dr c is computed for RK4
        r_j = self.grid_points[j]
        c_pos_j = self.concentration_positive[j]
        c_neg_j = self.concentration_negative[j]
        
        # k1
        k_1_pos = h * self.d_c(j, r_j, c_pos_j, self.charge_number_c_pos)
        k_1_neg = h * self.d_c(j, r_j, c_neg_j, self.charge_number_c_neg)
        # k2
        k_2_pos = h * self.d_c(j, r_j + h * mpfr('0.5'), c_pos_j + mpfr('0.5') * k_1_pos, self.charge_number_c_pos )
        k_2_neg = h * self.d_c(j, r_j + h * mpfr('0.5'), c_neg_j + mpfr('0.5') * k_1_neg, self.charge_number_c_neg )
        # k3
        k_3_pos = h * self.d_c(j, r_j + h * mpfr('0.5'), c_pos_j + mpfr('0.5') * k_2_pos, self.charge_number_c_pos )
        k_3_neg = h * self.d_c(j, r_j + h * mpfr('0.5'), c_neg_j + mpfr('0.5') * k_2_neg, self.charge_number_c_neg )            
        # k4
        k_4_pos = h * self.d_c(j, r_j + h, c_pos_j + k_3_pos, self.charge_number_c_pos )
        k_4_neg = h * self.d_c(j, r_j + h, c_neg_j + k_3_neg, self.charge_number_c_neg )
        
        # compute concentrations and electric field component at r_i
        self.concentration_positive[i] = self.concentration_positive[j] + (k_1_pos + mpfr('2') * k_2_pos + mpfr('2') * k_3_pos + k_4_pos) / mpfr('6')
        self.concentration_negative[i] = self.concentration_negative[j] + (k_1_neg + mpfr('2') * k_2_neg + mpfr('2') * k_3_neg + k_4_neg) / mpfr('6')
    
    def modify_bulk_concentrations(self, charge_number, delta_phi):
        """
        at r=0 and r=grid_points[self.res_in+self.res_mem-1] (first point of 
        extracell. space) the delta_phi variables modify the concentrations
        with respect to the bulk concentrations. Moreover this controls that 
        at r=grid_points[-1] the desired bulk concentrations are reached
        delta phi:
            at r=0 equal to self.delta_phi_in
            at first point of extracell. space equal to 
                    delta_phi_ext + electric_potential[res_in+res_mem-1] + phi_mem
                    (electric_potential[res_in+res_mem-1] + phi_mem is actually the
                    difference between the potentials as phi_mem < 0)
        """
        #print('ebt:', delta_phi)
        f = gmpy2.exp(mpfr('-1')*charge_number*self.CONST_EBT*delta_phi)
        # print('f:', f)
        return f
    
    def solve_intracellular_domain(self):
        """
        solve for concentrations and electric field in intracellular space using
        a RK4 algorithm
        """
        # set concentrations to concentrations at center
        self.concentration_positive[0] = self.c_pos_0_in * self.modify_bulk_concentrations(self.charge_number_c_pos, self.delta_phi_in)
        self.concentration_negative[0] = self.c_neg_0_in * self.modify_bulk_concentrations(self.charge_number_c_neg, self.delta_phi_in)
        self.concentration_background[0] = self.c_back_0_in

        # electric field is zero at center
        
        # for first step use euler method because k_1 is undefined in runge kutta as one gets at term r/r_n = 0/0 in implmentation of rk functions
        # using euler forward scheme dc = 0 as 
        self.concentration_positive[1] = self.concentration_positive[0]
        self.concentration_negative[1] = self.concentration_negative[0]
        self.concentration_background[1] = self.concentration_background[0]
        self.compute_electric_field(0)  # computes electric field at r_1
        
        # compute concentrations and electric fields at all points {r_1, ... ,r_{res_in-1}}
        for i in range(2, self.res_in):
            # i: index where concentrations and field is computed
            self.compute_concentrations_with_rk4(i)  # compute concentrations
            self.concentration_background[i] = self.c_back_0_in  # is constant
            j=i-1  # just another reference, argument for the following function
            self.compute_electric_field(j)  # computes electric field at i=j+1
            
    def solve_membrane_domain(self):
        """
        solve for concentrations and electric field in intracellular space 
        concentrations are zero in membrane and e-field is decaying
        """
        for i in range(self.res_in, self.res_in+self.res_mem-1):
            # i: index where concentrations and field is computed
            # concentrations of mobile ions are zero in membrane -> is already initial value
            # self.concentration_positive[i] = mpfr('0')  
            # self.concentration_negative[i] = mpfr('0')
            # self.concentration_background[i] = mpfr('0')
            # self.concentration_excess[i] = mpfr('0')
            j = i - 1  # argument for following function
            self.compute_electric_field(j)  # computes electric field at i=j+1
        
        
    def solve_extracellular_domain(self):
        """
        solve for concentrations and electric field in extracellular space using
        a RK4 algorithm
        """
        # when the concentrations at the first point after the boundary are set
        # the potential is unknown at that point
        # there for we estimate it by the last point of the membrane (this error can be reduced by an increased
        # resolution)
        # then we further decrease the error by repeatedly computing phi, c and e and relax potential
        
        # set concentrations of lists at first point of extracelluar space
        i = self.res_in+self.res_mem-1
        # first estimate of potential after the membrane
        self.electric_potential[i] = self.electric_potential[i-1]
        
        # set concentrations
        self.concentration_positive[i] = self.c_pos_0_out * self.modify_bulk_concentrations(self.charge_number_c_pos, self.delta_phi_ext) 
        self.concentration_negative[i] = self.c_neg_0_out * self.modify_bulk_concentrations(self.charge_number_c_neg, self.delta_phi_ext)
        self.concentration_background[i] = self.c_back_0_out 
        
        # compute electric field at first point of extracellular space
        j = i - 1  # argument for following function
        self.compute_electric_field(j)  # computes electric field at i=j+1
        
        ################
        # repetitions to relax potential after membrane to correct value
        for k in range(0):
            self.compute_electric_potential()
            d_phi = self.electric_potential[i] + self.membrane_potential + self.delta_phi_ext
            self.concentration_positive[i] = self.c_pos_0_out * self.modify_bulk_concentrations(self.charge_number_c_pos, self.delta_phi_ext)
            self.concentration_negative[i] = self.c_neg_0_out * self.modify_bulk_concentrations(self.charge_number_c_neg, self.delta_phi_ext) 
            self.concentration_background[i] = self.c_back_0_out 
            j = i - 1  # argument for following function
            self.compute_electric_field(j)  # computes electric field at i=j+1       
        ########################
        
        # compute concentrations and electric fields at all points {r_{}, ... ,r_{res_in-1}}
        for i in range(self.res_in+self.res_mem, self.res_in+self.res_mem+self.res_out-2):
            # i: index where concentrations and field is computed
            self.compute_concentrations_with_rk4(i)  # compute concentrations
            self.concentration_background[i] = self.c_back_0_out  # is constant
            j=i-1  # just another reference, argument for the following function
            self.compute_electric_field(j)  # computes electric field at i=j+1
    
    #####################################################################################
    # functions to compute electic potential, chemical field and chemical potential
    # are identical for spine head and spine neck
    # thus they can be part of the base class
    
    def compute_electric_potential(self):
        """
        computes electric potential as negative integral over the electric field
        uses trapezoidal rule for integration
        """
        # list of all potential and field arrays
        pot_field_list = [
            (self.electric_potential_pos, self.electric_field_pos),
            (self.electric_potential_neg, self.electric_field_neg),
            (self.electric_potential_back, self.electric_field_back),
            (self.electric_potential_mem_in, self.electric_field_mem_in),
            (self.electric_potential_mem_out, self.electric_field_mem_out),
        ]
        
        # compute electric potential of each ion type
        for potential , field in pot_field_list:
            
            for i in range(1, self.res_in+self.res_mem+self.res_out-2):
                potential[i] = potential[i-1] - (field[i] + field[i-1]) * mpfr('0.5') * (self.grid_points[i] - self.grid_points[i-1])
                # print(potential[i])
        
        # total electric potential
        for i in range(0, self.res_in+self.res_mem+self.res_out-2):
            self.electric_potential[i] = (self.electric_potential_pos[i] + self.electric_potential_neg[i] + 
                                          self.electric_potential_back[i] + self.electric_potential_mem_in[i] + 
                                          self.electric_potential_mem_out[i])

    def compute_chemical_potential(self):
        """
        chemical potential mu is defined by: mu=log(c/c_0)*k_B*T/e
        in intracellular space c_0 is given by concentration at center
        in extracellular space c_0 is given by concentration at first point after membrane
        """
        
        # 1. POSITIVE IONS
        # intracellular space
        if self.concentration_positive[0] == mpfr('0'):
            for i in range(0, self.res_in):
                self.chemical_potential_pos[i] = mpfr('0')
        else:
            for i in range(0, self.res_in):
                self.chemical_potential_pos[i] = self.CONST_k_B * self.CONST_T / self.CONST_e * gmpy2.log(self.concentration_positive[i]/self.concentration_positive[0])
        # membrane      
        for i in range(self.res_in, self.res_in+self.res_mem-1):
            self.chemical_potential_pos[i] = mpfr('0.0')
        #extracellular space ( reset potential to zero at boundary   
        if self.concentration_positive[self.res_in+self.res_mem-1] == mpfr('0'): 
            for i in range(self.res_in+self.res_mem-1, self.res_in+self.res_mem+self.res_out-2):   
                self.chemical_potential_pos[i] = mpfr('0')
        else:
            for i in range(self.res_in+self.res_mem-1, self.res_in+self.res_mem+self.res_out-2):
	            self.chemical_potential_pos[i] = self.CONST_k_B * self.CONST_T / self.CONST_e * gmpy2.log(self.concentration_positive[i]/self.concentration_positive[self.res_in+self.res_mem-1]) 
	        
        # 2. NEGATIVE IONS
        # intracellular space
        if self.concentration_negative[0] == mpfr('0'):
            for i in range(0, self.res_in):
                self.chemical_potential_neg[i] = mpfr('0')
        else:
            for i in range(0, self.res_in):
	            self.chemical_potential_neg[i] = self.CONST_k_B * self.CONST_T / self.CONST_e * gmpy2.log(self.concentration_negative[i]/self.concentration_negative[0])
	    # membrane
        for i in range(self.res_in, self.res_in+self.res_mem-1):
            self.chemical_potential_neg[i] = mpfr('0.0')
	    #extracellular space ( reset potential to zero at boundary       
        if self.concentration_negative[self.res_in+self.res_mem-1] == mpfr('0'):
            for i in range(self.res_in+self.res_mem-1, self.res_in+self.res_mem+self.res_out-2):   
                self.chemical_potential_neg[i] = mpfr('0')
        else:
            for i in range(self.res_in+self.res_mem-1, self.res_in+self.res_mem+self.res_out-2):   
	            self.chemical_potential_neg[i] = self.CONST_k_B * self.CONST_T / self.CONST_e * gmpy2.log(self.concentration_negative[i]/self.concentration_negative[self.res_in+self.res_mem-1])    
        
        # 3. BACKGROUND IONS
	    # zero potential everywhere for fixed charges   
        for i in range(0, self.res_in+self.res_mem+self.res_out-2):
            self.chemical_potential_back[i] = mpfr('0')
	

    def compute_chemical_field(self):
        """
        chemical field is the chemical equivalent to the electric field
        as the electric field gets computed by the negative gradient of the electric
        potential the chemical field gets computed as the negative gradient of the 
        chemical potential
        """
        # intracellular space
        
        for i in range(0, self.res_in-1):
            self.chemical_field_pos[i] = mpfr('-1') * d1r(self.grid_points, self.chemical_potential_pos, i)
            self.chemical_field_neg[i] = mpfr('-1') * d1r(self.grid_points, self.chemical_potential_neg, i)
            self.chemical_field_back[i] = mpfr('-1') * d1r(self.grid_points, self.chemical_potential_back, i)
        
        i = self.res_in - 1  # backward difference for last point
        self.chemical_field_pos[i] = mpfr('-1') * d1r(self.grid_points, self.chemical_potential_pos, i, method='bwd')
        self.chemical_field_neg[i] = mpfr('-1') * d1r(self.grid_points, self.chemical_potential_neg, i, method='bwd')
        self.chemical_field_back[i] = mpfr('-1') * d1r(self.grid_points, self.chemical_potential_back, i, method='bwd')
        
        for i in range(self.res_in, self.res_in+self.res_mem-1):
            self.chemical_field_pos[i] = mpfr('0')
            self.chemical_field_neg[i] = mpfr('0')
            self.chemical_field_back[i] = mpfr('0')
        
        for i in range(self.res_in+self.res_mem-1, self.res_in+self.res_mem+self.res_out-3):  
            self.chemical_field_pos[i] = mpfr('-1') * d1r(self.grid_points, self.chemical_potential_pos, i)
            self.chemical_field_neg[i] = mpfr('-1') * d1r(self.grid_points, self.chemical_potential_neg, i)
            self.chemical_field_back[i] = mpfr('-1') * d1r(self.grid_points, self.chemical_potential_back, i)
            
        i = self.res_in+self.res_mem+self.res_out-3  # backward difference for last point
        self.chemical_field_pos[i] = mpfr('-1') * d1r(self.grid_points, self.chemical_potential_pos, i, method='bwd')
        self.chemical_field_neg[i] = mpfr('-1') * d1r(self.grid_points, self.chemical_potential_neg, i, method='bwd')
        self.chemical_field_back[i] = mpfr('-1') * d1r(self.grid_points, self.chemical_potential_back, i, method='bwd')
    
    def solve_domain(self):
        """
        solve for all concentrations potentials fields...
        """
        # compute field generated by membrane charges
        self.compute_e_field_membrane_charge_in()
        self.compute_e_field_membrane_charge_out()

        # solve intracellular space
        self.solve_intracellular_domain()
        # solve conentrations in membrane
        self.solve_membrane_domain()

        # compute temp. el. pot
        self.compute_electric_potential()

        self.solve_extracellular_domain()

        self.compute_cumulative_charge()
        self.compute_electric_potential()
        self.compute_chemical_potential()
        self.compute_chemical_field()
        #head.compute_pnp()
       
    #####################################################################################
    # get functions 
    # use get funcitons to convert mpfr -> numpy 
    
    def convert_to_numpy_array(self, mpfr_list):
        """
        function to convert list of mpfr-variables to numpy arrays
        """
        arr = np.zeros(self.res_in + self.res_mem + self.res_out -2, dtype=float)
        for i in range(self.res_in + self.res_mem + self.res_out -2):
            arr[i] = np.float(mpfr_list[i])
        return arr
    
    # label of experiment 
    def get_experiment_id(self):
        return self.experiment_id
    
    # domain type: base, neck or head
    def get_domain_type(self):
        return self.domain
    
    # potentials
    def get_membrane_potential(self):
        return np.float(self.membrane_potential)
        
    def get_d_phi_in(self):
        return str(self.delta_phi_in)
        
    def get_d_phi_ext(self):
        return str(self.delta_phi_ext)
    
    # reolution and grid spacint
    def get_res_in(self):
        return self.res_in
        
    def get_res_mem(self):
        return self.res_mem
        
    def get_res_ext(self):
        return self.res_out
        
    def get_grid_spacing(self):
        return self.grid_spacing
    
    # radius
    def get_r_in(self):
        return np.float(self.radius_in)
        
    def get_r_mem(self):
        return np.float(self.radius_mem)
        
    def get_r_out(self):
        return np.float(self.radius_out)
    
    # relative permittivity    
    def get_eps_in(self):
        return np.float(self.eps_r_in)
    
    def get_eps_mem(self):
        return np.float(self.eps_r_mem)
        
    def get_eps_ext(self):
        return np.float(self.eps_r_out)
   
    # bulk concentrations in mmol
    def get_c_pos_0_in(self):
        return np.float(self.c_pos_0_in / self.CONST_N_A)
        
    def get_c_neg_0_in(self):
        return np.float(self.c_neg_0_in / self.CONST_N_A)
    
    def get_c_back_0_in(self):
        return np.float(self.c_back_0_in / self.CONST_N_A)
        
    def get_c_pos_0_ext(self):
        return np.float(self.c_pos_0_out / self.CONST_N_A)
        
    def get_c_neg_0_ext(self):
        return np.float(self.c_neg_0_out / self.CONST_N_A)
    
    def get_c_back_0_ext(self):
        return np.float(self.c_back_0_out / self.CONST_N_A)
        
    # surface concentrations
    def get_sigma_surf_in(self):
        return np.float(self.sigma_surf_in)
        
    def get_sigma_surf_ext(self):
        return np.float(self.sigma_surf_out)
        
    # charge numbers
    def get_charge_number_c_pos(self):
        return np.float(self.charge_number_c_pos)
        
    def get_charge_number_c_neg(self):
        return np.float(self.charge_number_c_neg)
        
    def get_charge_number_c_back(self):
        return np.float(self.charge_number_c_back)    
    
    # get class array variables
    def get_grid_points(self):
        return self.convert_to_numpy_array(self.grid_points)
        
    def get_eps_r(self):
        return self.convert_to_numpy_array(self.eps_r)
                                                                     
    def get_concentration_positive(self):
        return self.convert_to_numpy_array(self.concentration_positive) / np.float(self.CONST_N_A)
   
    def get_concentration_negative(self):
        return self.convert_to_numpy_array(self.concentration_negative) / np.float(self.CONST_N_A)
        
    def get_concentration_background(self):
        return self.convert_to_numpy_array(self.concentration_background) / np.float(self.CONST_N_A)
    
    def get_cumulative_positive(self):
        return self.convert_to_numpy_array(self.cumulative_positive)
    
    def get_cumulative_negative(self):
        return self.convert_to_numpy_array(self.cumulative_negative)
    
    def get_cumulative_background(self):
        return self.convert_to_numpy_array(self.cumulative_background)
        
    def get_cumulative_mem_in(self):
        return self.convert_to_numpy_array(self.cumulative_mem_in)
        
    def get_cumulative_mem_out(self):
        return self.convert_to_numpy_array(self.cumulative_mem_out)
        
    def get_cumulative(self):
        return self.convert_to_numpy_array(self.cumulative)
       
    def get_electric_potential_pos(self):
        return self.convert_to_numpy_array(self.electric_potential_pos)
        
    def get_electric_potential_neg(self):
        return self.convert_to_numpy_array(self.electric_potential_neg)
    
    def get_electric_potential_back(self):
        return self.convert_to_numpy_array(self.electric_potential_back)
        
    def get_electric_potential_mem_in(self):
        return self.convert_to_numpy_array(self.electric_potential_mem_in)
    
    def get_electric_potential_mem_out(self):
        return self.convert_to_numpy_array(self.electric_potential_mem_out)
    
    def get_electric_potential(self):
        return self.convert_to_numpy_array(self.electric_potential)       
        
    def get_electric_field_pos(self):
        return self.convert_to_numpy_array(self.electric_field_pos)
        
    def get_electric_field_neg(self):
        return self.convert_to_numpy_array(self.electric_field_neg)
     
    def get_electric_field_back(self):
        return self.convert_to_numpy_array(self.electric_field_back)
        
    def get_electric_field_mem_in(self):
        return self.convert_to_numpy_array(self.electric_field_mem_in)
        
    def get_electric_field_mem_out(self):
        return self.convert_to_numpy_array(self.electric_field_mem_out)
    
    def get_electric_field(self):
        return self.convert_to_numpy_array(self.electric_field)
      
    def get_chemical_potential_pos(self):
        return self.convert_to_numpy_array(self.chemical_potential_pos)
        
    def get_chemical_potential_neg(self):
        return self.convert_to_numpy_array(self.chemical_potential_neg)
    
    def get_chemical_potential_back(self):
        return self.convert_to_numpy_array(self.chemical_potential_back)
        
    def get_chemical_field_pos(self):
        return self.convert_to_numpy_array(self.chemical_field_pos)
        
    def get_chemical_field_neg(self):
        return self.convert_to_numpy_array(self.chemical_field_neg)
        
    def get_chemical_field_back(self):
        return self.convert_to_numpy_array(self.chemical_field_back)
    
    def get_pnp_lhs(self):
        return self.convert_to_numpy_array(self.pnp_lhs)

    def get_pnp_rhs(self):
        return self.convert_to_numpy_array(self.pnp_rhs)
       

