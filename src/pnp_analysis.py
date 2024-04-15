#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 13:26:47 2020

@author: Florian Eberhardt
"""

import numpy as np
from .file_io import load_domain_from_df
from .constants import constants
import matplotlib.pyplot as plt

def debye_lenght_theory(df, dom_id, side='intracellular', include_background=False):
    """
    definition taken from
    The Electrostatic Screening Length in Concentrated Electrolytes Increases With Concentration
    Smith, Lee, Perkin
    """
    
    e = float(constants['e'])
    k_B = float(constants['k_B'])
    T = float(constants['T'])
    eps_0 = float(constants['eps_0']) 
    N_A = float(constants['N_A'])
    if side == 'intracellular':
        eps_r = df.loc[dom_id, 'eps_in']
        c_pos = df.loc[dom_id, 'c_pos_in_0']
        c_neg = df.loc[dom_id, 'c_neg_in_0']
        c_back = df.loc[dom_id, 'c_back_in_0']
    elif side == 'extracellular':
        eps_r = df.loc[dom_id, 'eps_ext']
        c_pos = df.loc[dom_id, 'c_pos_ext_0']
        c_neg = df.loc[dom_id, 'c_pos_ext_0']
        c_back = df.loc[dom_id, 'c_pos_ext_0']
        
    if include_background == False:
        c_back = 0.
    nominator = eps_r * eps_0 * k_B * T
    denominator = N_A * (c_pos + c_neg + c_back) * e * e
    dbl = np.sqrt(nominator/denominator)
    return dbl

def search_index(df, param_dict):
    """
    df: pandas dataframe
    param_dict: python dict that contains dataframe keys as dict-keys and values to restrict the indices to lines
    that fullfill those restricions
    
    example:
    param_dict = {'membrane_potential': -0.07, 'r_in': 20e-9}
    ids = search_index(df, param_dict)
    df.loc[ids]
    """
    slc = [True] * df.shape[0]

    for key in param_dict:
        new_slc = df.loc[:, key] == param_dict[key]

        slc = np.logical_and(slc, new_slc)
    
    return df.index[slc]

def measure_double_layer_size(df, dom_id, side='intracellular', visualize=False):
    """
    measure the distance to boundary where the elctric potential drops to 1/e of the value at the membrane surface
    """
    domain = load_domain_from_df(df, dom_id)
    
    res_in = domain.get_res_in()
    res_mem = domain.get_res_mem()
    # res_ext = domain.get_res_ext()
    grid_points = domain.get_grid_points()
    potential = domain.get_electric_potential()
    
    
    
    # slice intracellaluar or extracelluar part of arrays
    if side == 'intracellular':
        x = grid_points[0: res_in]
        phi = potential[0 : res_in][::-1]
        x = np.abs(x - np.max(x))[::-1]
        
        
    elif side == 'extracellular':
        x = grid_points[res_in + res_mem -1:]
        phi = potential[res_in + res_mem - 1:]
        #x = x[::-1]
        #phi=phi[::-1]
    else: 
        raise ValueError('bad argument for "side"')
    
    # transform phi
    # phi should be positive and decreasing
    # phi[0] should be larges value (at membrane)
    # pho[-1] should be zero (approx value for x->inf)
    phi = phi - phi[-1]
    phi = phi * np.sign(phi[0])
    
    v_mem = phi[0]
    v_dl = v_mem / np.e
    dl_size_min = np.max(x[np.where(phi > v_dl)]) - x[0]
    dl_size_max = np.min(x[np.where(phi < v_dl)]) - x[0]
    phi_min = np.min(phi[np.where(phi > v_dl)])# potential at dl_size_min
    phi_max = np.max(phi[np.where(phi < v_dl)])# potential at dl_size_max
    print(phi_min, phi_max)
    dl_size = dl_size_min + (dl_size_max - dl_size_min) * (v_dl - phi_min) / ( phi_max - phi_min)
    
    if visualize == True:
        plt.plot(x,phi,'kx')
        plt.plot([x[0], x[0]+dl_size_min],[0,0])
        plt.plot([x[0], x[0]+dl_size_max],[0,0])
        plt.plot(grid_points, potential, 'r')
        
        plt.show()
        
    return dl_size_min, dl_size_max, dl_size


######################
# figure 5 
######################


def double_layer_free_charge(df, df_id, side='intracellular', dl_factor=1.):
    """
    measure the free charge inside the double layer region adjacent to the boundary
    """
    domain = load_domain_from_df(df, df_id)
    
    res_in = domain.get_res_in()
    res_mem = domain.get_res_mem()
    # res_ext = domain.get_res_ext()
    grid_points = domain.get_grid_points()
    cuml_pos = domain.get_cumulative_positive()
    cuml_neg = domain.get_cumulative_negative()
    
    dl_size = np.mean(measure_double_layer_size(df, df_id, side=side)) * dl_factor
    print('double layer size: ', dl_size)
    if side == 'intracellular':
        grid_points[res_in:] = - np.inf
        dl_indices = np.where( grid_points >= (grid_points[res_in - 1] - dl_size) )
    
    if side == 'extracellular':
        grid_points[:res_in + res_mem - 1] = np.inf
        dl_indices = np.where( grid_points < (grid_points[res_in + res_mem - 1] + dl_size) )
        
    min_index = np.min(dl_indices)
    max_index = np.max(dl_indices)
    
    # cuml pos is an increasing function -> max_index - min_index
    # cuml_neg is a decreasing function -> min_index - max_index
    charge_pos = cuml_pos[max_index] - cuml_pos[min_index]
    charge_neg = cuml_neg[min_index] - cuml_neg[max_index]
    
    abs_charge = charge_pos + charge_neg
    
    return abs_charge

def membrane_charge(df, df_id, side='intracellular'):
    """
    measure the number of charges on one side of the membrane
    """
    domain_type = df.loc[df_id, 'domain']
    
    if side == 'intracellular':
        r = df.loc[df_id, 'r_in']
        sigma = df.loc[df_id, 'sigma_in']
    elif side == 'extracellular':
        r = df.loc[df_id, 'r_ext']
        sigma = df.loc[df_id, 'sigma_in']
    else:
        raise ValueError('side')
        
    if domain_type == 'head':
        surface = 4.* np.pi * np.square(r)
    elif domain_type == 'neck':
        surface = 2. * np.pi * r
    
        
    charge = sigma * surface
    
    return charge

def total_charge(df, df_id, side='intracellular'):
    domain = load_domain_from_df(df, df_id)
    res_in = domain.get_res_in()
    res_mem = domain.get_res_mem()
    res_ext = domain.get_res_ext()
    cuml_back = domain.get_cumulative_background()  
    cuml_pos = domain.get_cumulative_positive()    
    cuml_neg = domain.get_cumulative_negative()
    
    # slice intracellaluar or extracelluar part of arrays
    if side == 'intracellular':
        charge = cuml_pos[res_in + res_mem - 1] - cuml_neg[res_in + res_mem - 1] - cuml_back[res_in + res_mem - 1]
        
        
    elif side == 'extracellular':
        charge_pos = cuml_pos[-1] - cuml_pos[res_in + res_mem - 1]
        charge_neg = cuml_pos[-1] - cuml_pos[res_in + res_mem - 1]
        charge_back = cuml_back[-1] - cuml_back[res_in + res_mem - 1]
        charge = charge_pos - charge_neg - charge_back
    else: 
        raise ValueError('bad argument for "side"')
    
    return charge

def free_charge(df, df_id, side='intracellular'):
    domain = load_domain_from_df(df, df_id)
    res_in = domain.get_res_in()
    res_mem = domain.get_res_mem()
    res_ext = domain.get_res_ext()
    cuml_pos = domain.get_cumulative_positive()    
    cuml_neg = domain.get_cumulative_negative()
    
    # slice intracellaluar or extracelluar part of arrays
    if side == 'intracellular':
        charge = cuml_pos[res_in + res_mem - 1] - cuml_neg[res_in + res_mem - 1]
        
        
    elif side == 'extracellular':
        charge_pos = cuml_pos[-1] - cuml_pos[res_in + res_mem - 1]
        charge_neg = cuml_pos[-1] - cuml_pos[res_in + res_mem - 1]
        charge = charge_pos - charge_neg
    else: 
        raise ValueError('bad argument for "side"')
    
    return charge

def excess_charge(df, df_id, side='intracellular'):
    """
    measure the excess charges on the intacellular side for a given solution
    """

    domain = load_domain_from_df(df, df_id)
    res_in = domain.get_res_in()
    res_mem = domain.get_res_mem()
    res_ext = domain.get_res_ext()
    cuml = domain.get_cumulative()
    
    # slice intracellaluar or extracelluar part of arrays
    if side == 'intracellular':
        charge = cuml[res_in -1]
        
        
    elif side == 'extracellular':
        charge = cuml[-1] - cuml[res_in + res_mem - 1]
    else: 
        raise ValueError('bad argument for "side"')
    
    return charge

def capacitance_charges(df_cap, id_1, id_2, side='intracellular'):
    """
    the number of charges to charge the membrane capacitance until the membrane voltage is -70mV
    can be computed as the difference of excess charges in the two states v_m = -70 & v_m =0 mV.
    """
    excess_charge_1 = excess_charge(df_cap, id_1,side=side)
    excess_charge_2 = excess_charge(df_cap, id_2,side=side)
    
    cap_charges = excess_charge_1 - excess_charge_2
    
    return cap_charges

