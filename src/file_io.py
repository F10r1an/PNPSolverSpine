import datetime
import os
import pandas as pd
import pickle
from gmpy2 import mpfr
from .spine_neck import SpineNeck
from .spine_head import SpineHead
import gmpy2.gmpy2


def load_parameter_overview(
    folder = '/home/work/Projects/spineIonConcentrations/results/'
    ):
    """
    load files that contain dataframes with parameters of solutions and 
    merge all dataframes into a single dataframe
    """
    files = os.listdir(folder)
    print('found files: ', files)
    
    # load dataframes with main results
    print('open :', folder+files[0])
    df = pickle.load(open(folder+files[0], 'rb'))
    
    for i in range(1, len(files)):
        df_tmp = pickle.load(open(folder+files[i], 'rb'))
        df = pd.concat([df, df_tmp], axis=0)
    df.index=df.loc[:, 'experiment_id']   
    df.sort_index(inplace=True)
    return df

def main_results_to_pandas(spine_domain, main_results_df=None, optimization_boundaries=[0,0,0,0]):
    """
    Save all relevant parameters and results of a found solution as new line to a dataframe. From that line the solution
    can be recomputed without optimizing the perturbations parameters again.
    
    spine_domain: spine_head or spine_neck domain with existing solution
    main_results_df: pandas dataframe to which the line should be appended, creates a new dataframe if None
    optimization_boundaries: [dPhi_in_min, dPhi_in_max, dPhi_ext_min, dPhi_ext_max], boundaries of perturbation parameters that where computed befor optimization, is set to zero if None argument is provided
    """
    
    s = spine_domain  # just a shorter refercence
    
    # get relevant arrays    
    c_pos = s.get_concentration_positive()
    c_neg = s.get_concentration_negative()
    v = s.get_electric_potential()
    e = s.get_electric_field()
    q = s.get_cumulative()
    chem_v_pos = s.get_chemical_potential_pos()
    chem_v_neg = s.get_chemical_potential_neg()
    chem_f_pos = s.get_chemical_field_pos()
    chem_f_neg = s.get_chemical_field_neg()
    pnp_lhs = s.get_pnp_lhs()
    pnp_rhs = s.get_pnp_rhs()  
    # spatial resolutions
    res_in, res_mem, res_ext = s.get_res_in(), s.get_res_mem(), s.get_res_ext()
    # highes index of intracell. space
    i_in =  res_in - 1
    # highest index of membrane space
    i_mem = res_in + res_mem - 2
    # highest index of extracell. space
    i_ext = res_in + res_mem + res_ext - 3
    
    row_dict = {
        # VARIABLES THAT ARE IMPORTANT TO REPRODUCE RESULTS
        'domain': [s.get_domain_type()], # head or neck
        # potential
        'membrane_potential': [s.get_membrane_potential()],
        # perturbation parameters
        'd_phi_in': [s.get_d_phi_in()],
        'd_phi_ext': [s.get_d_phi_ext()],
        # perturbation parameter boundaries
        'd_phi_in_min': [optimization_boundaries[0]],
        'd_phi_in_max': [optimization_boundaries[1]],
        'd_phi_ext_min': [optimization_boundaries[2]],
        'd_phi_ext_max': [optimization_boundaries[3]],
        # size 
        'r_in': [s.get_r_in()],
        'r_mem': [s.get_r_mem()],
        'r_ext': [s.get_r_out()],
        'eps_in': [s.get_eps_in()],
        'eps_mem': [s.get_eps_mem()],
        'eps_ext': [s.get_eps_ext()],
        # bulk concentations
        'c_pos_in_0': [s.get_c_pos_0_in()],
        'c_pos_ext_0': [s.get_c_pos_0_ext()],
        'c_neg_in_0': [s.get_c_neg_0_in()],
        'c_neg_ext_0': [s.get_c_neg_0_ext()],
        'c_back_in_0': [s.get_c_back_0_in()],
        'c_back_ext_0': [s.get_c_back_0_ext()],
        # surface concentrations
        'sigma_in': [s.get_sigma_surf_in()],
        'sigma_ext': [s.get_sigma_surf_ext()],
        # charge numbers
        'charge_number_c_pos': [s.get_charge_number_c_pos()],
        'charge_number_c_neg': [s.get_charge_number_c_neg()],
        'charge_number_c_back': [s.get_charge_number_c_back()],
        # spatial resolution
        'res_in': [s.get_res_in()],
        'res_mem': [s.get_res_mem()],
        'res_ext': [s.get_res_ext()],
        'grid_spacing': [s.get_grid_spacing()],
        # MAIN RESULTS    
        'c_pos_in': [c_pos[i_in]],
        'c_pos_ext': [c_pos[i_ext]],
        'c_neg_in': [c_neg[i_in]],
        'c_neg_ext': [c_neg[i_ext]],    
        'v_in': [v[i_in]],
        'v_mem': [v[i_mem]],
        'v_ext': [v[i_ext]],
        'e_in': [e[i_in]],
        'e_mem': [e[i_mem]],
        'e_ext': [e[i_ext]],
        'q_in': [q[i_in]],
        'q_mem': [q[i_mem]],
        'q_ext': [q[i_ext]],
        'chem_v_pos_in': [chem_v_pos[i_in]],
        'chem_v_pos_ext': [chem_v_pos[i_ext]],
        'chem_v_neg_in': [chem_v_neg[i_in]],
        'chem_v_neg_ext': [chem_v_neg[i_ext]],
        'chem_f_pos_in': [chem_f_pos[i_in]],
        'chem_f_pos_ext': [chem_f_pos[i_ext]],
        'chem_f_neg_in': [chem_f_neg[i_in]],
        'chem_f_neg_ext': [chem_f_pos[i_in]],
        'pnp_lhs_in': [pnp_lhs[i_in]],
        'pnp_lhs_ext': [pnp_lhs[i_ext]],
        'pnp_rhs_in': [pnp_rhs[i_in]],
        'pnp_rhs_ext': [pnp_rhs[i_ext]],
        # save date for version control of code        
        'date': [str(datetime.datetime.now())],
        # access data by experiments
        'experiment_id': [s.get_experiment_id()],
    }
    
    new_row_df = pd.DataFrame.from_dict(row_dict)
    
    if main_results_df is None:
        return new_row_df
    else:
        new_row_df.index = [main_results_df.shape[0]]
        return pd.concat([main_results_df, new_row_df], axis=0)
        
def load_domain_from_df(df, dfLineNr):
    
    # set parameters of spine domain
    r_in = str(df.loc[dfLineNr, 'r_in'])
    r_mem = str(df.loc[dfLineNr, 'r_mem'])
    r_out =  str(df.loc[dfLineNr, 'r_ext'])

    eps_in = str(df.loc[dfLineNr, 'eps_in'])
    eps_mem = str(df.loc[dfLineNr, 'eps_mem'])
    eps_out = str(df.loc[dfLineNr, 'eps_ext'])

    res_in = int(df.loc[dfLineNr, 'res_in'])
    res_mem = int(df.loc[dfLineNr, 'res_mem'])
    res_out = int(df.loc[dfLineNr, 'res_ext'])

    # particel concentrations
    c_pos_in = str(df.loc[dfLineNr, 'c_pos_in_0'])
    c_neg_in = str(df.loc[dfLineNr, 'c_neg_in_0'])
    c_back_in = str(df.loc[dfLineNr, 'c_back_in_0'])


    c_pos_out = str(df.loc[dfLineNr, 'c_pos_ext_0'])
    c_neg_out = str(df.loc[dfLineNr, 'c_neg_ext_0'])
    c_back_out = str(df.loc[dfLineNr, 'c_back_ext_0'])

    rho_surf_in = str(df.loc[dfLineNr, 'sigma_in'])
    rho_surf_out = str(df.loc[dfLineNr, 'sigma_ext'])

    v_mem = str(df.loc[dfLineNr, 'membrane_potential'])

    d_phi_in = mpfr(df.loc[dfLineNr, 'd_phi_in'])
    d_phi_ext = mpfr(df.loc[dfLineNr, 'd_phi_ext'])
    
    if df.loc[dfLineNr, 'domain'] == 'neck':
        domain_tmp = SpineNeck(r_in, eps_in, r_mem, eps_mem, r_out, eps_out)
    elif df.loc[dfLineNr, 'domain'] == 'head':
        domain_tmp = SpineHead(r_in, eps_in, r_mem, eps_mem, r_out, eps_out)

    domain_tmp.set_intracellular_concentrations(c_pos_in, c_neg_in, c_back_in)
    domain_tmp.set_surface_concentrations(rho_surf_in, rho_surf_out)
    domain_tmp.set_extracellular_concentrations(c_pos_out, c_neg_out, c_back_out)
    domain_tmp.set_membrane_potential(v_mem)
    domain_tmp.set_delta_phi_in(d_phi_in)
    domain_tmp.set_delta_phi_ext(d_phi_ext)
    domain_tmp.solve_domain()
    
    """
    print(r_in, r_mem, r_out)
    print(eps_in, eps_mem, eps_out)
    print(res_in, res_mem, res_out)
    print(c_pos_in, c_neg_in, c_back_in)
    print(c_pos_out, c_neg_out, c_back_out)
    print(rho_surf_in, rho_surf_out)
    print(v_mem)
    print(d_phi_in, d_phi_ext)
    """
    return domain_tmp
