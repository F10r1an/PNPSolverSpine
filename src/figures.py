import matplotlib.pyplot as plt
import numpy as np
import copy
from .constants import constants
from .pnp_analysis import search_index, measure_double_layer_size, debye_lenght_theory, total_charge, free_charge, excess_charge, membrane_charge, double_layer_free_charge, capacitance_charges
from .file_io import load_domain_from_df

def plot_v_mem_cap_vs_pnp(fig, pos, df):
    ax = fig.add_axes(pos)
    
    d_mem = df.loc[:, 'r_mem'].values - df.loc[:, 'r_in'].values
    delta_phi =  df.loc[:, 'v_in'].values- df.loc[:, 'v_mem'].values
    v_mem = df.loc[:, 'membrane_potential'].values
    
    print(np.shape(delta_phi), np.shape(v_mem),delta_phi)
    
    for i,v in enumerate(v_mem):
        if v in [-0.07, -0.035, 0.0, 0.035]:
            ax.plot(delta_phi[i], v, 'kx', ms=4)
    ax.plot([-0.07, 0.035], [-0.07, 0.035], color='grey', linestyle='--', zorder=-1)
    
    #ax.legend(fontsize=8, frameon=False, loc=(0.7, 0.8))
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xticks([-0.07, -0.035, 0.0, 0.035])
    ax.set_yticks([-0.07, -0.035, 0.0, 0.035])
    ax.set_xticklabels(np.array(ax.get_xticks()*1.e3, dtype=np.int))
    ax.set_yticklabels(np.array(ax.get_xticks()*1.e3, dtype=np.int))

    ax.set_ylabel('Membrane \n potential [mV]', fontsize=8)
    ax.set_xlabel('Potential drop \n within membrane [mV]', fontsize=8)
    
    ax.tick_params(labelsize=8)

def compare_domain_parameters(dataframe, domain1_id, domain_2id, exceptions=[]):
    keys = dataframe.keys()
    for ex in exceptions:
        keys.pop(ex)
    
    all_equal = 1
    for param in keys:
        p1 = None
        p2 = None
        if p1 != p2:
            all_equal = 0
    
    return all_equal
    
    




def number_of_charges_vs_radius(fig, pos, df,side='intracellular'):
    
    #check_parameters(df, exceptions=['membrane_potential', 'r_in'])
    
    if side == 'intracellular':
        radius_key = 'r_in'
    else:
        raise ValueError('not yet implemented for extracelluar space')
    
    N_A = np.float(constants['N_A'])
    e = np.float(constants['e'])
    factor = 1./e
    
    # volume charge density intracellular
    total_charge_density = 300. * N_A
    free_charge_density = 150. * N_A # mol/m^3 = mmol (1.e-3 mol/liter)
    sodium_charge_density = 10 * N_A
    membrane_potential= -0.07  # 70 mV
    
    ############################
    if 'neck' in df.loc[:, 'domain']:
        raise AssertionError('not yet implemented for neck')
    min_cell_radius = 10.e-9  # 50 nm
    max_cell_radius = 500.e-9   # 1 \mu m
    radius_cable = np.linspace(min_cell_radius, max_cell_radius, 100)    
    surface_area = 4. * np.pi * np.square(radius_cable)
    volume = 4./3. * np.pi * np.power(radius_cable, 3)
    
    # usually to estimate the number of uncompensated charges a simple capacitor model is used
    specific_membrane_capacitance=1.e-2 # 1 muF / cm^2 = 0.01 F/m^2    
    membrane_capacitance = specific_membrane_capacitance * surface_area
    
    total_number_of_ions = volume * total_charge_density 
    number_of_free_ions = volume * free_charge_density
    number_of_sodium_ions = volume *sodium_charge_density
    number_of_capacitor_ions = np.abs(membrane_capacitance * membrane_potential / e )    
    #############################
    
    
    
    
    
    # put values from PNP-model here
    radius = []
    total = []
    free = []
    excess = []
    cap = []
    membrane = []
    dl = []
    
    
    df_ids = df.index

    for i in df_ids:
        if df.loc[i, 'membrane_potential'] == -0.07:
            domain = load_domain_from_df(df, i)

            if side == 'intracellular':
                radius_key = 'r_in'
                r = domain.get_r_in()

            else:
                raise ValueError('TODO')
            
            radius.append(df.loc[i, radius_key])
            total.append(total_charge(df, df_id=i, side=side)*factor)
            free.append(free_charge(df, df_id=i, side=side)*factor)
            excess.append(excess_charge(df, df_id=i, side=side)*factor)
            membrane.append(membrane_charge(df, df_id=i, side=side)*factor)
            dl.append(double_layer_free_charge(df, df_id=i, side=side)*factor)

            id_depol = search_index(df, {'membrane_potential': 0.0, radius_key: r})[0]

            cap.append(capacitance_charges(df_cap=df, id_1=i, id_2=id_depol, side=side)*factor)
     
    # sort
    sort_indices = np.argsort(radius)
    radius = np.array(radius)[sort_indices]
    total = np.array(total)[sort_indices]
    free = np.array(free)[sort_indices]
    excess = np.array(excess)[sort_indices]
    cap = np.array(cap)[sort_indices]
    membrane = np.array(membrane)[sort_indices]
    dl = np.array(dl)[sort_indices]
        
        
    ax = fig.add_axes(pos)
    
    rs=1.e9
    # cable model
    ax.plot(radius_cable*rs, total_number_of_ions, color='darkred', label='Total', lw=1)
    ax.plot(radius_cable*rs, number_of_free_ions, color='steelblue', label='Free', lw=1)
    ax.plot(radius_cable*rs, number_of_sodium_ions, color='forestgreen', label='Sodium', lw=1)
    ax.plot(radius_cable*rs, number_of_capacitor_ions, color='gold', label='Capacitor', lw=1)
    # end cable model
    
    ax.plot(radius*rs, np.abs(membrane), color='k', label='Membrane', ls='-', lw=1)
    ax.plot(radius*rs, total, 'rx', label='Total', color='darkred',ms=4)
    ax.plot(radius*rs, free, 'rx', label='Free', color='steelblue',ms=4)
    ax.plot(radius*rs, np.abs(excess), 'rx', label='Excess',ms=4, color='darkorchid')
    
    ax.plot(radius*rs, np.abs(cap), 'rx', label='Cap.', color='gold',ms=4)
    ax.plot(radius*rs, dl, 'rx', label='Double \nlayer', color='lime',ms=4)
    #ax.plot([radius[0],radius[-1]], [np.abs(excess)[0],np.abs(excess)[-1]], label='excess line')
    
    ax.legend(loc=(-0.0, 0.95), frameon=False, fontsize=8, ncol=2,
        columnspacing=0.5, handletextpad=0.3, handlelength=1)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    
     # labels 
    ax.set_xlabel('Radius [nm]', fontsize=8)
    ax.set_ylabel('Number of \nelementary charges', fontsize=8)
    ax.tick_params('both', labelsize=8)
    
    # style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.grid()

def plot_excess_charge_distributions(fig, pos, df_dl, ids):
    ax = fig.add_axes(pos)
    
    linestyles = ['-', '-']
    colors = ['k', 'limegreen']
    lws = [1.5, 0.75]
    
    
    for i,tmp_id in enumerate( ids ):
        domain_tmp = load_domain_from_df(df_dl, tmp_id)
        r = domain_tmp.get_grid_points()
        v_mem = domain_tmp.get_membrane_potential() * 1000.
        c_pos = domain_tmp.get_concentration_positive()
        c_neg = domain_tmp.get_concentration_negative()
        c_back = domain_tmp.get_concentration_background()
        ax.plot(r,c_pos-c_neg-c_back, c=colors[i], ls=linestyles[i], label=str(int(v_mem))+ ' mV', lw=lws[i])
        
        
        print(df_dl.loc[:, 'r_ext'])
        
    ax.legend(fontsize=8, frameon=False, loc=(0.05, 0.95))
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    
    r_in = np.unique(df_dl.loc[:, 'r_in'].values)
    r_min = r_in - 5.0e-9
    r_max = r_in + 10.0e-9
    
    ax.set_xlim(r_min, r_max)
    
    ax.set_yticks([0,200,400])
    ax.set_ylabel('Excess ion \nconcentration [mmol]', fontsize=8)
    ax.set_xticklabels(np.array(ax.get_xticks()*1.e9, dtype=np.int))
    ax.set_xlabel('Radius  [nm]', fontsize=8)
    
    ax.tick_params(labelsize=8)

def plot_double_layer_size(fig, pos, df, x_axis='radius'):
    
    ax = fig.add_axes(pos)
        
    
    labels = [
        [],
        [],
        [],
        [],
    ]
    
    
    domain_keys = [('head', 'intracellular'),('head', 'extracellular'),
                   ('neck', 'intracellular'),('neck', 'extracellular')]
    
    #domain_keys = domain_keys[:1]
    
    for n in range(len(domain_keys)):
        
        domain_type, region = domain_keys[n]

        
        
        x = []
        y_debye = []  # ignore fixed background charges
        y_debye_wbg = []  # include background charges
        y_dl_min = []
        y_dl_max = []
        y_dl_est = []
    
        domain_ids = search_index(df, {'domain': domain_type})
    
        for i in domain_ids:
            
            if x_axis == 'radius':
                if region == 'intracellular':

                    x.append(df.loc[i, 'r_in'])
                elif region == 'extracellular':
                    x.append(df.loc[i, 'r_ext']-df.loc[i,'r_mem'])
        
            else:
                raise ValueError('not yet implemented')
            print(region)
            dl_min, dl_max, dl_size = measure_double_layer_size(df, i,  side=region, visualize=False)
            y_dl_min.append(dl_min)
            y_dl_max.append(dl_max)
            y_dl_est.append(dl_size)
            #print(n,domain_type, region,x[-1], dl_min, dl_max)
            if region == 'intracellular':         
                y_debye.append(debye_lenght_theory(df, i))         
                y_debye_wbg.append(debye_lenght_theory(df, i, include_background=True))

        
        sort_indices = np.argsort(x)
        x = np.array(x)[sort_indices]
        y_dl_min = np.array(y_dl_min)[sort_indices]
        y_dl_max = np.array(y_dl_max)[sort_indices]
        y_dl_est = np.array(y_dl_est)[sort_indices]
        #print(x[sort_indices])
        ##ax.fill_between(x, y_dl_min, y_dl_max, 'kx', color=colors[n], alpha = 0.3, label='Double layer\n(numerical boundaries)')
        ##ax.plot(x, y_dl_max,'kx', color=colors[n])
        ##ax.plot(x, y_dl_min,'kx', color=colors[n])
        if domain_type== 'head' and region == 'intracellular':
            col = 'darkorange'
            marker = 'x'
            label = 'Head \nintracell.'
        elif domain_type == 'neck' and region == 'intracellular':
            col = 'darkblue'
            marker = 'x'
            label = 'Neck \nintracell.'
        elif domain_type== 'head' and region == 'extracellular':
            col = 'red'
            marker = 'x'
            label = 'Head \nextracell.'
        elif domain_type == 'neck' and region == 'extracellular':
            col = 'c'
            marker = 'x'
            label = 'Neck \nextracell.'
            
        ax.plot(x, y_dl_est,'kx-', marker=marker, color=col, ms=4, lw=1,
        label=label)
        print('double layer size:', y_dl_est)
        if n == 0 :
            ax.plot(x, y_debye, 'green', lw=1,)
            ax.plot(x, y_debye_wbg, 'k', lw=1)
    
    ####################
    # Legend
    ax.legend(loc=(-0.03,.95), fontsize=8, frameon=False, ncol=2,
    columnspacing=0.3, handlelength=1.2, handletextpad=0.3)

         
    #############
    # style
    to_nm = 1.e9
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlabel('Radius [nm]', fontsize=8)
    ax.set_ylabel('Double layer \nsize [nm]', fontsize=8)
    
    x_ticklabels = ax.get_xticks() * to_nm
    y_ticklabels = np.round(ax.get_yticks() * to_nm, 4)
    ax.set_xticklabels(x_ticklabels)
    ax.set_yticklabels(y_ticklabels)
    
    ax.tick_params('both', labelsize=8)
    
    

def plot_pnp_solution(fig, posA, posB, posC, posD, domain, mask_extracellular_space=False):
    """
    fig 3 d-g
    """
    
    r_max = domain.get_res_in()+domain.get_res_mem()+domain.get_res_ext()-3
    r = domain.get_grid_points() * 1.e9
    v_mem = domain.get_membrane_potential()
    
    # membrane
    xm0, xm1 = domain.get_res_in()-1, domain.get_res_in()+domain.get_res_mem()-1
    r_mem = r[xm0], r[xm1]
    
    phi = (domain.get_electric_potential() + v_mem) * 1000.
    subplot(fig=fig, pos=posA, x=r, y=phi, x_mem=r_mem, y_label='Potential $\Phi$ [mV]', constant_values=[v_mem*1000., 0],
    mask_membrane=False, mask_extracellular_space=mask_extracellular_space)
    
    e_field = domain.get_electric_field()  * 1.e-6
    subplot(fig=fig, pos=posB, x=r, y=e_field, x_mem=r_mem, y_label='Electric Field \n $-{d\Phi}/{dr}$ [V/$\mu m$]', constant_values=[0.], mask_extracellular_space=mask_extracellular_space)
    
    c_pos = domain.get_concentration_positive()
    c_pos_in = domain.get_c_pos_0_in()
    c_pos_ext = domain.get_c_pos_0_ext()
    subplot(fig=fig, pos=posC, x=r, y=c_pos, x_mem=r_mem, y_label='$c_{+}$ [mmol]', 
            constant_values=[c_pos_in, c_pos_ext], mask_membrane=True, mask_extracellular_space=mask_extracellular_space)
    
    c_neg = domain.get_concentration_negative()
    c_neg_in = domain.get_c_neg_0_in()
    c_neg_ext = domain.get_c_neg_0_ext()
    subplot(fig=fig, pos=posD, x=r, y=c_neg, x_mem=r_mem, y_label='$c_{-}$ [mmol]', 
            constant_values=[c_neg_in, c_neg_ext], mask_membrane=True, mask_extracellular_space=mask_extracellular_space)
            

def plot_domain_overview(domain, mask_extracellular_space=False):
    fig = plt.figure(dpi=300, figsize=(6,3))
    
    r_max = domain.get_res_in()+domain.get_res_mem()+domain.get_res_ext()-3
    r = domain.get_grid_points() * 1.e9
    v_mem = domain.get_membrane_potential()
    
    # membrane
    xm0, xm1 = domain.get_res_in()-1, domain.get_res_in()+domain.get_res_mem()-1
    r_mem = r[xm0], r[xm1]
    
    phi = (domain.get_electric_potential() + v_mem) * 1000.
    subplot(fig=fig, pos=221, x=r, y=phi, x_mem=r_mem, y_label='Potential [mV]', constant_values=[v_mem*1000., 0],
    mask_membrane=False, mask_extracellular_space=mask_extracellular_space,)
    # limit_y_range=(-0.2, 0.2))
    
    e_field = domain.get_electric_field()# * 1.e9
    subplot(fig=fig, pos=222, x=r, y=e_field, x_mem=r_mem, y_label='Electric \nField [V/m]', constant_values=[0.], mask_extracellular_space=mask_extracellular_space)
    
    c_pos = domain.get_concentration_positive()
    c_pos_in = domain.get_c_pos_0_in()
    c_pos_ext = domain.get_c_pos_0_ext()
    subplot(fig=fig, pos=223, x=r, y=c_pos, x_mem=r_mem, y_label='$c_{+}$ [mmol]', 
            constant_values=[c_pos_in, c_pos_ext], mask_membrane=True, mask_extracellular_space=mask_extracellular_space)
    
    c_neg = domain.get_concentration_negative()
    c_neg_in = domain.get_c_neg_0_in()
    c_neg_ext = domain.get_c_neg_0_ext()
    subplot(fig=fig, pos=224, x=r, y=c_neg, x_mem=r_mem, y_label='$c_{-}$ [mmol]', 
            constant_values=[c_neg_in, c_neg_ext], mask_membrane=True, mask_extracellular_space=mask_extracellular_space)
    """
    c_free = c_pos + c_neg #- domain.get_concentration_background()
    subplot(fig=fig, pos=236, x=r, y=c_free, x_mem=r_mem, y_label='Free Ion-Density\n [mmol]', 
            constant_values=[], mask_membrane=True, mask_extracellular_space=mask_extracellular_space)
    
    cum_free_charge = (domain.get_cumulative_positive() - domain.get_cumulative_negative()) * 1.e9
    subplot(fig=fig, pos=233, x=r, y=cum_free_charge, x_mem=r_mem, y_label='Cumulative Free \nCharge [nC/m?]', 
            constant_values=[], mask_membrane=False, mask_extracellular_space=mask_extracellular_space)
    """
    
    fig.tight_layout()
    
def plot_electric_field(domain, mask_extracellular_space=False):
    fig = plt.figure(dpi=300, figsize=(6,3))
    
    r_max = domain.get_res_in()+domain.get_res_mem()+domain.get_res_ext()-3
    r = domain.get_grid_points() * 1.e9
    
    # membrane
    xm0, xm1 = domain.get_res_in()-1, domain.get_res_in()+domain.get_res_mem()-1
    r_mem = r[xm0], r[xm1]
    
    e_tot = domain.get_electric_field()
    subplot(fig=fig, pos=231, x=r, y=e_tot, x_mem=r_mem, y_label='Electric Field [V/m]', constant_values=[0.],
    mask_membrane=False, mask_extracellular_space=mask_extracellular_space)
    
    e_pos = domain.get_electric_field_pos()
    subplot(fig=fig, pos=232, x=r, y=e_pos, x_mem=r_mem, y_label='Epos [V/m]', constant_values=[0.],
    mask_membrane=False, mask_extracellular_space=mask_extracellular_space)
    
    e_neg = domain.get_electric_field_neg()
    subplot(fig=fig, pos=233, x=r, y=e_neg, x_mem=r_mem, y_label='Eneg [V/m]', constant_values=[0.],
    mask_membrane=False, mask_extracellular_space=mask_extracellular_space)
    
    e_back = domain.get_electric_field_back()
    subplot(fig=fig, pos=234, x=r, y=e_back, x_mem=r_mem, y_label='Eback [V/m]', constant_values=[0.],
    mask_membrane=False, mask_extracellular_space=mask_extracellular_space)
    
    e_surf_in = domain.get_electric_field_mem_in()
    subplot(fig=fig, pos=235, x=r, y=e_surf_in, x_mem=r_mem, y_label='Esurfin [V/m]', constant_values=[0.],
    mask_membrane=False, mask_extracellular_space=mask_extracellular_space)
    
    e_surf_ext = domain.get_electric_field_mem_out()
    subplot(fig=fig, pos=236, x=r, y=e_surf_ext, x_mem=r_mem, y_label='Esurfext [V/m]', constant_values=[0.],
    mask_membrane=False, mask_extracellular_space=mask_extracellular_space)
    
    fig.tight_layout()

def plot_PNP_solution(domain):
    
    fig = plt.figure(dpi=300, figsize=(6,3))
    ax = fig.add_subplot(111)
    
    r = domain.get_grid_points() * 1.e9
    
    lhs = domain.get_pnp_lhs()
    rhs = domain.get_pnp_rhs()
    
    ###############################
    # compute total charge density
    """
    TODO: make function in spine_domain_base for chage density
    """
    eps_in = float(domain.eps_r_in * domain.CONST_eps_0)
    eps_out = float(domain.eps_r_out * domain.CONST_eps_0)
    
    c_pos = domain.get_concentration_positive() * float(domain.CONST_e) * (-1.)
    c_neg = domain.get_concentration_negative() * float(domain.CONST_e)
    c_back = domain.get_concentration_background() * float(domain.CONST_e) 
    c_pos[0: domain.res_in - 1] /= eps_in
    c_neg[0: domain.res_in - 1] /= eps_in
    c_back[0: domain.res_in - 1] /= eps_in
    c_pos[domain.res_in+domain.res_mem-1: domain.res_in+domain.res_mem+domain.res_out-2] /= eps_out
    c_neg[domain.res_in+domain.res_mem-1: domain.res_in+domain.res_mem+domain.res_out-2] /= eps_out
    c_back[domain.res_in+domain.res_mem-1: domain.res_in+domain.res_mem+domain.res_out-2] /= eps_out
    c_tot = (c_pos + c_neg + c_back) * float(domain.CONST_N_A)
    ######################################
    
    ax.plot(r, lhs, 'k-', label='lhs')
    ax.plot(r, rhs, 'g--', label='rhs')
    ax.plot(r, c_tot, 'b--', label='c_tot')

    
    ax.legend(fontsize=8, frameon=False)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    
    plt.show()
    
    
    

def subplot(fig, pos, x, y, x_mem, y_label, constant_values=[], constant_labels=[], mask_membrane=False, mask_extracellular_space=False, limit_y_range=False):
    """
    
    limit_y_range: False if the y-axis should not be constrained manually, else a tuple of y-boundaries
    """
    
    
    # add axes
    if type(pos) == int:
        ax = fig.add_subplot(pos)
    elif type(pos) == list:
        ax = fig.add_axes(pos)
    else: 
        print('bad position argument in function line_plot_with_markers')
        return 0  # exit function
    
    x_local = copy.copy(x)
    if mask_membrane == True:        
        x_local[np.logical_and( (x_local>= x_mem[0]) , (x_local<=x_mem[1]) )] = np.nan
        y[np.logical_and( (x>x_mem[0]) , (x<x_mem[1]) )] = np.nan
        
    if mask_extracellular_space == True:
        y[np.where(x_local >= x_mem[1])] = np.nan
        x_local[np.where(x_local > x_mem[1])] = np.nan
        
    
    ax.plot(x_local,y, 'k-', lw=1)
    
    for i, const in enumerate(constant_values):
        ax.plot([x_local[0],x_local[-1]],[const, const], lw =.7, ls='--', c='dimgrey')
        
    # membrane
    y_min, y_max = ax.get_ylim()
    ax.fill_between(x_mem,[y_min, y_min],[y_max, y_max], alpha=.2, color='gray', zorder=-1, lw=0)
    
    #########################
    # style
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ax.set_xlabel('Radius [nm]', fontsize=8)
    ax.set_ylabel(y_label, fontsize=8)
    
    #ax.set_xticks([0., 1.])
    #ax.set_xticklabels(['A', 'B'])
    
    #ax.set_yticks([0, 50, 100, 150])
    #ax.set_yticklabels([0, '', 100, ''])
    
    ax.tick_params('both', labelsize=8)
    
    if limit_y_range != False:
        ax.set_ylim(limit_y_range)
    
############################################

def potential_drop_through_membrane_charge(fig, pos1=[0.05, 0.05, 0.35, 0.9], pos2=[0.55, 0.05, 0.35, 0.9]):

    #fig = plt.figure(dpi=300, figsize=(6,2))
    #ax1 = fig.add_axes([0.05, 0.05, 0.35, 0.9])
    #ax2 = fig.add_axes([0.55, 0.05, 0.35, 0.9])
    ax1 = fig.add_axes(pos1)
    ax2 = fig.add_axes(pos2)

    # set parameters
    charge_densities = np.linspace(-0.05, 0.01, 1000, endpoint=True)  # density of membrane charge (intracellular side)
    radii = [np.float(20.e-9) * np.power(3.,i) for i in range(6)]  # radius of intracellular space
    vacuum_perm = np.float(constants['eps_0'])  # vacuum permittivits eps_0
    rel_perm = 5.  # rel. permittivity of membrane
    d_mem = 4.e-9  # thicknens of membrane

    # compute the potential drop across the membrane caused by the membrane charges
    # the potential drop is fully determined by the intracellular side
    # in a spherical domain the extracelluar side evokes no electric field inside the membrane (gauss' law)
    for r in radii:
        # total charge as a function of membrane charge density for fixed radius
        
        q_in = 4. * np.pi * np.square(r) * charge_densities  # total charge on intracellular side of membrane

        phi_mem = 1. / 4. / np.pi / vacuum_perm / rel_perm * q_in * ( 1./ r - 1./ (r+d_mem))

        label=str(int(r*1.e9//1))+ ' nm'
        ax2.plot(charge_densities, phi_mem,label=label,lw=1)

    # upper limit for d_mem = 10 nm
    d_mem = 10.e-9
    for r in radii:
        # total charge as a function of membrane charge density for fixed radius
        q_in = 4. * np.pi * np.square(r) * charge_densities  # total charge on intracellular side of membrane

        phi_mem = 1. / 4. / np.pi / vacuum_perm / rel_perm * q_in * ( 1./ r - 1./ (r+d_mem))


        ax2.plot(charge_densities, phi_mem, ls='--', lw=1)

    # membrane potential -0.07 V
    ax2.plot([charge_densities[0], charge_densities[-1]], [-0.07, -0.07], color='gray', lw=0.5, ls='--')

    ax2.legend(fontsize=8, loc=(0.0, .95), frameon=False)   
    ax2.set_xlabel('Charge density [$C/m^2$]', fontsize=8)
    ax2.set_ylabel('Potential drop [V]', fontsize=8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(labelsize=8)
    ######################
    # example for r = 20 nm, sigma = -0.02, eps_mem=5, eps_ext=60
    x = np.linspace(0., 50e-9, 500)
    q_in = 4. * np.pi * np.square(20e-9) * -0.02  # total membrane charge on intracellular side
    q_ext = q_in  # set extracell total membrane charge equal to q_in (results in lower charge density) 
    y = np.zeros(500)
    # extracellular field is evoked by both sides of membrane
    y[300:] = 1. / 4. / np.pi / x[300:] / vacuum_perm / 60 * (q_in + q_ext)  
    # inside membrane field is only evoked by q_in
    y[200:300] = ( 1. / 4. / np.pi / x[200:300] / vacuum_perm / rel_perm *q_in) 
    # make potential continuous
    y[200:300] = y[200:300] - y[299] + y[300]
    # constant potential in intracellular space (charge free spherecial domain)
    y[0:200]=y[200]
    ax1.plot(x,y, label='El. potential', lw=1)
    ax1.fill_between([20.e-9, 30.e-9],[0.,0.], [-5,-5], color='lightgrey', zorder=-10, label='Membrane')
    ax1.plot([20.e-9, 20e-9],[0.,-5.], color='firebrick', zorder=-10, label='Surface charge', lw=1)
    ax1.plot([30.e-9, 30e-9],[0.,-5.], color='firebrick', zorder=-10, lw=1)
    ax1.plot([35.e-9, 35e-9],[y[200], y[300]], color='black', zorder=-10, lw=1)
    ax1.plot([34.e-9, 36e-9],[y[200], y[200]], color='black', zorder=-10, lw=1)
    ax1.plot([34.e-9, 36e-9],[y[300], y[300]], color='black', zorder=-10, lw=1)
    ax1.text(36.e-9,-3.6, 'Potential drop', rotation=90,fontsize=8)
    ax1.set_xlabel('Radius [nm]', fontsize=8)
    ax1.set_xticks([0., 20.e-9, 30.e-9])
    ax1.set_xticklabels([0, r"$r_{in}$", r"$r_{in}+d$"])
    ax1.set_yticks([-5,0])
    ax1.set_ylabel('Potential [V]', fontsize=8)
    ax1.set_ylim([-5,0])
    ax1.set_xlim([0,5e-8])
    ax1.legend(fontsize=8, loc=(0.0,0.98), frameon=False)
    ax1.tick_params(labelsize=8)
    #####################

    #plt.show()


#####################################################
# Figure 1 sub-plots
#######################################################

def spine_comic(fig, pos, leg_loc=None, make_legend=True):
    ax = fig.add_axes(pos)
    img = plt.imread('./../../../paper/FIg3A.png')
    
    print(np.shape(img))
    
    ax.imshow(img, alpha=0.7)
    
    h1cx, h1cy = 2700, 880
    ax.text(2700,1050, '$r_{in}$', fontsize=8) 
    ax.plot([h1cx, 3100],[h1cy, h1cy], color='k', ls='--', marker='|') # line showing r_in
    ax.text(3600,300, '$r_{ext}$', fontsize=8)
    ax.plot([h1cx, 3550],[h1cy, 250], color='r', ls='--', marker='|')
    ax.text(3200, 1250, '$d$', fontsize=8)
    ax.plot([3100, 3220],[1100, 1130], color='r', marker='', ls='--')
    
    # neck dimensions
    n1cx, n1cy = 2836, 2592
    dy = 250
    # rext
    ax.text(3250,2250, '$r_{ext}$', fontsize=8)
    ax.plot([2650, 3200],[2200,2200], color='k', ls='--', marker='|')
    # r_in
    ax.text(2900,2700, '$r_{in}$', fontsize=8)
    ax.plot([2600, 2800],[2700,2700], color='r', ls='--', marker='|')    
    # d
    ax.text(3050, 2500, '$d$', fontsize=8)
    ax.plot([2800, 2950],[2450, 2450], color='r', marker='', ls='--')
    
    # surface charge 
    ax.text(2600,1650, '$\sigma^{in}$', fontsize=8)
    ax.text(3000,1650, '$\sigma^{ext}$', fontsize=8)
    # permittivity
    ax.text(2550,2000, '$\epsilon^{in}$', fontsize=8)
    ax.text(2820,2000, '$\epsilon^{mem}$', fontsize=8)
    ax.text(3250,2000, '$\epsilon^{ext}$', fontsize=8)

    # ion concentration
    ax.text(2350,550, '$c_+^{in}$', fontsize=8)
    ax.text(2350,800, '$c_-^{in}$', fontsize=8)
    ax.text(2350,1150, '$c_{back}^{in}$', fontsize=8)
    ax.text(1900,-150, '$c_{+}^{ext}$', fontsize=8)
    ax.text(2350,-150, '$c_-^{ext}$', fontsize=8)
    ax.text(2900,-150, '$c_{back}^{ext}$', fontsize=8)
    
    # head
    ax.plot([5266,5716,5818,6150,6150],[1260,1260,1260,1260,50],
    color='k', marker='|', ls='--')
    ax.plot([6150],[50], color='k', marker='x', ls='--')
   
    ax.text(5150,1480, '$P_A$', fontsize=8)
    ax.text(5500,1480, '$P_B$', fontsize=8)
    ax.text(5800,1480, '$P_C$', fontsize=8)
    ax.text(6100,1480, '$P_D$', fontsize=8)
    ax.text(5750,-50, '$\delta\Phi^{ext}$', fontsize=8)   
    ax.text(6200,-50, '$P_E$', fontsize=8)
    
    
    # neck dendrite
    ax.plot([5650, 5450, 5330, 5210,5150,],[2300,2300,2300,2300,3450,],
     color='k', marker='|', ls='--')
    ax.plot([5150],[3450], color='k', marker='x', ls='--') 
    ax.text(5650,2200, '$P_D$', fontsize=8)
    ax.text(5450,2200, '$P_C$', fontsize=8)
    ax.text(5220,2200, '$P_B$', fontsize=8)
    ax.text(5020,2200, '$P_A$', fontsize=8)
    ax.text(5230,3600, '$\delta\Phi^{in}$', fontsize=8)
    ax.text(4850,3600, '$P_F$', fontsize=8)
   
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    # legend
    if make_legend == True:
        if leg_loc == None:
	        xLeg = 100
	        yLeg = 200
	        dyLeg = 300    
        else: 
	        xLeg = leg_loc[0]
	        yLeg = leg_loc[1]
	        dyLeg = leg_loc[2]
        ax.text(xLeg,yLeg + 1 * dyLeg, 'd: membrane thickness', fontsize=8)
        ax.text(xLeg,yLeg + 2 * dyLeg, 'r: radius', fontsize=8)
        ax.text(xLeg,yLeg + 3 * dyLeg, '$c_+$: positive ion density', fontsize=8)
        ax.text(xLeg,yLeg + 4 * dyLeg, '$c_-$: negative ion density', fontsize=8)
        ax.text(xLeg,yLeg + 5 * dyLeg, '$c_{back}$: fixed background ion density', fontsize=8)
        ax.text(xLeg,yLeg + 6 * dyLeg, '$\epsilon$: permittivity', fontsize=8)
        ax.text(xLeg,yLeg + 7 * dyLeg, '$\sigma$: membrane charge density', fontsize=8)
        ax.text(xLeg,yLeg + 8 * dyLeg, '$\delta\Phi$: voltage perturbation', fontsize=8)
        
        # ax.text(xLeg,yLeg + 17 * dyLeg, 'in: intracellular', fontsize=8)
        # ax.text(xLeg,yLeg + 18 * dyLeg, 'ext: extracellular', fontsize=8)
        
        ax.text(xLeg,yLeg + 9.5 * dyLeg, 'A: center', fontsize=8)
        ax.text(xLeg,yLeg + 10.5 * dyLeg, 'B: inner membrane surface', fontsize=8)
        ax.text(xLeg,yLeg + 11.5 * dyLeg, 'C: extracell. ', fontsize=8)
        ax.text(xLeg,yLeg + 12.5 * dyLeg, '   membrane surface', fontsize=8)
        ax.text(xLeg,yLeg + 13.5 * dyLeg, 'D: exterior boundary', fontsize=8)
        ax.text(xLeg,yLeg + 14.5 * dyLeg, 'E & F: reference point', fontsize=8)
 
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
def fig_2d_shooting(fig, pos):
    color_v = 'darkblue'
    color_q = 'firebrick'
    
    
    #fig_opt_bnds = plt.figure(figsize=(4,3), dpi=300)
    #ax = fig_opt_bnds.add_subplot(111)
    ax = fig.add_axes(pos)
    
    x = np.linspace(0.,1.,100)
    fq = np.cos(np.linspace(0.,np.pi/2.,100))
    ax.plot(x, fq, ls='--', color=color_q, lw=1)

    #ax.fill_between(x,np.zeros(100),fq, color='green', alpha=.2)
    #ax.fill_between(x, np.ones(100), fq, color='darkorchid', alpha=.2)
    
    
    fv = np.exp(x / 1.8) - 0.9
    ax.plot(x, fv, ls='--', c=color_v, lw=1)
    #ax.fill_between(x, np.zeros(100), fv, facecolor='none', hatch='o', edgecolor='dimgrey', lw=0.3)
    #ax.fill_between(x, np.ones(100), fv, facecolor='none', hatch='x', edgecolor='dimgrey', lw=0.3)

    #ax.text(0.45, 1.02, 'V=-70 mV', fontsize=6)
    #ax.text(0.2, 1.02, 'V$<$-70 mV', fontsize=6)
    #ax.text(0.7, 1.02, 'V$>$-70 mV', fontsize=6)
    ########################
    ax.text(0.5, 0.96, r'$\frac{d\Phi}{dr}(r^{ext})<0$',fontsize=7, color=color_q, 
            backgroundcolor='none',)
    ax.text(0.11, 0.75, r'$\frac{d\Phi}{dr}(r^{ext})=0$', rotation=-45, fontsize=7, color=color_q,
            backgroundcolor='none',)
    ax.text(-0.07, 0.60, r'$\frac{d\Phi}{dr}(r^{ext})>0$', fontsize=7, color=color_q, 
            backgroundcolor='none',)

    ax.text(-0.07, 0.4, '$\Phi(r^{ext})<0$', fontsize=7, rotation=0, color=color_v, backgroundcolor='none',)
    ax.text(0.07, 0.04, '$\Phi(r^{ext})=0$', fontsize=7, rotation=32, color=color_v, backgroundcolor='none',)
    ax.text(0.2, -0.05, '$\Phi(r^{ext})>0$', fontsize=7, rotation=0, color=color_v, backgroundcolor='none',)
    #####################
    
    # draw P1, P2 & T
    ax.plot([0],[fq[0]], 'kx', ms=4)
    ax.plot([1],[fq[-1]], 'kx', ms=4)
    ax.text(-0.08,fq[0]+0.08, '$P_1$', fontsize=8)
    ax.text(1.0,fq[-1]+0.05, '$P_2$', fontsize=8)
    ax.plot([0.645],[0.53], 'kx', ms=4)
    ax.text(0.62,0.57, 'T', fontsize=8)

    ax.set_xticks([0., 1.])
    ax.set_yticks([0., 1.])
    ax.set_xticklabels(['$\delta \Phi^{in}_{min}$', '$\delta \Phi^{in}_{max}$'], fontsize=8)
    ax.set_yticklabels(['$\delta \Phi^{ext}_{min}$', '$\delta \Phi^{ext}_{max}$'], fontsize=8)

    ax.set_xlabel('', fontsize=8)
    ax.set_ylabel('', fontsize=8)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlim((-0.1, 1.1))
    ax.set_ylim((-0.1, 1.15))


def fig_1d_shooting(fig, pos):
    
    def y(a,x):
        y = a * np.square(x) 
        y = y[::-1]
        y = y - np.max(y)
        y = np.abs(y)   
        
        y = y + 0.5*np.sin(x*np.pi*2)
        return y
    
    ax = fig.add_axes(pos)
    
    x = np.linspace(0., 1., 100)
    
    ax.plot(x, y(2., x),'k-', lw=1)
    ax.plot(x, y(1.5, x),'k-', lw=1)
    ax.plot(x, y(2.5, x),'k-', lw=1)
    ax.plot([0],[0], 'ro', ms=4)
    ax.plot([1],[2], 'rx', ms=4)
    
    
    ax.set_xlim((-0.1, 1.1))
    ax.set_ylim((-0.2, 2.8))
    
    ax.set_xlabel('Radius', fontsize=8)
    ax.set_ylabel(r'$\frac{d\Phi}{dr}$,$~\Phi$', fontsize=10)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    #ax.text(1.0, 2.0, 'x', fontsize=8, verticalalignment='center', horizontalalignment='center')
    ax.text(1.1, 2.0, 'T', fontsize=8, verticalalignment='center', horizontalalignment='center')
    
    ax.text(0.06, 0.1, r'$\delta \Phi^{in}$, $\delta \Phi^{ext}$', rotation=0, fontsize=8, color='k',
            backgroundcolor='none')
    
    ax.text(0.6, 1., '?', fontsize=8)
    ax.text(0.6, 1.425, '?', fontsize=8)
    ax.text(0.6, 1.85, '?', fontsize=8)
    
    ax.set_xticks([0,1])
    ax.set_xticklabels([0, r'$r^{ext}$'], fontsize=8)
    ax.set_yticks([2])
    ax.set_yticklabels([0], fontsize=8)
    ax.plot(ax.get_xlim(),[2,2], ls='--', color='lightgrey', zorder=-100)

#####################################################
# Figure 2 sub-plots
#######################################################

#####################################################
# Figure 3 sub-plots
#######################################################

#####################################################
# Figure 4 sub-plots
#######################################################
