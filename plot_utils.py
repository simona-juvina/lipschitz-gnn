import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Optional, List

# user-defined modules
from utils import get_loss_acc_f1, get_result_dict

SMALL_SIZE  = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 14
BIGGEST_SIZE = 16

plt.rc('axes', titlesize=BIGGEST_SIZE)
plt.rc('axes', labelsize=BIGGEST_SIZE)
plt.rc('legend',fontsize=BIGGER_SIZE)

def plot_results_robustness(
    results_filename: str,
    save_folder: str,
    results_filename_at: Optional[str] = None,
    results_filename_rs: Optional[str] = None,
    db_list: Optional[List[str]] = None,
    network_list: Optional[List[str]] = None,
    hidden_dim_list: Optional[List[int]] = None,
    attacks_list: Optional[List[str]] = None,
    epsilon_list: Optional[List[float]] = None,
    lip_list: Optional[List[float]] = None,
    sigma_list: Optional[List[float]] = None,
    adv_epsilon_list: Optional[List[float]] = None,
    norm_each_list: Optional[List[str]] = None,
    save_fig: bool = False,
) -> None:
    """
    Plot the results for robustness evaluation.

    Args:
        results_filename (str): Path to the results file.
        save_folder (str): Path to the folder where the plots will be saved.
        results_filename_at (str, optional): Path to the adversarial training results file. Defaults to None.
        results_filename_rs (str, optional): Path to the random smoothing results file. Defaults to None.
        db_list (List[str], optional): List of database names to consider. Defaults to None.
        network_list (List[str], optional): List of network types to consider. Defaults to None.
        hidden_dim_list (List[int], optional): List of hidden dimensions to consider. Defaults to None.
        attacks_list (List[str], optional): List of attack types to consider. Defaults to None.
        epsilon_list (List[float], optional): List of epsilon values to consider. Defaults to None.
        lip_list (List[float], optional): List of LIP values to consider. Defaults to None.
        sigma_list (List[float], optional): List of sigma values to consider. Defaults to None.
        adv_epsilon_list (List[float], optional): List of advanced epsilon values to consider. Defaults to None.
        norm_each_list (List[str], optional): List of norm each node values to consider. Defaults to None.
        save_fig (bool, optional): Flag indicating whether to save the figures. Defaults to False.
    """ 
                        
    file_list = [results_filename, results_filename_at, results_filename_rs]
    all_db_perf = {}
    
    lip_values   = []
    adv_values   = []
    sigma_values = []
    j = 0
    for i, file in enumerate(file_list):
        if (file != None):
            df = pd.read_csv(file, delimiter=",")
            num_splits = len(df['split_num'].unique())
            if (j==0):
                dataset_list = df['db_name'].unique()
                hd_dim_list  = df['hidden_dim'].unique()
                hd_dim_list  = sorted(hd_dim_list, reverse=True)
                netw_list    = df['network_type'].unique()
                att_list     = df['attack'].unique()
                eps_list     = df['epsilon'].unique()
                norm_list    = df['norm_each_node'].unique()
                j+=1
            else:
                dataset_list = list(set(dataset_list).intersection(df['db_name'].unique()))
                hd_dim_list  = list(set(hd_dim_list).intersection(df['hidden_dim'].unique()))
                hd_dim_list  = sorted(hd_dim_list, reverse=True)
                netw_list    = list(set(netw_list).intersection(df['network_type'].unique()))
                att_list     = list(set(att_list).intersection(df['attack'].unique()))
                eps_list     = list(set(eps_list).intersection(df['epsilon'].unique()))
                norm_list    = list(set(norm_list).intersection(df['norm_each_node'].unique()))

            # Average results for the number of splits
            averaged_df = df.groupby(np.arange(len(df))//num_splits).mean()
            averaged_df.drop(['split_num'], axis=1, inplace=True)
            averaged_df.insert(0, "hidden_dim", list(df.hidden_dim[::num_splits]), True)
            averaged_df.insert(0, "attack", list(df.attack[::num_splits]), True)
            averaged_df.insert(0, "network_type", list(df.network_type[::num_splits]), True)
            averaged_df.insert(0, "db_name", list(df.db_name[::num_splits]), True)
            all_db_perf[i]  = get_result_dict(averaged_df)  
            
            if (i==0):
                lip_values  = [col.split('_')[2] for col in averaged_df.columns if 'attack_success' in col] 
                lip_values = [i if ((i == 'no') or (i == 'NO')) else str((float(i))) for i in lip_values]                
            if (i==1):
                adv_values = [col.split('_')[-1] for col in averaged_df.columns if 'after_attack_acc' in col]
            elif (i==2):
                sigma_values = [float(col.split('_')[-1]) for col in averaged_df.columns if 'after_attack_acc' in col]
          
    if (db_list):
        dataset_list = list(set(dataset_list) & set(db_list))
    if (network_list):
        netw_list =  list(set(network_list) & set(netw_list))
    if (hidden_dim_list):
        hd_dim_list = list(set(hd_dim_list).intersection(set([str(i) for i in hidden_dim_list])))
    if (attacks_list):
        att_list = list(set(att_list) & set(attacks_list))
    if (epsilon_list):
        eps_list = list(set(eps_list) & set(epsilon_list))
    if (norm_each_list):
        norm_list =  list(set(norm_each_list) & set(norm_list))
        
    eps_list    = [0] + sorted(eps_list, reverse=False)
    dict_values = lip_values + [f'adv {(val)}' for val in adv_values] + [f'sigma {(val)}' for val in sigma_values]
   
    for db_name in dataset_list:  
        for attack in att_list:
            for norm_each_node in norm_list:
                for hd in hd_dim_list:
                    for network_type in netw_list:
                        print_str = f'{db_name} {attack.upper()} attack\n {network_type.upper()} network'
                        perf_dict = {}
                        j=0
                        for i in all_db_perf.keys():
                            perf_dict = all_db_perf[i][db_name][attack][norm_each_node][network_type][str(hd)]
                            
                            # plot only for some values of epsilon
                            perf_dict     = {k: perf_dict.get(k, None) for k in eps_list}
                            perf_dict     = {k: v for k, v in perf_dict.items() if v}
                        
                            after_attack_acc = [perf_dict[key]['after_attack_acc'] for key in perf_dict.keys()]
                            original_acc     = perf_dict[(eps_list[1])]['original_acc']
                            try:
                                original_acc_val = perf_dict[(eps_list[1])]['orig_acc_val']
                            except:
                                pass
                            if (i==0):
                                vals = lip_values
                                dict_values[0] = str(perf_dict[(eps_list[1])]['upper_lip'])
                                lip_values[0]  = str(perf_dict[(eps_list[1])]['upper_lip'])
                            elif(i==1):
                                vals = adv_values
                            elif(i==2):
                                vals = sigma_values
                                
                            original_acc = original_acc[:, np.newaxis]
                            after_attack_acc = np.stack(after_attack_acc, axis=1)
                            after_attack_acc = np.hstack((original_acc, after_attack_acc))
                            
                            if (j==0):
                                after_attack_acc_full = after_attack_acc
                                j+=1
                            else:
                                after_attack_acc_full = np.vstack((after_attack_acc_full, after_attack_acc))

                        xi = list(range(len(eps_list)))
                        acc_dict = dict(zip(dict_values, after_attack_acc_full))
                        
                        if (lip_list and (0 in all_db_perf.keys())):
                            lip_list = [str(i) for i in lip_list]
                            lip_list = list(set(lip_values) & set(lip_list))
                            values_plot = [lip_values[0]] + [str(x) for x in sorted(lip_list, reverse=True)]
                        else:
                            values_plot = lip_values
                        try:
                            values_plot = [i for i in values_plot if float(i) <= float(values_plot[0])]
                        except:
                            pass
                        
                        if (adv_epsilon_list):
                            adv_epsilon_list = [str(i) for i in adv_epsilon_list]
                            adv_epsilon_list = list(set(adv_values) & set(adv_epsilon_list))
                            values_plot = values_plot + [f'adv {val}' for val in adv_epsilon_list]
                        else:
                            values_plot = values_plot + [f'adv {val}' for val in adv_values]

                        if (sigma_list):
                            sigma_values = [str(float(i)) for i in sigma_values]
                            sigma_list = [str(float(i)) for i in sigma_list]
                            sigma_list = list(set(sigma_list) & set(sigma_values))

                            values_plot = values_plot + [f'sigma {val}' for val in sigma_list]
                        else:
                            values_plot = values_plot + [f'sigma {val}' for val in sigma_values]
                            
                        legend = []
#                         eps_list[0] = 1
                        for i in range(len(values_plot)):
                            l_val = values_plot[i]
                            val = acc_dict[l_val]
                            if (sum(val) == 0):
                                continue
                            if ('adv' in l_val):
                                if ('random' in values_plot[i]):
                                    legend.append('AT')
                                else:
                                    legend.append('AT ' + r'$\epsilon$' + f'={l_val[4:]}')
                                line, = plt.plot(eps_list, val)
                            elif ('sigma' in l_val):
                                legend.append('RS ' + r'$\sigma$' + f'={l_val[6:]}')
                                line, = plt.plot(eps_list, val)
                            else:
                                if (i==0):
                                    legend.append(r'$\theta$' + f'={values_plot[i]}')
                                    line, = plt.plot(eps_list, val)
                                else:
                                    legend.append(r'$\overline{\vartheta}$' + f'={values_plot[i]}')
                                    try:
                                        line, = plt.plot(eps_list, val)
                                    except:
                                        pass
                        try:
                            plt.margins(x=0)
                            plt.legend(legend, loc='upper right')
                            plt.xticks(xi, eps_list)
                            plt.grid()
                            plt.xlabel('ε')
                            plt.ylabel('After attack accuracy')
                            if (eps_list[-1] > 2):
                                plt.xlim([0, 600])
                            else:
                                plt.xlim([0, 0.8])
                            plt.xscale('linear') 
                            if save_fig:
                                len_hd = len(hd.split(','))
                                plt.savefig(f'{save_folder}/{db_name.replace(":", "_")}_{attack}_{network_type.upper()}_{len_hd}_adv.pdf', bbox_inches="tight")
                                plt.close()  
                            else:
                                plt.title(print_str)
                                plt.show()
                        except:
                            pass
                        
        
def plot_results_attacks(results_filename_list: List[str], save_folder: str, db_list: Optional[List[str]] = None,
                         network_list: Optional[List[str]] = None, hidden_dim_list: Optional[List[str]] = None,
                         attacks_list: Optional[List[str]] = None, epsilon_list: Optional[List[float]] = None,
                         lip_list: Optional[List[float]] = None, adv_epsilon_list: Optional[List[float]] = None,
                         norm_each_list: Optional[List[str]] = None, save_fig: bool = False,
                         plot_unconstrained: bool = False) -> None:
    """
    Plots the results of different attacks.

    Args:
        results_filename_list: List of filenames containing results data from different attacks.
        save_folder: Path to the folder where the plots will be saved.
        db_list: List of database names to include. Defaults to None.
        network_list: List of network types to include. Defaults to None.
        hidden_dim_list: List of hidden dimensions to include. Defaults to None.
        attacks_list: List of attack types to include. Defaults to None.
        epsilon_list: List of epsilon values to include. Defaults to None.
        lip_list: List of Lipschitz values to include. Defaults to None.
        adv_epsilon_list: List of adversarial epsilon values to include. Defaults to None.
        norm_each_list: List of normalization values to include. Defaults to None.
        save_fig: Indicates whether to save the figures. Defaults to False.
        plot_unconstrained: Indicates whether to plot unconstrained results. Defaults to False.
    """

    df_append = pd.DataFrame()
    for results_filename in results_filename_list:
        df = pd.read_csv(results_filename, delimiter=",")
        num_splits = len(df['split_num'].unique())
        # Average results for the number of splits
        averaged_df = df.groupby(np.arange(len(df))//num_splits).mean()
        averaged_df.drop(['split_num'], axis=1, inplace=True)
        averaged_df.insert(0, "hidden_dim", list(df.hidden_dim[::num_splits]), True)
        averaged_df.insert(0, "attack", list(df.attack[::num_splits]), True)
        averaged_df.insert(0, "network_type", list(df.network_type[::num_splits]), True)
        averaged_df.insert(0, "db_name", list(df.db_name[::num_splits]), True)
        df_append = df_append.append(averaged_df, ignore_index=True)

    all_db_perf = get_result_dict(df_append)

    dataset_list = df_append['db_name'].unique()
    hd_dim_list  = df_append['hidden_dim'].unique()
    hd_dim_list  = sorted(hd_dim_list, reverse=True)
    netw_list    = df_append['network_type'].unique()
    att_list     = df_append['attack'].unique()
    eps_list     = df_append['epsilon'].unique()
    norm_list    = df_append['norm_each_node'].unique()

    if db_list:
        dataset_list = db_list
    if network_list:
        netw_list = network_list
    if hidden_dim_list:
        hd_dim_list = hidden_dim_list
    if attacks_list:
        att_list = attacks_list
    if epsilon_list:
        eps_list = list(set(eps_list) & set(epsilon_list))
    if norm_each_list:
        norm_list = norm_each_list
        
    eps_list = sorted(eps_list, reverse=False)
    eps_list = [int(i) for i in eps_list]
    lip_values = [col.split('_')[-1] for col in averaged_df.columns if 'attack_success' in col]
    lip_values = [i if ((i == 'no') or (i == 'NO')) else str(float(i)) for i in lip_values]
    dict_values = lip_values
    
    for db_name in dataset_list:
        for norm_each_node in norm_list:
            for hd in hd_dim_list:
                for network_type in netw_list:
                    legend = []
                    for attack in att_list:
                        print_str = f'Attack comparison on {db_name} \n {network_type.upper()} network'
                        perf_dict = all_db_perf[db_name][attack][norm_each_node][network_type][str(hd)]

                        # plot only for some values of epsilon
                        perf_dict     = {k: perf_dict.get(k, None) for k in eps_list}
                        perf_dict     = {k: v for k, v in perf_dict.items() if v}

                        eps_list = [0] + list(perf_dict.keys())
                        xi = list(range(len(eps_list)))

                        after_attack_acc = [perf_dict[key]['after_attack_acc'] for key in perf_dict.keys()]
                        original_acc     = perf_dict[eps_list[1]]['original_acc']

                        original_acc     = original_acc[:, np.newaxis]

                        after_attack_acc = np.stack(after_attack_acc, axis=1)
                        after_attack_acc = np.hstack((original_acc, after_attack_acc))

                        dict_values[0] = str(perf_dict[eps_list[1]]['upper_lip'])
                        lip_values[0]  = str(perf_dict[eps_list[1]]['upper_lip'])
                        acc_dict = dict(zip(dict_values, after_attack_acc))

                        if lip_list:
                            lip_values_plot = [lip_values[0]] + [str(x) for x in sorted(lip_list, reverse=True)]
                        else:
                            lip_values_plot =  [lip_values[0]]
                        try:
                            lip_values_plot = [i for i in lip_values_plot if float(i) <= float(lip_values_plot[0])]
                        except:
                            pass

                        if attack == att_list[0]:
                            unconstr_lip = lip_values_plot[0]

                        for i in range(len(lip_values_plot)):
                            l_val = lip_values_plot[i]
                            val = acc_dict[str(l_val)]
                    
                            if (sum(val) == 0):
                                continue
                            else:
                                att_str = attack.split('_')[-1].upper().ljust(4)
                                att_str_c = attack.split('_')
                                att_str = f'{att_str_c[0]} {att_str_c[-1]} '.upper()
                                
                                if ('DLR' not in att_str):
                                    att_str = att_str + ' '
                                if (i==0):
                                    if (plot_unconstrained==True):
                                        legend.append(att_str + r'$\theta$' + f'={unconstr_lip}')
                                        plt.plot(eps_list,val)
                                else:
                                    legend.append(att_str)
                                    plt.plot(eps_list,val)
        

                    try:
                        plt.margins(x=0)
                        plt.legend(legend, loc='lower left')
                        plt.xticks(xi, eps_list)
                        plt.grid()
                        plt.xlabel('ε')
                        plt.ylabel('After attack accuracy')
                        if (eps_list[-1] > 2):
                            plt.xlim([0, 600])
                        else:
                            plt.xlim([0, 0.8])
                        plt.xscale('linear')
                        if save_fig:
                            lenhd = len(hd.split(','))
                            plt.savefig(f'{save_folder}/{db_name.replace(":", "_")}_{network_type.upper()}_{lenhd}_attacks.pdf', bbox_inches="tight")
                            plt.close()
                        else:
                            plt.title(print_str)
                            plt.show()
                    except:
                        pass
                    



def plot_results_sn(
    results_filename: str,
    save_folder: str,
    results_filename_sn: str,
    db_list: Optional[List[str]] = None,
    network_list: Optional[List[str]] = None,
    hidden_dim_list: Optional[List[int]] = None,
    attacks_list: Optional[List[str]] = None,
    epsilon_list: Optional[List[float]] = None,
    lip_list: Optional[List[float]] = None,
    sn_list: Optional[List[float]] = None,
    norm_each_list: Optional[List[str]] = None,
    save_fig: bool = False,
) -> None:
    """
    Plot the results for robustness evaluation.

    Args:
        results_filename (str): Path to the results file.
        save_folder (str): Path to the folder where the plots will be saved.
        results_filename_sn (str, optional): Path to the spectral normalization results file. Defaults to None.
        db_list (List[str], optional): List of database names to consider. Defaults to None.
        network_list (List[str], optional): List of network types to consider. Defaults to None.
        hidden_dim_list (List[int], optional): List of hidden dimensions to consider. Defaults to None.
        attacks_list (List[str], optional): List of attack types to consider. Defaults to None.
        epsilon_list (List[float], optional): List of epsilon values to consider. Defaults to None.
        lip_list (List[float], optional): List of Lipschitz values to consider for the constrained models. Defaults to None.
        sn_list (List[float], optional): List of upper Lipschitz values to consider for spectral normalization. Defaults to None.
        norm_each_list (List[str], optional): List of norm each node values to consider. Defaults to None.
        save_fig (bool, optional): Flag indicating whether to save the figures. Defaults to False.
    """ 
                        
    file_list = [results_filename, results_filename_sn]
    all_db_perf = {}
    
    lip_values   = []
    adv_values   = []
    sigma_values = []
    j = 0
    for i, file in enumerate(file_list):
        if (file != None):
            df = pd.read_csv(file, delimiter=",")
            num_splits = len(df['split_num'].unique())
            if (j==0):
                dataset_list = df['db_name'].unique()
                hd_dim_list  = df['hidden_dim'].unique()
                hd_dim_list  = sorted(hd_dim_list, reverse=True)
                netw_list    = df['network_type'].unique()
                att_list     = df['attack'].unique()
                eps_list     = df['epsilon'].unique()
                norm_list    = df['norm_each_node'].unique()
                j+=1
            else:
                dataset_list = list(set(dataset_list).intersection(df['db_name'].unique()))
                hd_dim_list  = list(set(hd_dim_list).intersection(df['hidden_dim'].unique()))
                hd_dim_list  = sorted(hd_dim_list, reverse=True)
                netw_list    = list(set(netw_list).intersection(df['network_type'].unique()))
                att_list     = list(set(att_list).intersection(df['attack'].unique()))
                eps_list     = list(set(eps_list).intersection(df['epsilon'].unique()))
                norm_list    = list(set(norm_list).intersection(df['norm_each_node'].unique()))

            # Average results for the number of splits
            averaged_df = df.groupby(np.arange(len(df))//num_splits).mean()
            averaged_df.drop(['split_num'], axis=1, inplace=True)
            averaged_df.insert(0, "hidden_dim", list(df.hidden_dim[::num_splits]), True)
            averaged_df.insert(0, "attack", list(df.attack[::num_splits]), True)
            averaged_df.insert(0, "network_type", list(df.network_type[::num_splits]), True)
            averaged_df.insert(0, "db_name", list(df.db_name[::num_splits]), True)
            all_db_perf[i]  = get_result_dict(averaged_df)  
            
            if (i==0):
                lip_values  = [col.split('_')[2] for col in averaged_df.columns if 'attack_success' in col] 
                lip_values = [i if ((i == 'no') or (i == 'NO')) else str((float(i))) for i in lip_values]                
            if (i==1):
                lip_values_SN = [col.split('_')[2] for col in averaged_df.columns if 'attack_success' in col][1:]
                lip_values_SN = [str((float(i))) for i in lip_values_SN] 
    if (db_list):
        dataset_list = list(set(dataset_list) & set(db_list))
    if (network_list):
        netw_list =  list(set(network_list) & set(netw_list))
    if (hidden_dim_list):
        hd_dim_list = list(set(hd_dim_list).intersection(set([str(i) for i in hidden_dim_list])))
    if (attacks_list):
        att_list = list(set(att_list) & set(attacks_list))
    if (epsilon_list):
        eps_list = list(set(eps_list) & set(epsilon_list))
    if (norm_each_list):
        norm_list =  list(set(norm_each_list) & set(norm_list))
        
    eps_list    = [0] + sorted(eps_list, reverse=False)
    dict_values = lip_values + [f'SN {(val)}' for val in lip_values_SN]
   
    for db_name in dataset_list:  
        for attack in att_list:
            for norm_each_node in norm_list:
                for hd in hd_dim_list:
                    for network_type in netw_list:
                        print_str = f'{db_name} {attack.upper()} attack\n {network_type.upper()} network'
                        perf_dict = {}
                        j=0
                        for i in all_db_perf.keys():
                            perf_dict = all_db_perf[i][db_name][attack][norm_each_node][network_type][str(hd)]
                            
                            # plot only for some values of epsilon
                            perf_dict     = {k: perf_dict.get(k, None) for k in eps_list}
                            perf_dict     = {k: v for k, v in perf_dict.items() if v}
                        
                            after_attack_acc = [perf_dict[key]['after_attack_acc'] for key in perf_dict.keys()]
                            original_acc     = perf_dict[int(eps_list[1])]['original_acc']
                            try:
                                original_acc_val = perf_dict[int(eps_list[1])]['orig_acc_val']
                            except:
                                pass
                            if (i==0):
                                vals = lip_values
                                dict_values[0] = str(perf_dict[int(eps_list[1])]['upper_lip'])
                                lip_values[0]  = str(perf_dict[int(eps_list[1])]['upper_lip'])
                            elif(i==1):
                                vals = lip_values_SN
                                
                            original_acc = original_acc[:, np.newaxis]
                            after_attack_acc = np.stack(after_attack_acc, axis=1)
                            after_attack_acc = np.hstack((original_acc, after_attack_acc))
                            
                            if (j==0):
                                after_attack_acc_full = after_attack_acc
                                j+=1
                            else:
                                after_attack_acc_full = np.vstack((after_attack_acc_full, after_attack_acc))

                        xi = list(range(len(eps_list)))
                        acc_dict = dict(zip(dict_values, after_attack_acc_full))
                        
                        if (lip_list and (0 in all_db_perf.keys())):
                            lip_list = [str(i) for i in lip_list]
                            lip_list = list(set(lip_values) & set(lip_list))
                            values_plot = [lip_values[0]] + [str(x) for x in sorted(lip_list, reverse=True)]
                        else:
                            values_plot = lip_values
                        try:
                            values_plot = [i for i in values_plot if float(i) <= float(values_plot[0])]
                        except:
                            pass
                        
                        if (sn_list):
                            sn_values = [str(float(i)) for i in lip_values_SN]
                            sn_list = [str(float(i)) for i in sn_list]
                            sn_list = list(set(sn_list) & set(lip_values_SN))

                            values_plot = values_plot + [f'SN {val}' for val in sn_list]
                        else:
                            values_plot = values_plot + [f'SN {val}' for val in lip_values_SN]
 
                        try:
                            values_plot = [i for i in values_plot if float(i) <= float(values_plot[0])]
                        except:
                            pass
                    
                        legend = []
                        for i in range(len(values_plot)):
                            l_val = values_plot[i]
                            val = acc_dict[l_val]
                            if (sum(val) == 0):
                                continue
                            if ('SN' in l_val):
                                legend.append('SN')
                                line, = plt.plot(eps_list,val)
                            else:
                                if (i==0):
                                    legend.append(r'Baseline')
                                    line, = plt.plot(eps_list,val)
                                else:
                                    legend.append('Constrained')
                                    try:
                                        line, = plt.plot(eps_list,val)
                                    except:
                                        pass
                        try:
                            plt.margins(x=0)
                            plt.legend(legend, loc='upper right')
                            plt.xticks(xi, eps_list)
                            plt.grid()
                            plt.xlabel('ε')
                            plt.ylabel('After attack accuracy')
                            if (eps_list[-1] > 2):
                                plt.xlim([0, 600])
                            else:
                                plt.xlim([0, 0.8])
                            plt.xscale('linear') 
                            if save_fig:
                                len_hd = len(hd.split(','))
                                plt.savefig(f'{save_folder}/{db_name.replace(":", "_")}_{attack}_{network_type.upper()}_{len_hd}_SN.pdf', bbox_inches="tight")
                                plt.close()  
                            else:
                                plt.title(print_str)
                                plt.show()
                        except:
                            pass

    
def plot_results_graph_defenses(
    results_filename: str,
    save_folder: str,
    results_filename_svd: str,
    results_filename_jaccard: str,
    results_filename_rgcn: str,
    db_list: Optional[List[str]] = None,
    network_list: Optional[List[str]] = None,
    hidden_dim_list: Optional[List[int]] = None,
    attacks_list: Optional[List[str]] = None,
    epsilon_list: Optional[List[float]] = None,
    lip_list: Optional[List[float]] = None,
    svd_list: Optional[List[int]] = None,
    jaccard_list: Optional[List[float]] = None,
    rgcn_list: Optional[List[float]] = None,
    norm_each_list: Optional[List[str]] = None,
    save_fig: bool = False,
) -> None:
    """
    Plot the results for comparison with graph defense models.

    Args:
        results_filename (str): Path to the results file.
        save_folder (str): Path to the folder where the plots will be saved.
        results_filename_rgcn (str, optional): Path to the RGCN results file. Defaults to None.
        results_filename_svd (str, optional): Path to the SVD-GCN results file. Defaults to None.
        results_filename_jaccard (str, optional): Path to the GCN-Jaccard results file. Defaults to None.
        db_list (List[str], optional): List of database names to consider. Defaults to None.
        network_list (List[str], optional): List of network types to consider. Defaults to None.
        hidden_dim_list (List[int], optional): List of hidden dimensions to consider. Defaults to None.
        attacks_list (List[str], optional): List of attack types to consider. Defaults to None.
        epsilon_list (List[float], optional): List of epsilon values to consider. Defaults to None.
        lip_list (List[float], optional): List of Lipschitz values to consider for the constrained models. Defaults to None.
        svd_list (List[float], optional): List of SVD-GCN parameters. Defaults to None.
        jaccard_list (List[float], optional): List of GCN-Jaccard parameters. Defaults to None.
        rgcn_list (List[float], optional): List of RGCN parameters. Defaults to None.
        norm_each_list (List[str], optional): List of norm each node values to consider. Defaults to None.
        save_fig (bool, optional): Flag indicating whether to save the figures. Defaults to False.
    """ 
                        
                        
    file_list = [results_filename, results_filename_svd, results_filename_jaccard, results_filename_rgcn] 
    all_db_perf = {}
    
    lip_values   = []
    svd_values   = []
    jaccard_values = []
    rgcn_values = []
    j = 0
    for i, file in enumerate(file_list):
        if (file != None):
            df = pd.read_csv(file, delimiter=",")
            num_splits = len(df['split_num'].unique())
            
            if (j==0):
                dataset_list = df['db_name'].unique()
                hd_dim_list  = df['hidden_dim'].unique()
                hd_dim_list  = sorted(hd_dim_list, reverse=True)
                netw_list    = df['network_type'].unique()
                att_list     = df['attack'].unique()
                eps_list     = df['epsilon'].unique()
                norm_list    = df['norm_each_node'].unique()
                j+=1
            else:
                df = df.sort_values(by=['epsilon','split_num'])
                dataset_list = list(set(dataset_list).intersection(df['db_name'].unique()))
                hd_dim_list  = list(set(hd_dim_list).intersection(df['hidden_dim'].unique()))
                hd_dim_list  = sorted(hd_dim_list, reverse=True)
                netw_list    = list(set(netw_list).intersection(df['network_type'].unique()))
                att_list     = list(set(att_list).intersection(df['attack'].unique()))
                eps_list     = list(set(eps_list).intersection(df['epsilon'].unique()))
                norm_list    = list(set(norm_list).intersection(df['norm_each_node'].unique()))

            # Average results for the number of splits
            averaged_df = df.groupby(np.arange(len(df))//num_splits).mean()
            averaged_df.drop(['split_num'], axis=1, inplace=True)
            try:
                averaged_df.drop(['hidden_dim'], axis=1, inplace=True)
            except:
                pass
            averaged_df.insert(0, "hidden_dim", list(df.hidden_dim[::num_splits]), True)
            averaged_df.insert(0, "attack", list(df.attack[::num_splits]), True)
            averaged_df.insert(0, "network_type", list(df.network_type[::num_splits]), True)
            averaged_df.insert(0, "db_name", list(df.db_name[::num_splits]), True)
            all_db_perf[i]  = get_result_dict(averaged_df)  
            
            if (i==0):
                lip_values  = [col.split('_')[2] for col in averaged_df.columns if 'attack_success' in col] 
                lip_values = [i if ((i == 'no') or (i == 'NO')) else str((float(i))) for i in lip_values]                
            if (i==1):
                svd_values = [col.split('_')[-1] for col in averaged_df.columns if 'after_attack_acc' in col]
            elif (i==2):
                jaccard_values = [col.split('_')[-1] for col in averaged_df.columns if 'after_attack_acc' in col]
            elif (i==3):
                rgcn_values = [col.split('_')[-1] for col in averaged_df.columns if 'after_attack_acc' in col]

    if (db_list):
        dataset_list = list(set(dataset_list) & set(db_list))
    if (attacks_list):
        att_list = list(set(att_list) & set(attacks_list))
    if (epsilon_list):
        eps_list = list(set(eps_list) & set(epsilon_list))
    if (norm_each_list):
        norm_list =  list(set(norm_each_list) & set(norm_list))
        
    eps_list    = [0] + sorted(eps_list, reverse=False)
    dict_values = lip_values + [f'svd {(val)}' for val in svd_values] + [f'jaccard {float(val)}' for val in jaccard_values] + [f'rgcn {float(val)}' for val in rgcn_values]
    
    network_type = 'gcn'
    for db_name in dataset_list:  
        for attack in att_list:
            for norm_each_node in norm_list:
                print_str = f'{db_name} {attack.upper()} attack\n {network_type.upper()} network'
                perf_dict = {}
                j=0
                for i in all_db_perf.keys():
                    perf_dict = all_db_perf[i][db_name][attack][norm_each_node]
                    netw = next(iter(perf_dict.keys()))
                    perf_dict = all_db_perf[i][db_name][attack][norm_each_node][netw]
                    hdl = next(iter(perf_dict.keys()))
                    perf_dict = all_db_perf[i][db_name][attack][norm_each_node][netw][hdl]

                    perf_dict     = {k: perf_dict.get(k, None) for k in eps_list}
                    perf_dict     = {k: v for k, v in perf_dict.items() if v}

                    after_attack_acc = [perf_dict[key]['after_attack_acc'] for key in perf_dict.keys()]
                    original_acc     = perf_dict[int(eps_list[1])]['original_acc']
                    try:
                        original_acc_val = perf_dict[int(eps_list[1])]['orig_acc_val']
                    except:
                        pass
                    if (i==0):
                        vals = lip_values
                        dict_values[0] = str(perf_dict[int(eps_list[1])]['upper_lip'])
                        lip_values[0]  = str(perf_dict[int(eps_list[1])]['upper_lip'])
                    elif(i==1):
                        vals = svd_values
                    elif(i==2):
                        vals = jaccard_values
                    elif(i==3):
                        vals = rgcn_values
                        
                    original_acc = original_acc[:, np.newaxis]
                    after_attack_acc = np.stack(after_attack_acc, axis=1)
                    after_attack_acc = np.hstack((original_acc, after_attack_acc))

                    if (j==0):
                        after_attack_acc_full = after_attack_acc
                        j+=1
                    else:
                        after_attack_acc_full = np.vstack((after_attack_acc_full, after_attack_acc))

                xi = list(range(len(eps_list)))
                acc_dict = dict(zip(dict_values, after_attack_acc_full))

                if (lip_list and (0 in all_db_perf.keys())):
                    lip_list = [str(i) for i in lip_list]
                    lip_list = list(set(lip_values) & set(lip_list))
                    values_plot = [lip_values[0]] + [str(x) for x in sorted(lip_list, reverse=True)]
                else:
                    values_plot = lip_values
                try:
                    values_plot = [i for i in values_plot if float(i) <= float(values_plot[0])]
                except:
                    pass

                if (svd_list):
                    svd_list = [str(i) for i in svd_list]
                    svd_list = list(set(svd_values) & set(svd_list))
                    values_plot = values_plot + [f'svd {val}' for val in svd_list]
                else:
                    values_plot = values_plot + [f'svd {val}' for val in svd_values]

                if (jaccard_list):
                    jaccard_values = [str(float(i)) for i in jaccard_list]
                    jaccard_list = [str(float(i)) for i in jaccard_list]
                    jaccard_list = list(set(jaccard_list) & set(jaccard_values))

                    values_plot = values_plot + [f'jaccard {val}' for val in jaccard_list]
                else:
                    values_plot = values_plot + [f'jaccard {float(val)}' for val in jaccard_values]
                    
                if (rgcn_list):
                    rgcn_values = [str(float(i)) for i in rgcn_list]
                    rgcn_list = [str(float(i)) for i in rgcn_list]
                    rgcn_list = list(set(rgcn_list) & set(rgcn_values))

                    values_plot = values_plot + [f'rgcn {val}' for val in rgcn_list]
                else:
                    values_plot = values_plot + [f'rgcn {float(val)}' for val in rgcn_values]
                    
                legend = []
                for i in range(len(values_plot)):
                    l_val = values_plot[i]
                    val = acc_dict[l_val]
                    
                    if (sum(val) == 0):
                        continue
                    if ('jaccard' in l_val):
                        legend.append('GCN-Jaccard')
                        line, = plt.plot(eps_list, val)
                    elif ('svd' in l_val):
                        legend.append('SVD-GCN')
                        line, = plt.plot(eps_list, val)
                    elif ('rgcn' in l_val):
                        legend.append('RGCN')
                        line, = plt.plot(eps_list, val)
                    else:
                        if (i==0):
                            legend.append('Baseline')
                            line, = plt.plot(eps_list, val)
                        else:
                            legend.append('Constrained')
                            try:
                                line, = plt.plot(eps_list, val)
                            except:
                                pass
                try:
                    plt.margins(x=0)
                    plt.legend(legend, loc='upper right')
                    plt.xticks(xi, eps_list)
                    plt.grid()
                    plt.xlabel('ε')
                    plt.ylabel('After attack accuracy')
                    if (eps_list[-1] > 2):
                        plt.xlim([0, 600])
                    else:
                        plt.xlim([0, 0.8])
                    plt.xscale('linear') 
                    if save_fig:
                        plt.savefig(f'{save_folder}/{db_name.replace(":", "_")}_{attack}_{network_type.upper()}_graphs.pdf', bbox_inches="tight")
                        plt.close()  
                    else:
                        plt.title(print_str)
                        plt.show()
                except:
                    pass



def plot_results_lipschitz(
    results_filename: str,
    save_folder: str,
    db_list: Optional[List[str]] = None,
    network_list: Optional[List[str]] = None,
    hidden_dim_list: Optional[List[int]] = None,
    attacks_list: Optional[List[str]] = None,
    epsilon_list: Optional[List[float]] = None,
    lip_list: Optional[List[float]] = None,
    norm_each_list: Optional[List[str]] = None,
    save_fig: bool = False,
) -> None:
    """
    Plot the results for different Lipschitz constants.

    Args:
        results_filename (str): Path to the results file.
        save_folder (str): Path to the folder where the plots will be saved.
        db_list (List[str], optional): List of database names to consider. Defaults to None.
        network_list (List[str], optional): List of network types to consider. Defaults to None.
        hidden_dim_list (List[int], optional): List of hidden dimensions to consider. Defaults to None.
        attacks_list (List[str], optional): List of attack types to consider. Defaults to None.
        epsilon_list (List[float], optional): List of epsilon values to consider. Defaults to None.
        lip_list (List[float], optional): List of LIP values to consider. Defaults to None.
        norm_each_list (List[str], optional): List of norm each node values to consider. Defaults to None.
        save_fig (bool, optional): Flag indicating whether to save the figures. Defaults to False.
    """ 
                        
    lip_values   = []
    adv_values   = []
    sigma_values = []
    if (results_filename != None):
        df = pd.read_csv(results_filename, delimiter=",")
        num_splits = len(df['split_num'].unique())
        dataset_list = df['db_name'].unique()
        hd_dim_list  = df['hidden_dim'].unique()
        hd_dim_list  = sorted(hd_dim_list, reverse=True)
        netw_list    = df['network_type'].unique()
        att_list     = df['attack'].unique()
        eps_list     = df['epsilon'].unique()
        norm_list    = df['norm_each_node'].unique()

        # Average results for the number of splits
        averaged_df = df.groupby(np.arange(len(df))//num_splits).mean()
        averaged_df.drop(['split_num'], axis=1, inplace=True)
        averaged_df.insert(0, "hidden_dim", list(df.hidden_dim[::num_splits]), True)
        averaged_df.insert(0, "attack", list(df.attack[::num_splits]), True)
        averaged_df.insert(0, "network_type", list(df.network_type[::num_splits]), True)
        averaged_df.insert(0, "db_name", list(df.db_name[::num_splits]), True)
        all_db_perf = get_result_dict(averaged_df)  

        lip_values  = [col.split('_')[2] for col in averaged_df.columns if 'attack_success' in col] 
        lip_values = [i if ((i == 'no') or (i == 'NO')) else str((float(i))) for i in lip_values]                

    if (db_list):
        dataset_list = list(set(dataset_list) & set(db_list))
    if (network_list):
        netw_list =  list(set(network_list) & set(netw_list))
    if (hidden_dim_list):
        hd_dim_list = list(set(hd_dim_list).intersection(set([str(i) for i in hidden_dim_list])))
    if (attacks_list):
        att_list = list(set(att_list) & set(attacks_list))
    if (epsilon_list):
        eps_list = list(set(eps_list) & set(epsilon_list))
    if (norm_each_list):
        norm_list =  list(set(norm_each_list) & set(norm_list))
        
    eps_list    = [0] + sorted(eps_list, reverse=False)
    dict_values = lip_values
   
    for db_name in dataset_list:  
        for attack in att_list:
            for norm_each_node in norm_list:
                for hd in hd_dim_list:
                    for network_type in netw_list:
                        print_str = f'{db_name} {attack.upper()} attack\n {network_type.upper()} network'
                        perf_dict = {}
                        j=0
                        perf_dict = all_db_perf[db_name][attack][norm_each_node][network_type][str(hd)]

                        # plot only for some values of epsilon
                        perf_dict     = {k: perf_dict.get(k, None) for k in eps_list}
                        perf_dict     = {k: v for k, v in perf_dict.items() if v}

                        after_attack_acc = [perf_dict[key]['after_attack_acc'] for key in perf_dict.keys()]
                        original_acc     = perf_dict[(eps_list[1])]['original_acc']
                        try:
                            original_acc_val = perf_dict[(eps_list[1])]['orig_acc_val']
                        except:
                            pass
                        vals = lip_values
                        dict_values[0] = str(perf_dict[(eps_list[1])]['upper_lip'])
                        lip_values[0]  = str(perf_dict[(eps_list[1])]['upper_lip'])

                        original_acc = original_acc[:, np.newaxis]
                        after_attack_acc = np.stack(after_attack_acc, axis=1)
                        after_attack_acc = np.hstack((original_acc, after_attack_acc))

                        xi = list(range(len(eps_list)))
                        acc_dict = dict(zip(dict_values, after_attack_acc))

                        if (lip_list):
                            lip_list = [str(i) for i in lip_list]
                            lip_list = list(set(lip_values) & set(lip_list))
                            values_plot = [lip_values[0]] + [str(x) for x in sorted(lip_list, reverse=True)]
                        else:
                            values_plot = lip_values
                        try:
                            values_plot = [i for i in values_plot if float(i) <= float(values_plot[0])]
                        except:
                            pass

                        legend = []
                        for i in range(len(values_plot)):
                            l_val = values_plot[i]
                            val = acc_dict[l_val]
                            if (sum(val) == 0):
                                continue
                            if (i==0):
                                legend.append(r'$\theta$' + f'={values_plot[i]}')
                                line, = plt.plot(eps_list, val)
                            else:
                                legend.append(r'$\overline{\vartheta}$' + f'={values_plot[i]}')
                                try:
                                    line, = plt.plot(eps_list, val)
                                except:
                                    pass
                        try:
                            plt.margins(x=0)
                            plt.legend(legend, loc='upper right')
                            plt.xticks(xi, eps_list)
                            plt.grid()
                            plt.xlabel('ε')
                            plt.ylabel('After attack accuracy')
                            if (eps_list[-1] > 2):
                                plt.xlim([0, 600])
                            else:
                                plt.xlim([0, 0.8])
                            plt.xscale('linear') 
                            if save_fig:
                                len_hd = len(hd.split(','))
                                plt.savefig(f'{save_folder}/{db_name.replace(":", "_")}_{attack}_{network_type.upper()}_{len_hd}_lipschitz.pdf', bbox_inches="tight")
                                plt.close()  
                            else:
                                plt.title(print_str)
                                plt.show()
                        except:
                            pass

                                        
def plot_history(perf_metrics, lip_constr, perf_measure = 'accuracy', fig_name=False, skip_epochs=0):
    """Input: dictionaries with model performance: dict_keys(['loss_train', 'loss_val', 'acc_train', 'acc_val'])"""
    
    acc_list_train, loss_list_train, f1_list_train = get_loss_acc_f1(perf_metrics['train'])
    acc_list_val,   loss_list_val,   f1_list_val   = get_loss_acc_f1(perf_metrics['val'])
    
    if ('f1' in perf_measure.lower()):
        acc_list_train        = f1_list_train
        acc_list_val          = f1_list_val
    
    # summarize history for accuracy    
    plt.plot(acc_list_train[skip_epochs:])
    plt.plot(acc_list_val[skip_epochs:])   
    
    plt.ylabel(perf_measure.capitalize())
    plt.xlabel('Epoch')
    plt.legend(['Train ' + r'$\overline{\vartheta}$' + f'={lip_constr:0.2f}', 'Test  ' + r'$\overline{\vartheta}$' + f'={lip_constr:0.2f}'], loc='lower right')
    plt.grid()
    if fig_name:
        plt.savefig(f'{fig_name}_accuracy.png', bbox_inches="tight")
        plt.close()
    else:
        plt.title(f'Model {perf_measure.lower()}')
        plt.show()

    # summarize history for loss  
    plt.plot(loss_list_train[skip_epochs:])
    plt.plot(loss_list_val[skip_epochs:]) 
    
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train ' + r'$\overline{\vartheta}$' + f'={lip_constr:0.2f}', 'Test  ' + r'$\overline{\vartheta}$' + f'={lip_constr:0.2f}'], loc='upper right')
    plt.grid()
    if fig_name:
        plt.savefig(f'{fig_name}_loss.png', bbox_inches="tight")
        plt.close()
    else:
        plt.title('Model loss')
        plt.show()

        
