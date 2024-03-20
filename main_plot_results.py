import argparse
import os
from pathlib import Path

# user-defined modules
from plot_utils import plot_results_robustness, plot_results_attacks, plot_results_sn, plot_results_graph_defenses, plot_results_lipschitz

                        
datasets   = ('FacebookPagePage', 'GitHub', 'LastFMAsia', 'DeezerEurope', None)
networks   = ('gcn', 'sage', None)
attacks    = ('pgd_linf', 'pgd_l2_ce', 'pgd_l2_dl', 'pgd_l2_dlr', 'apgd_linf', 'apgd_l2_ce', 'apgd_l2_dl', 'apgd_l2_dlr', None)
norm_each  = (True, False, None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Plot results from file')
    
    # Add arguments
    parser.add_argument('--results_filename', '-f', type=str, default='results_attacks.csv', help='Name of csv file with attack results.')
    parser.add_argument('--results_filename_sn', '-fsn', type=str, default=None, help='Name of csv file with attack results for Spectral Normalization.')
    parser.add_argument('--results_filename_at', '-fa', type=str, default=None, help='Name of csv file with attack results for AT. If None, do not plot AT results.')
    parser.add_argument('--results_filename_rs', '-frs', type=str, default=None, help='Name of csv file with attack results for RS. If None, do not plot RS results.')
    parser.add_argument('--results_filename_svd', '-fsvd', type=str, default=None, help='Name of csv file with attack results for SVD-GCN. If None, do not plot results.')
    parser.add_argument('--results_filename_jaccard', '-fj', type=str, default=None, help='Name of csv file with attack results for GCN-Jaccard. If None, do not plot results.')
    parser.add_argument('--results_filename_rgcn', '-fr', type=str, default=None, help='Name of csv file with attack results for RGCN. If None, do not plot results.')    
    parser.add_argument('--results_filename_list', '-fl', type=str, nargs='+', default=None, help='Name of csv files with attack results for different attacks. If None, plot AT results')
    parser.add_argument('--dataset_list', '-dl', type=str, nargs='+', help='Dataset(s) to attack. If None, plot all.', choices=datasets)
    parser.add_argument('--hidden_dim_list', '-hdl', type=int, nargs='+', action='append', help='Hidden dimensions for the network (list of lists). If None, plot all.')
    parser.add_argument('--constr_list', '-cl', type=float, nargs='+', help='Lipschitz constraint(s). If None, plot all.') 
    parser.add_argument('--sn_list', '-snl', type=float, nargs='+', help='Upper Lipschitz constraint(s) for SN. If None, plot all.') 
    parser.add_argument('--adv_epsilon_list', '-ael', type=int, nargs='+', help='Perturbation value(s) used for adversarial training. If None, plot all') 
    parser.add_argument('--sigma_list', '-sl', type=float, nargs='+', help='Sigma for RS. If None, plot all') 
    parser.add_argument('--svd_list', '-svdl', type=int, nargs='+', help='Parameters for SVD-GCN. If None, plot all') 
    parser.add_argument('--jaccard_list', '-jl', type=float, nargs='+', help='Parameters for GCN-Jaccard. If None, plot all') 
    parser.add_argument('--rgcn_list', '-rl', type=float, nargs='+', help='Parameters for RGCN. If None, plot all') 
    parser.add_argument('--network_list', '-nl', type=str, nargs='+', help='Type of network(s). If None, plot all', choices=networks)
    parser.add_argument('--attack_list', '-al', type=str.lower, nargs='+', help='Attack(s) to use. If None, plot all', choices=attacks)
    parser.add_argument('--epsilon_list', '-el', type=int, nargs='+', help='Constraint(s) on the size of the perturbation. If None, plot all')
    parser.add_argument('--norm_each_list', '-ne', type=bool, nargs='+', help='Constraint the norm on each node instead of the whole graph. If None, plot all', choices=norm_each)   
    parser.add_argument('--save_plot', '-sp', action='store_true', help='Save plot or display it.')
    parser.add_argument('--saved_figures_path', '-sfp', type=str, default='./saved_figures', help='Name of folder to save figures.')
    args = parser.parse_args()
    
    if (args.save_plot):
        if not os.path.isdir(args.saved_figures_path):
            os.makedirs(args.saved_figures_path)
    if args.results_filename_sn:   
        # Compare rubustness between our method and Spectral Normalization
        assert Path(args.results_filename).exists(), f"Results file {args.results_filename} does not exist"
        assert Path(args.results_filename_sn).exists(), f"Spectral normalization results file {args.results_filename_sn} does not exist"
        plot_results_sn(
            args.results_filename,
            args.saved_figures_path,
            results_filename_sn=args.results_filename_sn,
            db_list=args.dataset_list,
            hidden_dim_list=args.hidden_dim_list,
            network_list=args.network_list,
            attacks_list=args.attack_list,
            epsilon_list=args.epsilon_list,
            norm_each_list=args.norm_each_list,
            lip_list=args.constr_list,
            sn_list=args.sn_list,
            save_fig=args.save_plot
        )
        
    elif args.results_filename_svd or args.results_filename_jaccard or args.results_filename_rgcn or args.results_filename_sn:
        # Compare rubustness between graph defenses
        assert Path(args.results_filename).exists(), f"Results file {args.results_filename} does not exist"
        if args.results_filename_svd:
            assert Path(args.results_filename_svd).exists(), f"SVD-GCN results file {args.results_filename_svd} does not exist"
        if args.results_filename_at:
            assert Path(args.results_filename_jaccard).exists(), f"GCN-Jaccard results file {args.results_filename_jaccard} does not exist"
        if args.results_filename_rgcn:
            assert Path(args.results_filename_rgcn).exists(), f"RGCN results file {args.results_filename_rgcn} does not exist"
        
        plot_results_graph_defenses(
            args.results_filename,
            args.saved_figures_path,
            results_filename_svd=args.results_filename_svd,
            results_filename_jaccard=args.results_filename_jaccard,
            results_filename_rgcn=args.results_filename_rgcn,
            db_list=args.dataset_list,
            hidden_dim_list=args.hidden_dim_list,
            network_list=args.network_list,
            attacks_list=args.attack_list,
            epsilon_list=args.epsilon_list,
            norm_each_list=args.norm_each_list,
            lip_list=args.constr_list,
            svd_list=args.svd_list,
            jaccard_list=args.jaccard_list,
            rgcn_list=args.rgcn_list,
            save_fig=args.save_plot
        )
        
    elif args.results_filename_list:
        # Compare rubustness between different attacks
        plot_results_attacks(
            args.results_filename_list,
            args.saved_figures_path,
            db_list=args.dataset_list,
            hidden_dim_list=args.hidden_dim_list,
            network_list=args.network_list,
            attacks_list=args.attack_list,
            epsilon_list=args.epsilon_list,
            norm_each_list=args.norm_each_list,
            lip_list=args.constr_list,
            save_fig=args.save_plot,
        )
    
    elif args.results_filename_at or args.results_filename_rs:
        # Compare rubustness between different methods (conventional models, Lipschitz constrained models, RM, AT)
        if args.results_filename:
            assert Path(args.results_filename).exists(), f"Results file {args.results_filename} does not exist"
        if args.results_filename_at:
            assert Path(args.results_filename_at).exists(), f"AT results file {args.results_filename_at} does not exist"
        if args.results_filename_rs:
            assert Path(args.results_filename_rs).exists(), f"RS results file {args.results_filename_rs} does not exist"

        plot_results_robustness(
            args.results_filename,
            args.saved_figures_path,
            results_filename_at=args.results_filename_at,
            results_filename_rs=args.results_filename_rs,
            db_list=args.dataset_list,
            hidden_dim_list=args.hidden_dim_list,
            network_list=args.network_list,
            attacks_list=args.attack_list,
            epsilon_list=args.epsilon_list,
            norm_each_list=args.norm_each_list,
            lip_list=args.constr_list,
            adv_epsilon_list=args.adv_epsilon_list,
            sigma_list=args.sigma_list,
            save_fig=args.save_plot
        )
        
    else:
        # Compare rubustness between models constrained with different Lipschitz constants
        plot_results_lipschitz(
            args.results_filename,
            args.saved_figures_path,
            db_list=args.dataset_list,
            hidden_dim_list=args.hidden_dim_list,
            network_list=args.network_list,
            attacks_list=args.attack_list,
            epsilon_list=args.epsilon_list,
            norm_each_list=args.norm_each_list,
            lip_list=args.constr_list,
            save_fig=args.save_plot
        )