import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import torch
import torchmetrics
import sklearn
import sklearn.model_selection 
import torch_geometric.transforms as T
import torch_geometric
from torch_geometric.datasets import GitHub, FacebookPagePage, DeezerEurope, LastFMAsia
from scipy.stats import norm, binom_test
from typing import Dict, Any, Tuple, List, Optional, Union


class Smooth(object):
    """A smoothed classifier."""

    # To abstain, Smooth returns this int
    ABSTAIN = -1
    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, sigma: float, device: torch.device):
        """
        Initializes the Smooth classifier.

        Args:
            base_classifier (torch.nn.Module): The base classifier model.
            num_classes (int): The number of classes in the classifier.
            sigma (float): The sigma value for noise sampling.
            device (torch.device): The device to perform computations on.
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma
        self.device = device

    def predict(self, dataset: torch_geometric.data.data.Data, n: int, alpha: float, return_abtained: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, float]]:
        """
        Predicts the labels using the smoothed classifier.

        Args:
            dataset: The input dataset.
            n (int): The number of noise samples to generate.
            alpha (float): The threshold value for abstaining.
            return_abtained (bool): Whether to return the percentage of abstained samples or not.

        Returns:
            torch.Tensor: The predicted labels.
            float: The percentage of abstained samples.
        """
        self.base_classifier.eval()
        counts = self._sample_noise(dataset, n)
        top2 = (-counts).argsort(axis=-1)[:, :2]
        count1 = [counts[i, top2[i, 0]] for i in range(len(counts))]
        count2 = [counts[i, top2[i, 1]] for i in range(len(counts))]
        res = []
        if return_abtained:
            for i in range(len(count1)):
                test = binom_test(count1[i], count1[i] + count2[i], p=0.5)
                if test > alpha:
                    res.append(Smooth.ABSTAIN)
                else:
                    res.append(top2[i, 0])
            print(f"abstained: {res.count(-1) / len(res) * 100} %")
            return top2[:, 0], res.count(-1) / len(res) * 100
        return top2[:, 0]
                              
    def _sample_noise(self, dataset: torch_geometric.data.data.Data, num: int) -> np.ndarray:
        """
        Samples noise and generates counts for each class.

        Args:
            dataset: The input dataset.
            num (int): The number of noise samples to generate.

        Returns:
            np.ndarray: The counts for each class.
        """
        mask = dataset.test_mask
        test_x = dataset.x[mask].clone().to(self.device)
        with torch.no_grad():
            counts = np.zeros([sum(mask), self.num_classes], dtype=int)
            for _ in range(num):
                add_sh = torch.randn_like(test_x, device=self.device) * self.sigma
                dataset.x[mask] = test_x + add_sh
                predictions = self.base_classifier(dataset.x, dataset.edge_index, dataset.edge_weight)[mask].argmax(1)
                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        """
        Counts the occurrences of each element in the input array.

        Args:
            arr (np.ndarray): The input array.
            length (int): The length of the output counts array.

        Returns:
            np.ndarray: The counts for each element.
        """
        counts = np.zeros([len(arr),length], dtype=int)
        for i in range(len(arr)):
            idx = arr[i]
            counts[i, idx] += 1
        return counts

def dump_object(filepath: str, obj: Any) -> None:
    """
    Serialize and dump an object to a file using pickle.

    Args:
        filepath (str): Filepath to save the object.
        obj (Any): Object to be dumped.

    Returns:
        None
    """
    with open(filepath, 'wb') as fn:
        pickle.dump(obj, fn)


def load_object(filepath: str) -> Any:
    """
    Load and deserialize an object from a file using pickle.

    Args:
        filepath (str): Filepath to load the object from.

    Returns:
        Any: Deserialized object
    """
    with open(filepath, 'rb') as fn:
        obj = pickle.load(fn)
    return obj


def calculate_metrics_torch(y_pred: torch.Tensor, y_true: torch.Tensor, num_classes: int = 2) -> Dict[str, Union[float, torch.Tensor]]:
    """
    Calculate various evaluation metrics.

    Args:
        y_pred (torch.Tensor): Predicted labels.
        y_true (torch.Tensor): True labels.
        num_classes (int, optional): Number of classes. Defaults to 2.

    Returns:
        dict: Dictionary containing calculated metrics.
    """
    acc_metrics = {
        'acc':         torchmetrics.functional.accuracy(y_pred, y_true),
        'f1_sc_micro': torchmetrics.functional.f1_score(y_pred, y_true, average='micro'),
        'prec_micro':  torchmetrics.functional.precision(y_pred, y_true, average='micro'),
        'rec_micro':   torchmetrics.functional.recall(y_pred, y_true, average='micro'),
        'f1_sc':       torchmetrics.functional.f1_score(y_pred, y_true, average=None, num_classes=num_classes),
        'prec':        torchmetrics.functional.precision(y_pred, y_true, average=None, num_classes=num_classes),
        'rec':         torchmetrics.functional.recall(y_pred, y_true, average=None, num_classes=num_classes),
    }
    return acc_metrics


def get_Lips_constant(model: torch.nn.Module) -> float:
    """
    Calculate the Lipschitz constant of a model.

    Args:
        model (torch.nn.Module): The model for which the Lipschitz constant is calculated.

    Returns:
        float: The Lipschitz constant.
    """
    w_list = []
    num_layers = len(model.gcn_stack)
    
    if model.network_type == 'sage':
        for i, gcn_block in enumerate(model.gcn_stack):
            w0 = torch.clone(gcn_block.lin_r.weight.detach().data)
            w1 = torch.clone(gcn_block.lin_l.weight.detach().data)
            w_list.append(w0 + w1)
    elif model.network_type == 'gcn':
        for i, gcn_block in enumerate(model.gcn_stack):
            w = torch.clone(gcn_block.lin.weight.detach().data)
            w_list.append(w)
            
    theta_bar_constraint = w_list[-1]
    for i in range(num_layers - 2, -1, -1):
        theta_bar_constraint @= w_list[i]  
            
    theta_bar_constraint = torch.linalg.norm(theta_bar_constraint, ord=2).item()
    
    return theta_bar_constraint


def get_Lips_constant_upper(model: torch.nn.Module) -> float:
    """
    Calculate the upper bound of the Lipschitz constant of a model.

    Args:
        model (torch.nn.Module): The model for which the Lipschitz constant upper bound is calculated.

    Returns:
        float: The upper bound of the Lipschitz constant.
    """
    theta_bar_no_constraint = 1
    
    if model.network_type == 'sage':
        for i, gcn_block in enumerate(model.gcn_stack):
            w0 = torch.clone(gcn_block.lin_r.weight.detach().data)
            w1 = torch.clone(gcn_block.lin_l.weight.detach().data)
            layer_sum = torch.linalg.norm(w0, ord=2).item() + torch.linalg.norm(w1, ord=2).item()
            theta_bar_no_constraint *= layer_sum
    elif model.network_type == 'gcn':
        for i, gcn_block in enumerate(model.gcn_stack):
            w = torch.clone(gcn_block.lin.weight.detach().data)
            layer_sum = torch.linalg.norm(w, ord=2).item()
            theta_bar_no_constraint *= layer_sum
    
    return theta_bar_no_constraint


def get_loss_acc_f1(perf_metrics: List[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract accuracy, loss, and F1 score from performance metrics.

    Args:
        perf_metrics (List[dict]): List of performance metrics dictionaries.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing accuracy, loss, and F1 score arrays.
    """
    acc_list = np.zeros(len(perf_metrics))
    loss_list = np.zeros(len(perf_metrics))
    f1_list = np.zeros(len(perf_metrics))
    
    for i in range(len(perf_metrics)):
        acc_list[i] = perf_metrics[i]['acc']
        loss_list[i] = perf_metrics[i]['loss']
        f1_list[i] = perf_metrics[i]['f1_sc_micro']
    
    return acc_list, loss_list, f1_list


def update_parameters(
    conf_param: dict,
    with_constraint: Optional[bool] = None,
    constraint_type: Optional[str] = None,
    continue_training: Optional[bool] = None,
    rho: Optional[float] = None,
    old_rho: Optional[float] = None,
    num_epochs: Optional[int] = None,
    early_stop: Optional[bool] = None,
    patience: Optional[int] = None,
    print_results: Optional[bool] = None,
    print_every_k: Optional[int] = None,
    learning_rate: Optional[float] = None,
    lr_scheduler: Optional[str] = None,
    hidden_dim: Optional[int] = None,
    num_layers: Optional[int] = None,
    weight_decay: Optional[float] = None,
    plot: Optional[bool] = None,
) -> dict:
    """
    Update the parameters in a configuration dictionary.

    Args:
        conf_param (dict): The configuration dictionary to be updated.
        with_constraint (bool, optional): Whether to apply the constraint. Defaults to None.
        constraint_type (str, optional): Type of constraint. Defaults to None.
        continue_training (bool, optional): Whether to continue training. Defaults to None.
        rho (float, optional): Lipschitz constraint value. Defaults to None.
        old_rho (float, optional): Old constraint value. Defaults to None.
        num_epochs (int, optional): Number of epochs. Defaults to None.
        early_stop (bool, optional): Whether to use early stopping. Defaults to None.
        patience (int, optional): Patience value for early stopping. Defaults to None.
        print_every_k (int, optional): Print performance every k epochs. Defaults to None.
        learning_rate (float, optional): Learning rate value. Defaults to None.
        lr_scheduler (str, optional): Learning rate scheduler. Defaults to None.
        hidden_dim (int, optional): Hidden dimension. Defaults to None.
        num_layers (int, optional): Number of layers. Defaults to None.
        weight_decay (float, optional): Weight decay value. Defaults to None.
        plot (bool, optional): Whether to plot. Defaults to None.
        print_results (bool, optional): Whether to print results. Defaults to None.

    Returns:
        dict: The updated configuration dictionary.
    """
    parameters = locals()
    for param in parameters:
        if parameters[param] is not None:
            if param in conf_param.keys():
                conf_param[param] = parameters[param]
    
    return conf_param


def load_dataset(db_name: str, dataset_params: Dict[str, Any], seed: int) -> Tuple:
    """
    Loads the dataset and creates train, validation, and test splits.

    Args:
        db_name (str): The name of the dataset.
        dataset_params (dict): Parameters for dataset configuration.
        seed (int): The random seed value.

    Returns:
        dataset: The loaded dataset.
        num_classes (int): The number of classes in the dataset.
        num_features (int): The number of features in the dataset.
    """
    np.random.seed(seed) 
    
    print_info = dataset_params['print_info']
    test_size = dataset_params['test_size']
    val_size = dataset_params['val_size']
    
    if print_info:
        print(f"Dataset: {db_name}")

    folder_name = db_name

    if ':' in db_name:
        db_name, subdb_name = db_name.split(':')
    if db_name in ['GitHub', 'FacebookPagePage', 'DeezerEurope', 'LastFMAsia']:
        db = eval(db_name)(root=f'/tmp/{folder_name}')
    dataset = db[0]

    # Create masks
    dataset_len = len(dataset.x)
    targets = dataset.y.numpy()
    num_nodes = dataset.x.shape[0]
    train_indices, test_indices = sklearn.model_selection.train_test_split(
        np.arange(num_nodes), test_size=test_size, random_state=seed, stratify=targets)
    y_train = targets[train_indices]
    train_indices, val_indices = sklearn.model_selection.train_test_split(
        train_indices, test_size=val_size/(1-test_size), random_state=seed, stratify=y_train)

    dataset_len = len(dataset.x)
    train_mask = np.array([False] * dataset_len)
    val_mask = np.array([False] * dataset_len)
    test_mask = np.array([False] * dataset_len)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    dataset.train_mask = torch.tensor(train_mask)
    dataset.val_mask = torch.tensor(val_mask)
    dataset.test_mask = torch.tensor(test_mask)

    num_classes = len(dataset.y[dataset.test_mask].unique())
    num_features = dataset.x.shape[1]
    num_edges = dataset.edge_index.shape[1]

    classes, counts = dataset.y.unique(return_counts=True)

    if print_info:
        print(f"Num. classes:  {num_classes}")
        print(f"Num. edges:    {num_edges}")
        print(f"Num. features: {num_features}")
        print(f"Num. nodes:    {num_nodes}")
        print(f"Classes:       {classes.cpu().numpy()}")
        print(f"Counts :       {counts.cpu().numpy()}")
        print(f"\ntrain num:   {sum(dataset.train_mask)}")
        print(f"val num:       {sum(dataset.val_mask)}")
        print(f"test num:      {sum(dataset.test_mask)}")

    if dataset_params['network_type'] == 'gcn':
        dataset.edge_index, dataset.edge_weight = torch_geometric.utils.add_self_loops(
            dataset.edge_index, dataset.edge_weight)
    else:
        dataset.edge_index, dataset.edge_weight = torch_geometric.utils.remove_self_loops(
            dataset.edge_index, dataset.edge_weight)
    return dataset, num_classes, num_features


def get_result_dict(df: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, float]]]]]]:
    """
    Extracts performance metrics from a DataFrame and organizes them into a nested dictionary.

    Args:
        df (pd.DataFrame): DataFrame containing the performance metrics.

    Returns:
        all_db_perf (Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, float]]]]]]):
            Nested dictionary containing the extracted performance metrics.
            Format: {db_name: {attack: {norm_each_node: {network_type: {hidden_dim: {epsilon: {perf_metric: value}}}}}}}
    """
    dataset_list = df['db_name'].unique()
    hidden_dim_list = df['hidden_dim'].unique()
    netw_list = df['network_type'].unique()
    att_list = df['attack'].unique()
    eps_list = df['epsilon'].unique()
    eps_list = sorted(eps_list, reverse=True)
    norm_list = df['norm_each_node'].unique()

    result_dict = {}
    for db_name in dataset_list:
        result_dict[db_name] = df.loc[df['db_name'] == db_name]

    all_db_perf = {}
    for db_name in dataset_list:
        db1 = result_dict[db_name]
        one_db_perf_dict = {}

        for attack in att_list:
            db_att = db1.loc[(db1['attack'] == attack)]
            perf_dict_norm = {}
            for norm_each_node in norm_list:
                db_norm = db_att.loc[(db_att['norm_each_node'] == norm_each_node)]
                perf_dict_network = {}
                for network_type in netw_list:
                    db_aux = db_norm.loc[(db_norm['network_type'] == network_type)]
                    perf_dict_hd = {}
                    for hd in hidden_dim_list:
                        db_hd = db_aux.loc[db_aux['hidden_dim'] == hd]
                        perf_dict_eps = {}
                        for epsilon in eps_list:
                            db_eps = db_hd.loc[(db_hd['epsilon'] == epsilon)]
                            perf_dict = {}
                            for perf_metrics in ['attack_success', 'original_acc', 'orig_acc_val', 'after_attack_acc', 'l2', 'logit_diff', 'nll_adv']:
                                pm_col = [col for col in db_eps.columns if perf_metrics in col]
                                pm_values = db_eps[pm_col].head(1)
                                try:
                                    perf_dict[perf_metrics] = pm_values.values[0]
                                    perf_dict['upper_lip'] = round(db_eps['upper_Lip_ct'].item(), 2)
                                except:
                                    pass
                            perf_dict_eps[epsilon] = perf_dict
                        perf_dict_hd[hd] = perf_dict_eps
                    perf_dict_network[network_type] = perf_dict_hd
                perf_dict_norm[norm_each_node] = perf_dict_network
            one_db_perf_dict[attack] = perf_dict_norm
        all_db_perf[db_name] = one_db_perf_dict

    return all_db_perf


def summarize_results(results_filename: str, summarize_filename: Optional[str] = None) -> None:
    """
    Summarizes the results from a CSV file and saves the summary to an Excel file.

    Args:
        results_filename: The filename of the results CSV file.
        summarize_filename: The filename for the summarized Excel file. Defaults to None.
    """

    # Set the default summarize filename if not provided
    if summarize_filename is None:
        summarize_filename = 'summarize_' + results_filename.split('.')[0] + '.xlsx'

    # Read the CSV file into a DataFrame
    df = pd.read_csv(results_filename, delimiter=",")

    # Calculate the mean and standard deviation DataFrames
    num_splits = len(df['split_num'].unique())
    mean_df = df.groupby(np.arange(len(df)) // num_splits).mean().round(3)
    std_df = df.groupby(np.arange(len(df)) // num_splits).std().round(3)

    # Modify the mean DataFrame
    mean_df.drop(['split_num'], axis=1, inplace=True)
    mean_df.insert(0, "hidden_dim", list(df.hidden_dim[::num_splits]), True)
    mean_df.insert(0, "attack", list(df.attack[::num_splits]), True)
    mean_df.insert(0, "network_type", list(df.network_type[::num_splits]), True)
    mean_df.insert(0, "db_name", list(df.db_name[::num_splits]), True)

    # Modify the std DataFrame
    std_df.drop(['split_num'], axis=1, inplace=True)
    try:
        std_df.adv_epsilon_train = mean_df.adv_epsilon_train
    except:
        pass
    std_df.epsilon = mean_df.epsilon
    std_df.insert(0, "hidden_dim", list(df.hidden_dim[::num_splits]), True)
    std_df.insert(0, "attack", list(df.attack[::num_splits]), True)
    std_df.insert(0, "network_type", list(df.network_type[::num_splits]), True)
    std_df.insert(0, "db_name", list(df.db_name[::num_splits]), True)

    # Get the mean and std performance dictionaries
    mean_perf = get_result_dict(mean_df)
    std_perf = get_result_dict(std_df)

    # Create an Excel writer and workbook
    writer = pd.ExcelWriter(summarize_filename, engine='xlsxwriter')
    workbook = writer.book

    # Get lip values from DataFrame columns
    lip_values = [col.split('_')[-1] for col in df.columns if 'original_acc' in col]

    # Iterate over datasets, attacks, etc.
    for dataset in mean_perf.keys():
        for attack in mean_perf[dataset].keys():
            # Create a worksheet for each dataset and attack combination
            ws = f'{dataset.replace(":", "")} {attack}'
            i = 1
            worksheet = workbook.add_worksheet(ws)
            writer.sheets[ws] = worksheet
            worksheet.write_string(i, 7, f'DATASET: {dataset}.    ATTACK: {attack}')
            i += 2
            for norm_each_node in mean_perf[dataset][attack].keys():
                for network in mean_perf[dataset][attack][norm_each_node].keys():
                    for hidden_dim in mean_perf[dataset][attack][norm_each_node][network].keys():
                        worksheet.write_string(i, 1, f'NETWORK: {network.upper()}. HIDDEN DIM: {hidden_dim}.\n NORM EACH NODE: {bool(norm_each_node)}')
                        i+=2
                        # Get performance dictionaries for mean and std
                        perf_dict = mean_perf[dataset][attack][norm_each_node][network][hidden_dim]
                        perf_dict_std = std_perf[dataset][attack][norm_each_node][network][hidden_dim]

                        # Sort and filter epsilon values
                        eps_list = list(perf_dict.keys())
                        eps_list.sort()
                        perf_dict = {k: perf_dict.get(k, None) for k in eps_list}
                        perf_dict = {k: v for k, v in perf_dict.items() if v}
                        perf_dict_std = {k: perf_dict_std.get(k, None) for k in eps_list}
                        perf_dict_std = {k: v for k, v in perf_dict_std.items() if v}
                        eps_list = [0] + list(perf_dict.keys())
                        eps_list.sort()

                        # Calculate after attack accuracy matrices
                        after_attack_acc = [perf_dict[key]['after_attack_acc'] for key in perf_dict.keys()]
                        original_acc = perf_dict[float(eps_list[1])]['original_acc']
                        original_acc = original_acc[:, np.newaxis]
                        after_attack_acc = np.stack(after_attack_acc, axis=1)
                        after_attack_acc = np.hstack((original_acc, after_attack_acc))

                        after_attack_acc_std = [perf_dict_std[key]['after_attack_acc'] for key in perf_dict_std.keys()]
                        original_acc_std = perf_dict_std[float(eps_list[1])]['original_acc']
                        original_acc_std = original_acc_std[:, np.newaxis]
                        after_attack_acc_std = np.stack(after_attack_acc_std, axis=1)
                        after_attack_acc_std = np.hstack((original_acc_std, after_attack_acc_std))

                        try:
                            lip_values[0] = perf_dict[float(eps_list[1])]['upper_lip']
                        except:
                            pass
                        lip_values = [float(l) for l in lip_values]

                        after_attack_acc_mean_std = []
                        for c in zip(after_attack_acc, after_attack_acc_std):
                            after_attack_acc_mean_std.append(np.array([f'{e[0]:.3f}+/-{e[1]:.3f}' for e in zip(c[0], c[1])]))

                        acc_dict = dict(zip(lip_values, after_attack_acc_mean_std))

                        if 'adv_epsilon_train' in df.columns or lip_values[0] <= 0.5:
                            keep_lip_values = lip_values
                        else:
                            keep_lip_values = [i for i in lip_values if i <= lip_values[0]]

                        acc_table = pd.DataFrame.from_dict(acc_dict)[keep_lip_values]
                        acc_table.insert(0, "δ", eps_list, True)
                        acc_table = acc_table.T

                        # Write the accuracy table to the worksheet
                        acc_table.to_excel(writer, sheet_name=ws, startrow=i, startcol=0)
                        i += len(acc_table) + 5

    # Save the Excel file
    writer.save()
    print(f"Finished converting. See summary file {summarize_filename}")
