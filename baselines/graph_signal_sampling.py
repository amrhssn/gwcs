from typing import List, Tuple, Optional

import torch
from thgsp.graphs.core import SparseTensor, GraphBase
from thgsp.sampling import (
    ess, recon_ess,
    bsgda, recon_bsgda,
    fastgsss, recon_fastssss,
    rsbs, recon_rsbs
)

from baselines.graph_signal_sampling_utils import resolve_method_args


def build_graph_from_edge_index(
        edge_index: torch.Tensor,
        num_nodes: int,
        x: torch.Tensor
) -> Tuple[GraphBase, SparseTensor]:
    row, col = edge_index
    value = torch.ones(row.numel(), dtype=x.dtype, device=x.device)
    adj = SparseTensor(row=row, col=col, value=value,
                       sparse_sizes=(num_nodes, num_nodes))
    G = GraphBase(adj, dtype=x.dtype, device=x.device)
    return G, adj


def sampling_and_reconstruction(
        method_name: str,
        x: torch.Tensor,
        args,
        search_params
):
    if method_name == "ess":
        search_params = resolve_method_args(
            method_name=method_name,
            args=args,
            search_params=search_params
        )
        S, x_hat = ess_sampling_and_reconstruction(
            x=x,
            thgsp_L_sym=args['thgsp_L_sym'],
            thgsp_eigenvectors=args['thgsp_eigenvectors'],
            num_nodes=args['num_nodes'],
            sample_size=args['sample_size'],
            bd_ess=search_params['bd_ess'],
            k_ess=search_params['k_ess'],
            block_size_ess=search_params['block_size_ess']
        )
    elif method_name == 'bsgda':
        S, x_hat = bsgda_sampling_and_reconstruction(x=x,
                                                     thgsp_adj=args['thgsp_adj'],
                                                     thgsp_L_comb=args['thgsp_L_comb'],
                                                     num_nodes=args['num_nodes'],
                                                     sample_size=args['sample_size'],
                                                     bsgda_mu=search_params['bsgda_mu'],
                                                     bsgda_epsilon=search_params['bsgda_epsilon'],
                                                     bsgda_p_hops=search_params['bsgda_p_hops'],
                                                     bsgda_boost=True,
                                                     bsgda_reg_order=search_params['bsgda_reg_order'])
    elif method_name == 'fastgsss':
        search_params = resolve_method_args(
            method_name=method_name,
            args=args,
            search_params=search_params
        )
        S, x_hat = fastgsss_sampling_and_reconstruction(x=x,
                                                        thgsp_graph=args['thgsp_graph'],
                                                        num_nodes=args['num_nodes'],
                                                        sample_size=args['sample_size'],
                                                        fastgsss_bandwidth=search_params['fastgsss_bandwidth'],
                                                        fastgsss_nu=search_params['fastgsss_nu'],
                                                        fastgsss_cheby=True,
                                                        fastgsss_order=20,
                                                        fastgsss_recon_order=search_params['fastgsss_recon_order'],
                                                        fastgsss_sd=0.5,
                                                        fastgsss_verbose=False)
    elif method_name == 'rsbs':
        search_args = resolve_method_args(
            method_name=method_name,
            args=args,
            search_params=search_params
        )
        S, x_hat = rsbs_sampling_and_reconstruction(x=x,
                                                    thgsp_graph=args['thgsp_graph'],
                                                    thgsp_L_comb=args['thgsp_L_comb'],
                                                    num_nodes=args['num_nodes'],
                                                    sample_size=args['sample_size'],
                                                    bandwidth=search_args['bandwidth'],
                                                    rsbs_num_estimation=search_args['rsbs_num_estimation'],
                                                    rsbs_num_rv=search_args['rsbs_num_rv'],
                                                    rsbs_epsilon=search_args['rsbs_epsilon'],
                                                    rsbs_lmin=search_args['rsbs_lmin'],
                                                    rsbs_lmax=search_args['rsbs_lmax'],
                                                    rsbs_order=search_args['rsbs_order'],
                                                    rsbs_lap_type=search_args['rsbs_lap_type'],
                                                    rsbs_return_list=search_args['rsbs_return_list'],
                                                    rsbs_verbose=search_args['rsbs_verbose'],
                                                    rsbs_mu=search_args['rsbs_mu'],
                                                    rsbs_reg_order=search_args['rsbs_reg_order'])
    else:
        raise ValueError(f"Unknown method name: {method_name}")
    return S, x_hat


def ess_sampling_and_reconstruction(
        x: torch.Tensor,
        thgsp_L_sym: SparseTensor,
        thgsp_eigenvectors: torch.Tensor,
        num_nodes: int,
        sample_size: int,
        bd_ess: int,
        k_ess: int = 2,
        block_size_ess: int = 2
) -> Tuple[List[int], torch.Tensor]:
    S_ess = ess(operator=thgsp_L_sym, M=sample_size, k=k_ess, block_size=block_size_ess)  # list of node indices
    y_ess = x[S_ess]
    x_hat_ess = recon_ess(
        y=y_ess.squeeze(-1),
        S=S_ess,
        U=thgsp_eigenvectors,
        bd=bd_ess,
    )
    x_hat_ess = x_hat_ess.view(num_nodes, 1)  # (N, 1)
    return S_ess, x_hat_ess


def bsgda_sampling_and_reconstruction(
        x: torch.Tensor,
        thgsp_adj: SparseTensor,
        thgsp_L_comb: SparseTensor,
        num_nodes: int,
        sample_size: int,
        bsgda_mu: float = 0.01,
        bsgda_epsilon: float = 1e-5,
        bsgda_p_hops: int = 12,
        bsgda_boost: bool = True,
        bsgda_reg_order: int = 1,
):
    S_bsgda, T_lower = bsgda(
        spm=thgsp_adj,
        K=sample_size,
        mu=bsgda_mu,
        epsilon=bsgda_epsilon,
        p_hops=bsgda_p_hops,
        boost=bsgda_boost,
    )
    y_bsgda = x[S_bsgda]
    x_hat_bsgda = recon_bsgda(
        y=y_bsgda,         # (M, 1)
        S=S_bsgda,         # list or tensor of indices
        L=thgsp_L_comb,          # SparseTensor Laplacian
        mu=bsgda_mu,
        reg_order=bsgda_reg_order,
    )
    x_hat_bsgda = x_hat_bsgda.view(num_nodes, 1)
    return S_bsgda, x_hat_bsgda


def fastgsss_sampling_and_reconstruction(
        x: torch.Tensor,
        thgsp_graph: GraphBase,
        num_nodes: int,
        sample_size: int,
        fastgsss_bandwidth: int = 10,
        fastgsss_nu: float = 75.0,
        fastgsss_cheby: bool = True,
        fastgsss_order: int = 12,
        fastgsss_recon_order: int = 3,
        fastgsss_sd: float = 0.5,
        fastgsss_verbose: bool = False,
):
    S_fastgsss, T = fastgsss(
        G=thgsp_graph,
        M=sample_size,
        bandwidth=fastgsss_bandwidth,
        nu=fastgsss_nu,
        cheby=fastgsss_cheby,
        order=fastgsss_order,
        verbose=fastgsss_verbose,
    )
    y_fastgsss = x[S_fastgsss]
    x_hat_fastgsss = recon_fastssss(
        y=y_fastgsss,
        S=S_fastgsss,
        T=T,
        order=fastgsss_recon_order,
        sd=fastgsss_sd,
    )
    x_hat_fastgsss = x_hat_fastgsss.view(num_nodes, 1)
    return S_fastgsss, x_hat_fastgsss


def rsbs_sampling_and_reconstruction(
        x: torch.Tensor,
        thgsp_graph: GraphBase,
        thgsp_L_comb: SparseTensor,
        num_nodes: int,
        sample_size: int,
        bandwidth: int,
        rsbs_num_estimation: int = 1,
        rsbs_num_rv: Optional[int] = None,
        rsbs_epsilon: float = 1e-2,
        rsbs_lmin: Optional[int] = None,
        rsbs_lmax: Optional[int] = None,
        rsbs_order: int = 30,
        rsbs_lap_type: str = "comb",
        rsbs_return_list: bool = True,
        rsbs_verbose: bool = False,
        rsbs_mu: float = 0.01,
        rsbs_reg_order: int = 1,

):
    S_rsbs, cum_coh = rsbs(
        G=thgsp_graph,
        M=sample_size,
        k=bandwidth,
        num_estimation=rsbs_num_estimation,      # increase for more stable estimation
        num_rv=rsbs_num_rv,          # uses 2*round(log N) by default
        epsilon=rsbs_epsilon,
        lmin=rsbs_lmin,
        lmax=rsbs_lmax,
        order=rsbs_order,
        lap_type=rsbs_lap_type,
        return_list=rsbs_return_list,      # recon_rsbs expects S as list in tests
        verbose=rsbs_verbose,
    )

    y_rsbs = x[S_rsbs]  # (M, 1)
    x_hat_rsbs = recon_rsbs(
        y=y_rsbs,          # (M, 1)
        S=S_rsbs,
        L=thgsp_L_comb,
        cum_coh=cum_coh,
        mu=rsbs_mu,
        reg_order=rsbs_reg_order,
    )
    x_hat_rsbs = x_hat_rsbs.view(num_nodes, 1)
    return S_rsbs, x_hat_rsbs
