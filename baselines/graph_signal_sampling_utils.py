from typing import Dict


def get_hparams_dict(method_name: str):
    if method_name == 'ess':
        param_grid = {
            'k_ess': [1, 2, 3, 4, 5],
            'block_size_ess': [1, 2, 3],
            'bd_ess_mode': ['exact', 0.1, 0.2, 0.5, 1, 1.5, 2, 2.5, 5]
        }
    elif method_name == 'bsgda':
        param_grid = {
            'bsgda_mu': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0],
            'bsgda_epsilon': [1e-6, 1e-5, 1e-4],
            'bsgda_p_hops': [3, 6, 9, 12, 15, 21],
            'bsgda_reg_order': [1, 2, 3],
        }
    elif method_name == 'fastgsss':
        param_grid = {
            'fastgsss_bandwidth': ['exact', 0.1, 0.2, 0.5, 1, 1.5, 2, 2.5, 5],
            'fastgsss_nu': [25.0, 50.0, 75.0, 100.0, 150.0],  # Heat kernel width
            'fastgsss_order': [8, 10, 12, 15, 20],  # Chebyshev order for localization
            'fastgsss_recon_order': [1, 2, 3, 4, 5],  # T^k for reconstruction
        }
    elif method_name == 'rsbs':
        param_grid = {
            'rsbs_bandwidth_mode': ['exact', 0.75, 1.0, 1.25, 1.5],  # k relative to bandwidth
            'rsbs_num_estimation': [1, 3, 5],  # More = more stable λ_k estimation
            'rsbs_epsilon': [1e-3, 1e-2, 1e-1],  # Binary search precision for λ_k
            'rsbs_order': [20, 30, 40],  # Chebyshev order for λ_k estimation
            'rsbs_mu': [0.001, 0.01, 0.1, 0.5],  # Regularization in reconstruction
            'rsbs_reg_order': [1, 2],  # Laplacian order
            'rsbs_lap_type': ['comb', 'sym'],  # Laplacian type
        }
    else:
        raise ValueError(f"Unknown method name: {method_name}")

    return param_grid


def resolve_method_args(
        method_name: str,
        args: Dict,
        search_params: Dict
):
    if method_name == 'ess':
        params = {
            'k_ess': search_params['k_ess'],
            'block_size_ess': search_params['block_size_ess'],
        }
        if search_params['bd_ess_mode'] == 'exact':
            params['bd_ess'] = args['signal_bandwidth']
        else:
            k = max(2, int(search_params['bd_ess_mode'] * args['sample_size']))
            k = min(args['num_nodes'], k)
            params['bd_ess'] = k
    elif method_name == 'fastgsss':
        params = {
            'fastgsss_nu': search_params['fastgsss_nu'],
            'fastgsss_order': search_params['fastgsss_order'],
            'fastgsss_recon_order': search_params['fastgsss_recon_order'],
        }
        if search_params['fastgsss_bandwidth'] == 'exact':
            params['fastgsss_bandwidth'] = args['signal_bandwidth']
        else:
            k = max(2, int(search_params['fastgsss_bandwidth'] * args['sample_size']))
            k = min(args['num_nodes'], k)
            params['fastgsss_bandwidth'] = k
    elif method_name == 'rsbs':
        params = {
            'rsbs_num_estimation': search_params['rsbs_num_estimation'],
            'rsbs_epsilon': search_params['rsbs_epsilon'],
            'rsbs_order': search_params['rsbs_order'],
            'rsbs_mu': search_params['rsbs_mu'],
            'rsbs_reg_order': search_params['rsbs_reg_order'],
            'rsbs_lap_type': search_params['rsbs_lap_type'],
            'rsbs_num_rv': None,  # Use default (2*round(log N))
            'rsbs_lmin': None,  # Auto-compute
            'rsbs_lmax': None,  # Auto-compute
            'rsbs_return_list': True,
            'rsbs_verbose': False,
        }
        if search_params['rsbs_bandwidth_mode'] == 'exact':
            params['bandwidth'] = args['signal_bandwidth']
        else:
            k = max(2, int(search_params['rsbs_bandwidth_mode'] * args['sample_size']))
            k = min(args['num_nodes'], k)
            params['bandwidth'] = k
    else:
        raise ValueError(f"Unknown method name: {method_name}")
    return params
