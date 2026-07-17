
from dataclasses import dataclass
from typing import Tuple, List


@dataclass(frozen=True)
class GraphGenerationConfig:
    graph_types: List[str]
    num_nodes_range: Tuple[int, int]
    num_graphs_per_type: int
    barabasi_albert_min_num_edges_per_new_node_range: Tuple[int, int]
    erdos_renyi_edge_prob_range: Tuple[float, float]
    sbm_num_blocks_range: Tuple[int, int]
    sbm_intra_cluster_prob_range: Tuple[float, float]
    sbm_inter_cluster_prob_range: Tuple[float, float]


@dataclass(frozen=True)
class SignalGenerationConfig:
    signal_types: List[str]
    R_range_max: float
    coeffs_mean: float
    coeffs_std_range: Tuple[float, float]
    noise_level_range: Tuple[float, float]


@dataclass(frozen=True)
class GraphWaveletTransformConfig:
    lambda_max: float
    K: int
    lambda_min: float
    num_scales: int
    num_filters: int
    scales: List[float]


@dataclass(frozen=True)
class ReconstructorPretrainingConfig:
    base_data_dir: str
    random_seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    batch_size: int
    generate_mask: bool
    keep_ratio: float
    mask_generation_mode: str  # "mlis", "weighted_random_per_scale", "random"
    sparsity_initial_value: float
    sparsity_final_value: float
    sparsity_scheduler_final_epoch: int
    sparsity_scheduler_strategy: str
    lr_scheduler_factor: float
    lr_scheduler_patience: int
    lr_scheduler_min_lr: float
    early_stopping_patience: int
    early_stopping_patience_min_delta: float
    model_name: str
    processor_num_layers: int
    output_dim: int
    hidden_dim: int
    activation_function: str
    encoder_type: str
    fusion_type: str
    decoder_type: str
    conv_layer_type: str
    pooling_type: str
    pooling_ratio: float
    predict_residual: bool
    enable_scale_embeddings: bool
    lr: float
    weight_decay: float
    gamma_coeff_consistency: float
    gamma_regularization: float
    epochs: int
    # Optimized data loading config
    cache_mode: str  # 'full' or 'lru'
    lru_cache_size: int  # Max matrices in LRU cache (only used if cache_mode='lru')
    num_workers: int  # DataLoader workers (0 recommended when data is preloaded)
    preload_to_memory: bool  # Whether to preload dataset to RAM


@dataclass(frozen=True)
class SparseGAEConfig:
    base_data_dir: str
    random_seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    batch_size: int
    keep_ratio: float
    sparsity_initial_value: float
    sparsity_final_value: float
    sparsity_scheduler_final_epoch: int
    sparsity_scheduler_strategy: str
    lr_scheduler_factor: float
    lr_scheduler_patience: int
    lr_scheduler_min_lr: float
    early_stopping_patience: int
    early_stopping_patience_min_delta: float
    model_name: str
    processor_num_layers: int
    output_dim: int
    latent_dim: int
    activation_function: str
    conv_layer_type: str
    lr: float
    weight_decay: float
    epochs: int
    variational: bool
    sparsity_mask_mode: str
    beta_kl: float