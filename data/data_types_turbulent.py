from dataclasses import dataclass
from typing import List, Optional, Dict


@dataclass(frozen=True)
class GraphGenerationConfigTurbulent:
    graph_type: str
    grid_shape: List[int]
    num_edges: int
    num_nodes: int

@dataclass(frozen=True)
class SignalGenerationConfigTurbulent:
    num_fields: int
    field_names: List[str]
    signal_type: str
    field_to_index: Dict[str, int] = None
    index_to_field: Dict[str, str] = None

@dataclass(frozen=True)
class GraphWaveletTransformConfigTurbulent:
    num_filters: int
    num_scales: int
    lambda_min: float
    lambda_max: float
    chebyshev_max_degree: int
    K: int = None
    bandpass_lambda_min: float = None
    gwt_scales: List[float] = None


@dataclass(frozen=True)
class ReconstructorPretrainingConfigTurbulent:
    """Configuration for turbulent radiative layer 2D reconstruction pretraining."""
    base_data_dir: str
    train_data_dir: str
    validation_data_dir: str
    test_data_dir: str
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
    cache_mode: str  # 'full' or 'lru' (for turbulent, typically 'full' since graph is shared)
    lru_cache_size: int  # Max matrices in LRU cache (only used if cache_mode='lru')
    num_workers: int  # DataLoader workers (0 recommended when data is preloaded)
    preload_to_memory: bool  # Whether to preload dataset to RAM
    # Turbulent-specific config
    field_index: Optional[int]  # None = all 4 fields, 0-3 = specific field (density, pressure, velocity_x, velocity_y)
    field_name: Optional[str]  # For logging: "density", "pressure", "velocity_x", "velocity_y", or "all_fields"
    igwt_mode: str  # "dense" or "chebyshev"
    # Normalization config
    use_normalization: bool = False  # Whether to use Z-score normalization
    # Exploration config for MLIS sampling
    exploration_rate: float = 0.0  # Exploration rate for mlis_with_exploration mode (0.0 = pure exploitation, 1.0 = pure exploration)
