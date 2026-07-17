from typing import Tuple, List, Optional, Dict

import torch
import torch.nn as nn
from torch_geometric.nn.pool import TopKPooling, SAGPooling

from models.utils_activation import get_activation
from models.utils_conv import get_gconv_layer
from graph_wavelet_transform.inverse import batch_igwt, batch_chebyshev_igwt

class SimpleGNN(nn.Module):
    def __init__(self):
        super().__init__()


class GNNNeuralIGWT(nn.Module):
    def __init__(
            self,
            num_filters: int,
            hidden_dim: int,
            output_dim: int,
            activation_function: str,
            encoder_type: str,
            decoder_type: str,
            conv_layer_type: str,
            processor_num_layers: int,
            predict_residual: bool = True,
            debug_mode: bool = False,
            enable_scale_embeddings: bool = True,
            igwt_mode: str = "dense",
            use_normalization: bool = False,
            normalization_stats: Optional[Dict] = None,
    ):
        super().__init__()
        # inputs
        self.num_filters = num_filters
        self.act = get_activation(activation_name=activation_function)
        self.predict_residual = predict_residual
        self.debug_mode = debug_mode
        self.enable_scale_embeddings = enable_scale_embeddings
        self.igwt_mode = igwt_mode
        assert igwt_mode in ["dense", "chebyshev"], \
            f"igwt_mode must be 'dense' or 'chebyshev', got {igwt_mode}"

        # Normalization support
        self.use_normalization = use_normalization
        if use_normalization:
            assert normalization_stats is not None, \
                "normalization_stats required when use_normalization=True"
            
            # ✅ FIX: Get field names dynamically from normalization_stats instead of hardcoding
            # OLD (WRONG): field_names = ['density', 'pressure', 'velocity_x', 'velocity_y']
            field_names = list(normalization_stats.keys())  # ✅ Use actual field names from stats
            
            means = torch.tensor([normalization_stats[name]['mean'] for name in field_names], dtype=torch.float32)
            stds = torch.tensor([normalization_stats[name]['std'] for name in field_names], dtype=torch.float32)
        
            # For single-field models (output_dim=1), extract the correct field's stats
            if output_dim == 1:
                self.register_buffer('norm_mean', means.view(1, -1))  # (1, num_fields)
                self.register_buffer('norm_std', stds.view(1, -1))
            else:
                # Multi-field model
                self.register_buffer('norm_mean', means.view(1, -1))  # (1, num_fields)
                self.register_buffer('norm_std', stds.view(1, -1))  # (1, num_fields)

        # Encoder
        # [Options]
        if self.enable_scale_embeddings:
            self.scale_embeddings = nn.Embedding(num_embeddings=num_filters, embedding_dim=hidden_dim)
        if encoder_type == 'linear':
            self.encoder_projection_1 = nn.Linear(in_features=2 * num_filters, out_features=hidden_dim)
            if self.enable_scale_embeddings:
                self.encoder_projection_2 = nn.Linear(in_features=2 * hidden_dim, out_features=hidden_dim)
        elif encoder_type == 'mlp':
            self.encoder_projection_1 = nn.Sequential(
                nn.Linear(in_features=2 * num_filters, out_features=hidden_dim),
                nn.LayerNorm(hidden_dim),
                get_activation(activation_name=activation_function, return_module=True),
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            )
            if self.enable_scale_embeddings:
                self.encoder_projection_2 = nn.Sequential(
                    nn.Linear(in_features=2 * hidden_dim, out_features=hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    get_activation(activation_name=activation_function, return_module=True),
                    nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        # Processor
        self.conv_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for _ in range(processor_num_layers):
            conv = get_gconv_layer(in_channels=hidden_dim,
                                   out_channels=hidden_dim,
                                   conv_layer_type=conv_layer_type)
            self.conv_layers.append(conv)
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # Decoder
        # [Options]
        if decoder_type == 'linear':
            self.decoder = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        elif decoder_type == 'mlp':
            self.decoder = nn.Sequential(
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                get_activation(activation_name=activation_function, return_module=True),
                nn.Linear(in_features=hidden_dim, out_features=output_dim)
            )
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")

    def forward(
            self,
            coeffs: torch.Tensor,
            sparsification_mask: torch.Tensor,
            edge_index: torch.Tensor,
            gwt_matrices: List[torch.Tensor] = None,
            batch: Optional[torch.Tensor] = None,
            chebyshev_data: Optional[dict] = None,
            batch_ptr: Optional[torch.Tensor] = None,
    ):
        encoded_inputs, sparse_coeffs = self.encode(coeffs, sparsification_mask)
        processed_inputs = self.process(encoded_inputs, edge_index)
        x_hat = self.decode(processed_inputs, sparse_coeffs, gwt_matrices, chebyshev_data, batch_ptr)
        return x_hat

    def encode(
            self,
            coeffs: torch.Tensor,
            sparsification_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # applying the sparsification mask to coefficients to get sparse coefficients
        sparsification_mask = sparsification_mask.float()   # (num_nodes, num_filters)
        sparse_coeffs = coeffs * sparsification_mask        # (num_nodes, num_filters)

        # concatenating the sparse coefficients and the sparsification mask as the input of the first projection layer
        inputs = torch.cat([sparse_coeffs, sparsification_mask], dim=-1)  # (num_nodes, 2 * num_filters)
        # Applying the first linear projection layer
        encoded_inputs = self.encoder_projection_1(inputs)  # (num_nodes, D)
        encoded_inputs = self.act(encoded_inputs)

        if self.enable_scale_embeddings:
            # fetching the scale embeddings
            scale_embeddings = self.scale_embeddings.weight   # (num_filters, hidden_dim_0)

            # computing the node scale embeddings
            # as linear combinations of the scale embeddings weighted by sparse coefficients of the nodes
            node_scale_embeddings = sparse_coeffs @ scale_embeddings  # (num_nodes, hidden_dim_0)

            # concatenating the encoded inputs with the node scale embeddings: (num_nodes, 2 * hidden_dim_0)
            augmented_encoded_inputs = torch.cat([encoded_inputs, node_scale_embeddings], dim=-1)

            # applying the second linear projection layer
            encoded_inputs = self.encoder_projection_2(augmented_encoded_inputs)  # (num_nodes, hidden_dim_0)
            encoded_inputs = self.act(encoded_inputs)
        return encoded_inputs, sparse_coeffs

    def process(
            self,
            encoded_inputs: torch.Tensor,
            edge_index: torch.Tensor
    ) -> torch.Tensor:
        processed_inputs = encoded_inputs
        for conv, norm in zip(self.conv_layers, self.layer_norms):
            normalized = norm(processed_inputs)
            processed_inputs = processed_inputs + conv(self.act(normalized), edge_index)
        return processed_inputs

    def decode(
            self,
            processed_inputs: torch.Tensor,
            sparse_coeffs: torch.Tensor,
            gwt_matrices: List[torch.Tensor] = None,
            chebyshev_data: dict = None,
            batch_ptr: torch.Tensor = None,
    ) -> torch.Tensor:
        # applying the decoder to reconstruct the graph signal, either by predicting the residual or
        # directly predicting the signal
        if self.predict_residual:
            with torch.no_grad():
                if self.igwt_mode == "dense":
                    x_hat_igwt = batch_igwt(coeffs=sparse_coeffs, Gs=gwt_matrices)
                elif self.igwt_mode == "chebyshev":
                    x_hat_igwt = batch_chebyshev_igwt(
                        coeffs=sparse_coeffs,
                        chebyshev_data=chebyshev_data,
                        batch_ptr=batch_ptr
                    )

                # Normalize IGWT output if normalization is enabled
                # Normalize IGWT output if normalization is enabled
                if self.use_normalization:
                    if x_hat_igwt.shape[-1] == 1:
                        # Single field - use first field's stats
                        x_hat_igwt = (x_hat_igwt - self.norm_mean[:, 0:1]) / self.norm_std[:, 0:1]
                    else:
                        # Multi-field - use all fields' stats
                        num_output_fields = x_hat_igwt.shape[-1]
                        x_hat_igwt = (x_hat_igwt - self.norm_mean[:, :num_output_fields]) / self.norm_std[:, :num_output_fields]

            residual = self.decoder(processed_inputs)
            x_hat = x_hat_igwt + residual
        else:
            x_hat = self.decoder(processed_inputs)
        return x_hat


class GUNetNeuralIGWT(nn.Module):
    def __init__(
            self,
            num_filters: int,
            hidden_dim: int,
            output_dim: int,
            activation_function: str,
            encoder_type: str,
            fusion_type: str,
            decoder_type: str,
            conv_layer_type: str,
            processor_num_layers: int,
            pooling_type: str,
            pooling_ratio: float,
            predict_residual: bool = True,
            debug_mode: bool = False,
            enable_scale_embeddings: bool = True,
            igwt_mode: str = "dense",
            use_normalization: bool = False,
            normalization_stats: Optional[Dict] = None,
    ):
        super().__init__()
        self.debug_mode = debug_mode
        self.num_filters = num_filters
        self.act = get_activation(activation_name=activation_function, return_module=False)
        self.processor_num_layers = processor_num_layers
        # pooling types: "top_k_pooling", "sag_pooling", "dense_diff_pooling", "dense_mincut_pooling"
        self.pooling_type = pooling_type
        self.pooling_ratio = pooling_ratio
        self.predict_residual = predict_residual
        self.enable_scale_embeddings = enable_scale_embeddings
        self.igwt_mode = igwt_mode
        assert igwt_mode in ["dense", "chebyshev"], \
            f"igwt_mode must be 'dense' or 'chebyshev', got {igwt_mode}"

        # Normalization support
        # Normalization support
        self.use_normalization = use_normalization
        if use_normalization:
            assert normalization_stats is not None, \
                "normalization_stats required when use_normalization=True"
            
            # ✅ FIX: Get field names dynamically from normalization_stats instead of hardcoding
            field_names = list(normalization_stats.keys())  # ✅ Use actual field names from stats
            
            means = torch.tensor([normalization_stats[name]['mean'] for name in field_names], dtype=torch.float32)
            stds = torch.tensor([normalization_stats[name]['std'] for name in field_names], dtype=torch.float32)
        
            # For single-field models (output_dim=1), extract the correct field's stats
            if output_dim == 1:
                self.register_buffer('norm_mean', means.view(1, -1))
                self.register_buffer('norm_std', stds.view(1, -1))
            else:
                # Multi-field model (output_dim=num_fields)
                self.register_buffer('norm_mean', means.view(1, -1))  # (1, num_fields)
                self.register_buffer('norm_std', stds.view(1, -1))  # (1, num_fields)

        # Encoder
        if self.enable_scale_embeddings:
            self.scale_embeddings = nn.Embedding(num_embeddings=num_filters, embedding_dim=hidden_dim)
        if encoder_type == 'linear':
            self.encoder_projection_1 = nn.Linear(in_features=2 * num_filters, out_features=hidden_dim)
            if self.enable_scale_embeddings:
                self.encoder_projection_2 = nn.Linear(in_features=2 * hidden_dim, out_features=hidden_dim)
        elif encoder_type == 'mlp':
            self.encoder_projection_1 = nn.Sequential(
                nn.Linear(in_features=2 * num_filters, out_features=hidden_dim),
                # nn.LayerNorm(hidden_dim),  # Add normalization
                get_activation(activation_name=activation_function, return_module=True),
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            )
            if self.enable_scale_embeddings:
                self.encoder_projection_2 = nn.Sequential(
                    nn.Linear(in_features=2 * hidden_dim, out_features=hidden_dim),
                    # nn.LayerNorm(hidden_dim),  # Add normalization
                    get_activation(activation_name=activation_function, return_module=True),
                    nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        # Processor: Graph U-Net architecture
        # Pooling layers for each level (k=1 to num_filters-1)
        self.use_sparse_pooling = pooling_type in ["top_k_pooling", "sag_pooling"]
        self.processor_pooling_layers = nn.ModuleList()
        if self.use_sparse_pooling:
            # Sparse pooling methods (TopKPooling, SAGPooling)
            for _ in range(num_filters):
                if pooling_type == "top_k_pooling":
                    pool = TopKPooling(in_channels=hidden_dim, ratio=pooling_ratio)
                elif pooling_type == "sag_pooling":
                    pool = SAGPooling(in_channels=hidden_dim, ratio=pooling_ratio)
                else:
                    raise ValueError(f"Unsupported pooling type: {pooling_type}")
                self.processor_pooling_layers.append(pool)
        else:
            if pooling_type == "dense_diff_pooling":
                # DiffPool: use GNN for assignment computation
                # S^(l) = softmax(GNN_pool(A^(l), X^(l)))
                raise NotImplementedError
            elif pooling_type == "dense_mincut_pooling":
                raise NotImplementedError
            else:
                raise ValueError(f"Unsupported pooling type: {pooling_type}")
            pass

        # Message-passing layers specific to each level but shared between down and up
        self.processor_message_passing_layers = nn.ModuleList()
        self.processor_layer_norms = nn.ModuleList()
        for level in range(num_filters):
            level_convs = nn.ModuleList()
            level_norms = nn.ModuleList()
            for _ in range(processor_num_layers):
                conv = get_gconv_layer(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    conv_layer_type=conv_layer_type
                )
                level_convs.append(conv)
                level_norms.append(nn.LayerNorm(hidden_dim))
            self.processor_message_passing_layers.append(level_convs)
            self.processor_layer_norms.append(level_norms)

        # Fusion layers for skip connections in up path
        self.processor_fusion_layers = nn.ModuleList()
        for _ in range(num_filters - 1):
            # Fuse unpooled features with skip connection
            if fusion_type == 'linear':
                fusion = nn.Linear(in_features=2 * hidden_dim, out_features=hidden_dim)
            elif fusion_type == 'mlp':
                fusion = nn.Sequential(
                    nn.Linear(in_features=2 * hidden_dim, out_features=hidden_dim),
                    # nn.LayerNorm(hidden_dim),  # Add normalization
                    get_activation(activation_name=activation_function, return_module=True),
                    nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                )
            else:
                raise ValueError(f"Unknown fusion type: {fusion_type}")
            self.processor_fusion_layers.append(fusion)

        # Decoder
        # [Options]
        if decoder_type == 'linear':
            self.decoder = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        elif decoder_type == 'mlp':
            self.decoder = nn.Sequential(
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                # nn.LayerNorm(hidden_dim),  # Add normalization
                get_activation(activation_name=activation_function, return_module=True),
                nn.Linear(in_features=hidden_dim, out_features=output_dim)
            )
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")

    def forward(
            self,
            coeffs: torch.Tensor,
            sparsification_mask: torch.Tensor,
            edge_index: torch.Tensor,
            gwt_matrices: List[torch.Tensor] = None,
            batch: Optional[torch.Tensor] = None,
            chebyshev_data: Optional[dict] = None,
            batch_ptr: Optional[torch.Tensor] = None,
    ):
        encoded_inputs, sparse_coeffs = self.encode(coeffs, sparsification_mask)
        processed_inputs = self.process(encoded_inputs, edge_index, batch)
        x_hat = self.decode(processed_inputs, sparse_coeffs, gwt_matrices, chebyshev_data, batch_ptr)
        return x_hat

    def encode(
            self,
            coeffs: torch.Tensor,
            sparsification_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # applying the sparsification mask to coefficients to get sparse coefficients
        sparsification_mask = sparsification_mask.float()  # (num_nodes, num_filters)
        sparse_coeffs = coeffs * sparsification_mask  # (num_nodes, num_filters)

        # NaN CHECK ENCODE-1: Input coefficients and mask
        if not torch.isfinite(coeffs).all():
            print(f"\n🔥 NaN in encode() input coeffs")
            print(f"   coeffs stats: min={coeffs.min().item():.6f}, max={coeffs.max().item():.6f}")
        if not torch.isfinite(sparsification_mask).all():
            print(f"\n🔥 NaN in encode() input sparsification_mask")
        if not torch.isfinite(sparse_coeffs).all():
            print(f"\n🔥 NaN in sparse_coeffs after masking")

        if self.debug_mode:
            # DEBUG: Check input magnitudes
            print(f"[DEBUG ENCODE] sparse_coeffs: min={sparse_coeffs.min().item():.4f}, "
                  f"max={sparse_coeffs.max().item():.4f}, "
                  f"mean={sparse_coeffs.mean().item():.4f}")

        # concatenating the sparse coefficients and the sparsification mask as the input of the first projection layer
        inputs = torch.cat([sparse_coeffs, sparsification_mask], dim=-1)  # (num_nodes, 2 * num_filters)

        # NaN CHECK ENCODE-2: After concatenation
        if not torch.isfinite(inputs).all():
            print(f"\n🔥 NaN in inputs after torch.cat")

        # Applying the first linear projection layer
        encoded_inputs = self.encoder_projection_1(inputs)  # (num_nodes, D)

        # NaN CHECK ENCODE-3: After first projection (before activation)
        if not torch.isfinite(encoded_inputs).all():
            print(f"\n🔥 NaN after encoder_projection_1 (before activation)")
            print(
                f"   encoded_inputs stats: min={encoded_inputs.min().item():.6f}, max={encoded_inputs.max().item():.6f}")
            # Check if weights/biases have NaN
            for name, param in self.encoder_projection_1.named_parameters():
                if not torch.isfinite(param).all():
                    print(f"   🔥 NaN in encoder_projection_1.{name}")

        encoded_inputs = self.act(encoded_inputs)

        # NaN CHECK ENCODE-4: After first activation
        if not torch.isfinite(encoded_inputs).all():
            print(f"\n🔥 NaN after first activation (after encoder_projection_1)")

        if self.debug_mode:
            print(f"[DEBUG ENCODE] after projection_1: min={encoded_inputs.min().item():.4f}, "
                  f"max={encoded_inputs.max().item():.4f}")

        if self.enable_scale_embeddings:
            # fetching the scale embeddings
            scale_embeddings = self.scale_embeddings.weight  # (num_filters, hidden_dim_0)

            # NaN CHECK ENCODE-5: Scale embeddings
            if not torch.isfinite(scale_embeddings).all():
                print(f"\n🔥 NaN in scale_embeddings.weight")

            if self.debug_mode:
                print(f"[DEBUG ENCODE] scale_embeddings: min={scale_embeddings.min().item():.4f}, "
                      f"max={scale_embeddings.max().item():.4f}")

            # computing the node scale embeddings
            # as linear combinations of the scale embeddings weighted by sparse coefficients of the nodes
            node_scale_embeddings = sparse_coeffs @ scale_embeddings  # (num_nodes, hidden_dim_0)
            # NaN CHECK ENCODE-6: After matmul with scale embeddings
            if not torch.isfinite(node_scale_embeddings).all():
                print(f"\n🔥 NaN in node_scale_embeddings after sparse_coeffs @ scale_embeddings")
                print(
                    f"   node_scale_embeddings stats: min={node_scale_embeddings.min().item():.6f}, max={node_scale_embeddings.max().item():.6f}")

            if self.debug_mode:
                print(f"[DEBUG ENCODE] node_scale_embeddings: min={node_scale_embeddings.min().item():.4f}, "
                      f"max={node_scale_embeddings.max().item():.4f}")

            # concatenating the encoded inputs with the node scale embeddings: (num_nodes, 2 * hidden_dim_0)
            augmented_encoded_inputs = torch.cat([encoded_inputs, node_scale_embeddings], dim=-1)

            # NaN CHECK ENCODE-7: After second concatenation
            if not torch.isfinite(augmented_encoded_inputs).all():
                print(f"\n🔥 NaN in augmented_encoded_inputs after torch.cat")

            # applying the second linear projection layer
            encoded_inputs = self.encoder_projection_2(augmented_encoded_inputs)  # (num_nodes, hidden_dim_0)
            # NaN CHECK ENCODE-8: After second projection (before activation)
            if not torch.isfinite(encoded_inputs).all():
                print(f"\n🔥 NaN after encoder_projection_2 (before activation)")
                print(
                    f"   encoded_inputs stats: min={encoded_inputs.min().item():.6f}, max={encoded_inputs.max().item():.6f}")
                # Check if weights/biases have NaN
                for name, param in self.encoder_projection_2.named_parameters():
                    if not torch.isfinite(param).all():
                        print(f"   🔥 NaN in encoder_projection_2.{name}")

            encoded_inputs = self.act(encoded_inputs)
            # NaN CHECK ENCODE-9: After second activation
            if not torch.isfinite(encoded_inputs).all():
                print(f"\n🔥 NaN after second activation (after encoder_projection_2)")

        if self.debug_mode:
            print(f"[DEBUG ENCODE] after projection_2: min={encoded_inputs.min().item():.4f}, "
                  f"max={encoded_inputs.max().item():.4f}")
        # NaN CHECK ENCODE-10: Final output from encode
        if not torch.isfinite(encoded_inputs).all():
            print(f"\n🔥 NaN in final output from encode()")

        return encoded_inputs, sparse_coeffs

    def process(
            self,
            encoded_inputs: torch.Tensor,
            edge_index: torch.Tensor,
            batch: Optional[torch.Tensor] = None,
    ):
        # k=0: coarsest graph (bottleneck)
        # k=1: finest graph (original)
        # k=2, ..., num_filters-1: intermediate scales
        skip_connections = []
        pool_indices_list = []
        edge_indices = []

        x = encoded_inputs
        current_edge_index = edge_index

        # NaN CHECK MODEL-1: Input to process
        if not torch.isfinite(x).all():
            print(f"\n🔥 NaN in process() input (encoded_inputs)")

        # ============ DOWN PATH ============
        # scales in the GWT coefficients are constructed using the following order:
        # k = 0 corresponds to the scaling functions (H) (lowpass)
        # k = 1 corresponds to the smallest scale (G_1) (highpass)
        # ...
        # k = num_filters - 1 corresponds to the largest scale (G_J) (lowpass)

        # First, process k=1 (finest/original graph)
        level = 1
        edge_indices.append(current_edge_index)

        # Apply message-passing with Pre-LN (Layer Normalization before transformation)
        for conv, norm in zip(self.processor_message_passing_layers[level],
                              self.processor_layer_norms[level]):
            # Pre-LN: norm → activation → conv → residual
            # x → normalize → activate → transform → add to original x
            normalized = norm(x)
            x = x + conv(self.act(normalized), current_edge_index)
        # NaN CHECK MODEL-2: After first message-passing (level=1)
        if not torch.isfinite(x).all():
            print(f"\n🔥 NaN after message-passing at level={level}")

        skip_connections.append(x)

        # Pool the graph for next level
        x_pooled, edge_index_pooled, _, batch, perm, _ = self.processor_pooling_layers[level](
            x=x,
            edge_index=current_edge_index,
            batch=batch
        )
        # NaN CHECK MODEL-3: After first pooling (level=1)
        if not torch.isfinite(x_pooled).all():
            print(f"\n🔥 NaN after pooling at level={level}")

        pool_indices_list.append(perm)
        x = x_pooled
        current_edge_index = edge_index_pooled

        # Process intermediate scales k=2, 3, ..., num_filters-1
        for level in range(2, self.num_filters):
            edge_indices.append(current_edge_index)

            # Apply message-passing at current level
            for conv, norm in zip(self.processor_message_passing_layers[level],
                                  self.processor_layer_norms[level]):
                normalized = norm(x)
                x = x + conv(self.act(normalized), current_edge_index)
                # x = x + self.act(conv(x, current_edge_index))
            # NaN CHECK MODEL-4: After message-passing at intermediate level
            if not torch.isfinite(x).all():
                print(f"\n🔥 NaN after message-passing at level={level}")

            skip_connections.append(x)

            # Pool the graph for next coarser level
            x_pooled, edge_index_pooled, _, batch, perm, _ = self.processor_pooling_layers[level](
                x=x,
                edge_index=current_edge_index,
                batch=batch
            )

            # NaN CHECK MODEL-5: After pooling at intermediate level
            if not torch.isfinite(x_pooled).all():
                print(f"\n🔥 NaN after pooling at level={level}")

            pool_indices_list.append(perm)
            x = x_pooled
            current_edge_index = edge_index_pooled

        # ============ BOTTLENECK ============
        # Process k=0 (coarsest level)
        edge_indices.append(current_edge_index)
        for conv, norm in zip(self.processor_message_passing_layers[0],
                              self.processor_layer_norms[0]):
            normalized = norm(x)
            x = x + conv(self.act(normalized), current_edge_index)
            # x = x + self.act(conv(x, current_edge_index))
        # NaN CHECK MODEL-6: After bottleneck (level=0)
        if not torch.isfinite(x).all():
            print(f"\n🔥 NaN after bottleneck (level=0)")

        # ============ UP PATH ============
        # Start from coarsest (k=0) back to finest (k=1)
        # Order: k=0 -> k=num_filters-1 -> ... -> k=2 -> k=1

        skip_connections.reverse()
        pool_indices_list.reverse()
        edge_indices.reverse()

        # Unpool from k=0 through intermediate scales back to k=1
        for idx, level in enumerate(list(range(2, self.num_filters))[::-1] + [1]):
            # Unpool to finer graph
            x_unpooled = self._unpool(
                x=x,
                pool_indices=pool_indices_list[idx],
                original_size=skip_connections[idx].size(0)
            )
            # NaN CHECK MODEL-7: After unpooling
            if not torch.isfinite(x_unpooled).all():
                print(f"\n🔥 NaN after unpooling to level={level} (idx={idx})")

            # Fuse with skip connection from corresponding encoder layer
            x_fused = torch.cat([x_unpooled, skip_connections[idx]], dim=-1)
            x = self.processor_fusion_layers[idx](x_fused)
            x = self.act(x)

            # NaN CHECK MODEL-8: After fusion
            if not torch.isfinite(x).all():
                print(f"\n🔥 NaN after fusion at level={level} (idx={idx})")

            # Update edge index to finer graph
            current_edge_index = edge_indices[idx + 1]

            # Apply message-passing at current level (shared weights with down path)
            for conv, norm in zip(self.processor_message_passing_layers[level],
                                  self.processor_layer_norms[level]):
                normalized = norm(x)
                x = x + conv(self.act(normalized), current_edge_index)  # ← HAS normalization
            # NaN CHECK MODEL-9: After message-passing in up path
            if not torch.isfinite(x).all():
                print(f"\n🔥 NaN after message-passing in up path at level={level} (idx={idx})")

        # NaN CHECK MODEL-10: Final output from process
        if not torch.isfinite(x).all():
            print(f"\n🔥 NaN in final output from process()")

        return x
        #######################
        # message-passing to mix the scale features across the original graph
        # Down path
        # for each graph-level in range of num_filters
        #   x_pool, edge_index_pool <- pool the graph
        #   message-passing num_message_passing at scale k

        # Bottleneck: is the coarsest graph level

        # Up path
        # for each graph-level in range of num_filters reversed
        #   x_unpool, edge_index_unpool <- unpool the graph
        #   fuse with skip connections from the corresponding encoder's layer
        #   message-passing num_message_passing at scale k
        pass

    def decode(
            self,
            processed_inputs: torch.Tensor,
            sparse_coeffs: torch.Tensor,
            gwt_matrices: List[torch.Tensor] = None,
            chebyshev_data: dict = None,
            batch_ptr: torch.Tensor = None,
    ):
        if self.debug_mode:
            print(f"[DEBUG DECODE] processed_inputs: min={processed_inputs.min().item():.4f}, "
                  f"max={processed_inputs.max().item():.4f}")
        # applying the decoder to reconstruct the graph signal, either by predicting the residual or
        # directly predicting the signal
        if self.predict_residual:
            with torch.no_grad():
                if self.igwt_mode == "dense":
                    x_hat_igwt = batch_igwt(coeffs=sparse_coeffs, Gs=gwt_matrices)
                elif self.igwt_mode == "chebyshev":
                    x_hat_igwt = batch_chebyshev_igwt(
                        coeffs=sparse_coeffs,
                        chebyshev_data=chebyshev_data,
                        batch_ptr=batch_ptr
                    )

                # Normalize IGWT output if normalization is enabled
                # Normalize IGWT output if normalization is enabled
                if self.use_normalization:
                    # x_hat_igwt shape: (N, 1) for single field or (N, num_fields) for multi-field
                    # norm_mean/std shape: (1, num_fields) - we use only the needed field(s)
                    if x_hat_igwt.shape[-1] == 1:
                        # Single field - use first field's stats
                        x_hat_igwt = (x_hat_igwt - self.norm_mean[:, 0:1]) / self.norm_std[:, 0:1]
                    else:
                        # Multi-field - use all fields' stats
                        # ✅ FIX: Handle case where x_hat_igwt.shape[-1] != self.norm_mean.shape[-1]
                        num_output_fields = x_hat_igwt.shape[-1]
                        x_hat_igwt = (x_hat_igwt - self.norm_mean[:, :num_output_fields]) / self.norm_std[:, :num_output_fields]

            residual = self.decoder(processed_inputs)
            x_hat = x_hat_igwt + residual
        else:
            x_hat = self.decoder(processed_inputs)

        if self.debug_mode:
            print(f"[DEBUG DECODE] x_hat: min={x_hat.min().item():.4f}, "
                  f"max={x_hat.max().item():.4f}")
        return x_hat

    @staticmethod
    def _unpool(
            x: torch.Tensor,
            pool_indices: torch.Tensor,
            original_size: int
    ) -> torch.Tensor:
        hidden_dim = x.size(1)
        x_unpooled = torch.zeros(original_size, hidden_dim, device=x.device, dtype=x.dtype)
        # pool_indices contains global node indices from PyG's pooling layer
        x_unpooled[pool_indices] = x

        # Fill zeros with mean of pooled values to avoid sparse tensors
        # This prevents LayerNorm issues with all-zero features
        mean_features = x.mean(dim=0, keepdim=True)  # (1, hidden_dim)
        zero_mask = (x_unpooled.sum(dim=1) == 0)  # Identify all-zero rows
        if zero_mask.any():
            x_unpooled[zero_mask] = mean_features
        return x_unpooled


def main():
    import os

    from paths import ROOT_PROJECT_PATH
    from utils.random import set_random_seed
    from data.dataset import GraphSignalDataset, create_data_splits, create_dataloaders
    from models.loss_functions import compute_loss

    random_seed = 42
    set_random_seed(random_seed=random_seed)

    # Create dataset
    local_storage_dir = os.path.join(ROOT_PROJECT_PATH, "local_storage")
    base_data_dir = os.path.join(local_storage_dir, 'data', '2025-10-23_01-41-32')
    dataset = GraphSignalDataset(base_data_dir=base_data_dir)

    # Split into train/val/test
    train_dataset, val_dataset, test_dataset = create_data_splits(
        dataset,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=random_seed
    )

    # Create dataloaders
    batch_size = 32
    keep_ratio = 0.1
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        generate_mask=True,
        keep_ratio=keep_ratio
    )

    # model
    model = GUNetNeuralIGWT(num_filters=dataset.gwt_config.num_filters,
                            hidden_dim=128,
                            output_dim=1,
                            activation_function='relu',
                            encoder_type='mlp',
                            fusion_type='mlp',
                            decoder_type='mlp',
                            conv_layer_type='SAGEConv',
                            processor_num_layers=dataset.gwt_config.num_scales,
                            pooling_type="sag_pooling",
                            pooling_ratio=0.5,
                            predict_residual=True)
    # train loop over batches
    for batch_index, batch in enumerate(train_loader):
        x_hat = model(coeffs=batch.coeffs,
                      sparsification_mask=batch.sparsification_mask,
                      edge_index=batch.edge_index,
                      gwt_matrices=batch.gwt_matrices,
                      batch=batch.batch)
        loss = compute_loss(x_hat=x_hat,
                            x=batch.x,
                            coeffs=batch.coeffs,
                            sparsification_mask=batch.sparsification_mask,
                            gwt_matrices=batch.gwt_matrices,
                            L_sym=batch.L_sym_matrices,
                            gamma_coeff_consistency=0.0,
                            gamma_regularization=0.001,
                            batch_ptr=batch.ptr)
        pass


if __name__ == '__main__':
    main()
