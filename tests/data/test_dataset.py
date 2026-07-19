import torch
from torch_geometric.loader import DataLoader

from configs.abl import GraphWaveletTransformConfig
from data.dataset import ABLGraphSignalDataset, sample_to_data, split_dataset
from transform.chebyshev_sgwt import _rescale_laplacian_spectrum_to_chebyshev_domain
from model.nigwt import NeuralInverseGraphWaveletTransform

from tests.data.conftest import BATCH_SIZE


def _assert_honestly_coalesced(laplacian: torch.Tensor) -> None:
  with torch.sparse.check_sparse_tensor_invariants():
    torch.sparse_coo_tensor(laplacian.indices(), laplacian.values(),
                           size=laplacian.shape, is_coalesced=True)


def test_single_sample_laplacian_is_honestly_coalesced(
        small_abl_dataset: ABLGraphSignalDataset,
        gwt_config: GraphWaveletTransformConfig,
) -> None:
  for sample_idx, sample in enumerate(small_abl_dataset.samples):
    data = sample_to_data(sample, sample_idx, gwt_config)
    _assert_honestly_coalesced(data.symmetric_normalized_laplacian())
    _assert_honestly_coalesced(data.rescaled_symmetric_normalized_laplacian())


def test_single_sample_laplacian_csr_matmul_matches_dense(
        small_abl_dataset: ABLGraphSignalDataset,
        gwt_config: GraphWaveletTransformConfig,
) -> None:
  # A backend that trusts is_coalesced=True (e.g. CUDA's COO->CSR fast path) will silently
  # compute garbage if the indices aren't actually sorted, even though .to_dense() looks fine.
  for sample_idx, sample in enumerate(small_abl_dataset.samples):
    data = sample_to_data(sample, sample_idx, gwt_config)
    laplacian = data.symmetric_normalized_laplacian()
    v = torch.randn(laplacian.shape[0], 3)
    assert torch.allclose(laplacian.to_sparse_csr() @ v, laplacian.to_dense() @ v, atol=1e-5)


def test_batch_laplacian_is_honestly_coalesced(
        small_abl_dataset: ABLGraphSignalDataset,
) -> None:
  loader = DataLoader(dataset=small_abl_dataset, batch_size=BATCH_SIZE, shuffle=False)
  batch = next(iter(loader))
  _assert_honestly_coalesced(batch.symmetric_normalized_laplacian())
  _assert_honestly_coalesced(batch.rescaled_symmetric_normalized_laplacian())


def test_single_sample_laplacian_matches_source_exactly(
        small_abl_dataset: ABLGraphSignalDataset,
        gwt_config: GraphWaveletTransformConfig,
) -> None:
  for sample_idx, sample in enumerate(small_abl_dataset.samples):
    data = sample_to_data(sample, sample_idx, gwt_config)
    assert torch.equal(data.symmetric_normalized_laplacian().to_dense(),
                       sample.graph.symmetric_normalized_laplacian.to_dense())


def test_single_sample_rescaled_laplacian_matches_direct_rescale(
        small_abl_dataset: ABLGraphSignalDataset,
        gwt_config: GraphWaveletTransformConfig,
) -> None:
  for sample_idx, sample in enumerate(small_abl_dataset.samples):
    data = sample_to_data(sample, sample_idx, gwt_config)
    expected = _rescale_laplacian_spectrum_to_chebyshev_domain(
      symmetric_normalized_laplacian=sample.graph.symmetric_normalized_laplacian,
      spectrum_lower_bound=gwt_config.spectrum_lower_bound,
      spectrum_upper_bound=gwt_config.spectrum_upper_bound,
    )
    assert torch.allclose(data.rescaled_symmetric_normalized_laplacian().to_dense(), expected.to_dense())


def test_dataset_split_round_trip_preserves_laplacians(
        small_abl_dataset: ABLGraphSignalDataset,
        gwt_config: GraphWaveletTransformConfig,
) -> None:
  train_dataset, val_dataset, test_dataset = split_dataset(dataset=small_abl_dataset,
                                                            train_ratio=0.5,
                                                            val_ratio=0.3,
                                                            random_generator=torch.Generator().manual_seed(0))
  for split in (train_dataset, val_dataset, test_dataset):
    for data in split:
      sample_idx = int(data.sample_idx)
      expected = sample_to_data(small_abl_dataset.get_sample(sample_idx), sample_idx, gwt_config)
      assert torch.equal(data.laplacian_index, expected.laplacian_index)
      assert torch.equal(data.laplacian_weight, expected.laplacian_weight)
      assert torch.equal(data.symmetric_normalized_laplacian().to_dense(),
                         expected.symmetric_normalized_laplacian().to_dense())
      assert torch.allclose(data.rescaled_symmetric_normalized_laplacian().to_dense(),
                            expected.rescaled_symmetric_normalized_laplacian().to_dense())


def _block_diag_laplacians(
        dataset: ABLGraphSignalDataset,
        sample_indices: torch.Tensor,
) -> torch.Tensor:
  return torch.block_diag(*[dataset.get_sample(int(idx)).graph.symmetric_normalized_laplacian.to_dense()
                            for idx in sample_indices])


def _block_diag_rescaled_laplacians(
        dataset: ABLGraphSignalDataset,
        sample_indices: torch.Tensor,
        gwt_config: GraphWaveletTransformConfig,
) -> torch.Tensor:
  return torch.block_diag(*[
    _rescale_laplacian_spectrum_to_chebyshev_domain(
      symmetric_normalized_laplacian=dataset.get_sample(int(idx)).graph.symmetric_normalized_laplacian,
      spectrum_lower_bound=gwt_config.spectrum_lower_bound,
      spectrum_upper_bound=gwt_config.spectrum_upper_bound,
    ).to_dense()
    for idx in sample_indices
  ])


def test_batch_laplacian_is_block_diagonal_over_heterogeneous_graphs(
        small_abl_dataset: ABLGraphSignalDataset,
        gwt_config: GraphWaveletTransformConfig,
) -> None:
  num_nodes_per_sample = {int(sample.graph.metadata.num_nodes) for sample in small_abl_dataset.samples}
  assert len(num_nodes_per_sample) > 1, "fixture must produce heterogeneous graph sizes to exercise batching"

  loader = DataLoader(dataset=small_abl_dataset, batch_size=BATCH_SIZE, shuffle=False)
  batch = next(iter(loader))

  assert batch.symmetric_normalized_laplacian().shape == (batch.num_nodes, batch.num_nodes)
  expected_laplacian = _block_diag_laplacians(small_abl_dataset, batch.sample_idx)
  assert torch.equal(batch.symmetric_normalized_laplacian().to_dense(), expected_laplacian)

  expected_rescaled = _block_diag_rescaled_laplacians(small_abl_dataset, batch.sample_idx, gwt_config)
  assert torch.allclose(batch.rescaled_symmetric_normalized_laplacian().to_dense(), expected_rescaled)


def test_batch_to_data_round_trip(
        small_abl_dataset: ABLGraphSignalDataset,
) -> None:
  loader = DataLoader(dataset=small_abl_dataset, batch_size=BATCH_SIZE, shuffle=False)
  batch = next(iter(loader))

  for position in range(batch.num_graphs):
    original = small_abl_dataset[position]
    recovered = batch.get_example(position)
    assert torch.equal(recovered.laplacian_index, original.laplacian_index)
    assert torch.equal(recovered.laplacian_weight, original.laplacian_weight)
    assert torch.equal(recovered.rescaled_laplacian_weight, original.rescaled_laplacian_weight)
    assert torch.equal(recovered.sample_idx, original.sample_idx)


def test_batch_of_size_one_is_consistent(
        small_abl_dataset: ABLGraphSignalDataset,
        gwt_config: GraphWaveletTransformConfig,
) -> None:
  loader = DataLoader(dataset=small_abl_dataset, batch_size=1, shuffle=False)
  batch = next(iter(loader))

  expected_laplacian = _block_diag_laplacians(small_abl_dataset, batch.sample_idx)
  assert torch.equal(batch.symmetric_normalized_laplacian().to_dense(), expected_laplacian)

  expected_rescaled = _block_diag_rescaled_laplacians(small_abl_dataset, batch.sample_idx, gwt_config)
  assert torch.allclose(batch.rescaled_symmetric_normalized_laplacian().to_dense(), expected_rescaled)

  original = small_abl_dataset[0]
  recovered = batch.get_example(0)
  assert torch.equal(recovered.laplacian_index, original.laplacian_index)
  assert torch.equal(recovered.laplacian_weight, original.laplacian_weight)


def test_model_forward_on_heterogeneous_batch(
        small_abl_dataset: ABLGraphSignalDataset,
        gwt_config: GraphWaveletTransformConfig,
) -> None:
  loader = DataLoader(dataset=small_abl_dataset, batch_size=3, shuffle=False)
  batch = next(iter(loader))

  model = NeuralInverseGraphWaveletTransform(
    num_channels=small_abl_dataset.num_channels,
    hidden_dim=8,
    activation_fn="relu",
    conv_layer_type="SAGEConv",
    processor_num_layers=2,
    gwt_mode="chebyshev",
    spectral_kernels=small_abl_dataset.spectral_kernels,
    spectrum_lower_bound=gwt_config.spectrum_lower_bound,
    spectrum_upper_bound=gwt_config.spectrum_upper_bound,
    chebyshev_degree=gwt_config.chebyshev_degree,
  )

  sparsification_mask = torch.ones_like(batch.gwt_coeffs, dtype=torch.bool)
  sparse_gwt_coeffs = batch.gwt_coeffs * sparsification_mask.to(batch.gwt_coeffs.dtype)

  x_hat = model(
    sparse_gwt_coeffs=sparse_gwt_coeffs,
    sparsification_mask=sparsification_mask,
    edge_index=batch.edge_index,
    symmetric_normalized_laplacian=batch.symmetric_normalized_laplacian(),
    rescaled_symmetric_normalized_laplacian=batch.rescaled_symmetric_normalized_laplacian(),
  )

  assert x_hat.shape == (batch.num_nodes, small_abl_dataset.num_channels)
