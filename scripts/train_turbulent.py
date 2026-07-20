import logging
import time
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

import tracker
from compression import resolve_gwcs_compression
from configs.abl import GraphWaveletTransformConfig, GWCSConfig, TrainingConfig
from steps.turbulent_preprocessing import load_or_preprocess_turbulent_split
from steps.training import run_epoch_gwcs
from model.nigwt import NeuralInverseGraphWaveletTransform
from model.loss import reconstruction_loss


REPO_ROOT = Path(__file__).resolve().parent.parent
NUM_EPOCHS = 5


def main():
  torch.sparse.check_sparse_tensor_invariants.disable()

  seed = 42
  torch.manual_seed(seed)
  random_generator = torch.Generator().manual_seed(seed)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  gwt_config = GraphWaveletTransformConfig(gwt_mode="chebyshev")
  model_config = GWCSConfig()
  training_config = TrainingConfig()

  raw_dir = REPO_ROOT / "datasets" / "turbulent_radiative_layer_2D" / "raw" / "data"
  dataset_dir = REPO_ROOT / "datasets" / "turbulent_radiative_layer_2D" / "processed"

  run_dir = tracker.create_run_dir(REPO_ROOT / "runs" / "turbulent_GWCS")
  tracker.setup_logging(run_dir)
  tracker.log_run_info(run_dir=run_dir, dataset_dir=dataset_dir, device=device, config=gwt_config)

  train_dataset = load_or_preprocess_turbulent_split(dataset_dir=dataset_dir / "train",
                                                     raw_split_dir=raw_dir / "train",
                                                     gwt_config=gwt_config)
  val_dataset = load_or_preprocess_turbulent_split(dataset_dir=dataset_dir / "valid",
                                                   raw_split_dir=raw_dir / "valid",
                                                   gwt_config=gwt_config)

  compression = resolve_gwcs_compression(target_compression_ratio=0.05, num_filters=train_dataset.num_filters)
  assert compression.keep_ratio is not None

  train_loader = DataLoader(dataset=train_dataset, batch_size=training_config.batch_size, shuffle=True,
                            generator=random_generator)
  val_loader = DataLoader(dataset=val_dataset, batch_size=training_config.batch_size, shuffle=False)

  model = NeuralInverseGraphWaveletTransform(
    num_channels=train_dataset.num_channels,
    hidden_dim=model_config.hidden_dim,
    activation_fn=model_config.activation_fn,
    conv_layer_type=model_config.conv_layer_type,
    processor_num_layers=model_config.processor_num_layers,
    gwt_mode=gwt_config.gwt_mode,
    spectral_kernels=train_dataset.spectral_kernels,
    spectrum_lower_bound=gwt_config.spectrum_lower_bound,
    spectrum_upper_bound=gwt_config.spectrum_upper_bound,
    chebyshev_degree=gwt_config.chebyshev_degree,
  ).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.lr,
                                weight_decay=training_config.weight_decay)

  for epoch in range(NUM_EPOCHS):
    epoch_start = time.perf_counter()
    train_loss = run_epoch_gwcs(loader=train_loader, model=model, device=device, loss_fn=reconstruction_loss,
                                keep_ratio=compression.keep_ratio, random_generator=random_generator,
                                optimizer=optimizer,
                                on_batch_end=lambda i, total, loss, batch_time: tracker.log_batch_progress(
                                  epoch=epoch, total_epochs=NUM_EPOCHS, batch_idx=i, total_batches=total,
                                  loss=loss, batch_time=batch_time, prefix="Train batch"))
    val_loss = run_epoch_gwcs(loader=val_loader, model=model, device=device, loss_fn=reconstruction_loss,
                              keep_ratio=compression.keep_ratio, random_generator=random_generator,
                              on_batch_end=lambda i, total, loss, batch_time: tracker.log_batch_progress(
                                epoch=epoch, total_epochs=NUM_EPOCHS, batch_idx=i, total_batches=total,
                                loss=loss, batch_time=batch_time, prefix="Val batch"))
    epoch_time = time.perf_counter() - epoch_start
    logging.info(f"Epoch: {epoch:5d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Time: {epoch_time:.2f}s")


if __name__ == "__main__":
  main()
