import time
import logging
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

import tracker
from configs.abl import default_abl_configs
from data.dataset import split_dataset
from steps.data_generation import load_or_generate_abl_dataset
from steps.training import run_epoch, forward_batch
from model.nigwt import NeuralInverseGraphWaveletTransform
from model.loss import reconstruction_loss
from eval.utils import relative_error
from visualization import (
  visualize_grid_reconstruction,
  visualize_grid_gwt_coefficients,
  visualize_grid_sparsification_mask,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
  torch.sparse.check_sparse_tensor_invariants.disable()
  configs = default_abl_configs()
  torch.manual_seed(configs.seed)
  random_generator: torch.Generator | None = torch.Generator().manual_seed(configs.seed)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  dataset_dir = REPO_ROOT / "datasets" / "ABL"
  run_dir = tracker.create_run_dir(REPO_ROOT / "runs" / "ABL")
  tracker.setup_logging(run_dir)

  # loading/generating dataset
  dataset = load_or_generate_abl_dataset(dataset_dir=dataset_dir,
                                         graph_config=configs.graph_config,
                                         signal_config=configs.signal_config,
                                         gwt_config=configs.gwt_config,
                                         random_generator=random_generator)
  # creating data loaders
  train_dataset, val_dataset, test_dataset = split_dataset(dataset=dataset,
                                                           train_ratio=configs.train_ratio,
                                                           val_ratio=configs.val_ratio,
                                                           random_generator=random_generator)
  train_loader = DataLoader(dataset=train_dataset,
                            batch_size=configs.batch_size,
                            shuffle=True,
                            generator=random_generator)
  val_loader = DataLoader(dataset=val_dataset, batch_size=configs.batch_size, shuffle=False)
  test_loader = DataLoader(dataset=test_dataset, batch_size=configs.batch_size, shuffle=False)

  # initializing model, optimizer, and schedulers
  model = NeuralInverseGraphWaveletTransform(
    num_channels=dataset.num_channels,
    hidden_dim=configs.hidden_dim,
    activation_fn=configs.activation_fn,
    conv_layer_type=configs.conv_layer_type,
    processor_num_layers=configs.processor_num_layers,
    gwt_mode=dataset.configs.gwt_config.gwt_mode,
    spectral_kernels=dataset.spectral_kernels,
    spectrum_lower_bound=dataset.configs.gwt_config.spectrum_lower_bound,
    spectrum_upper_bound=dataset.configs.gwt_config.spectrum_upper_bound,
    chebyshev_degree=dataset.configs.gwt_config.chebyshev_degree
  ).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=configs.lr, weight_decay=configs.weight_decay)
  lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=configs.lr_scheduler_factor,
                                                            patience=configs.lr_scheduler_patience,
                                                            min_lr=configs.lr_scheduler_min_lr)
  # tracking run metadata
  tracker.save_run_metadata(
    run_dir=run_dir,
    metadata=tracker.RunMetadata(
      seed=configs.seed,
      device=str(device),
      dataset_dir=str(dataset.storage_path.relative_to(REPO_ROOT)),
      config=configs,
    )
  )

  # training loop
  best_val_loss = float('inf')
  best_epoch = 0
  epochs_without_improvement = 0
  history: list[tracker.EpochRecord] = []
  for epoch in range(configs.num_epochs):
    epoch_start = time.perf_counter()
    # train
    train_loss = run_epoch(loader=train_loader, model=model, device=device, loss_fn=reconstruction_loss,
                           keep_ratio=configs.keep_ratio, random_generator=random_generator, optimizer=optimizer)
    # validation
    val_loss = run_epoch(loader=val_loader, model=model, device=device, loss_fn=reconstruction_loss,
                         keep_ratio=configs.keep_ratio, random_generator=random_generator)
    epoch_time = time.perf_counter() - epoch_start
    history.append(tracker.EpochRecord(epoch=epoch, train_loss=train_loss, val_loss=val_loss))
    logging.info(f"[*] Epoch: {epoch:5d} | "
                 f"Train Loss: {train_loss:.4f} | "
                 f"Val Loss: {val_loss:.4f} | "
                 f"Time: {epoch_time:.2f}s")
    # scheduling lr
    lr_scheduler.step(val_loss)
    # checkpointing best model so far
    if val_loss < best_val_loss - configs.early_stopping_min_delta:
      best_val_loss = val_loss
      best_epoch = epoch
      epochs_without_improvement = 0
      torch.save(model.state_dict(), run_dir / "checkpoint_best.pt")
    else:
      # counting for early stopping
      epochs_without_improvement += 1
      if epochs_without_improvement >= configs.early_stopping_patience:
        logging.info(f"[*] Early stopping at epoch: {epoch} "
                     f"(best val loss: {best_val_loss:.4f} at epoch {best_epoch})")
        break
  # tracking training history
  tracker.save_history(run_dir=run_dir, history=history)

  # test
  model.eval()
  test_losses, test_errors = [], []
  with torch.no_grad():
    for batch in test_loader:
      batch = batch.to(device)
      x_hat, mask = forward_batch(model=model, batch=batch, keep_ratio=configs.keep_ratio,
                                  random_generator=random_generator)
      for i in range(batch.num_graphs):
        start, end = batch.ptr[i].item(), batch.ptr[i + 1].item()
        signal = batch.x[start:end].cpu()
        reconstructed = x_hat[start:end].cpu()
        coeffs = batch.gwt_coeffs[start:end].cpu()
        sample_mask = mask[start:end].cpu()
        sample_idx = int(batch.sample_idx[i])

        test_losses.append(reconstruction_loss(reconstructed=reconstructed, signal=signal).item())
        test_errors.append(relative_error(reconstructed=reconstructed, signal=signal))

        graph_metadata = test_dataset.get_sample(sample_idx).graph.metadata
        if graph_metadata.graph_type != 'grid':
          logging.warning(f"[*] Skipping figures for sample {sample_idx}: "
                          f"visualization not implemented for graph_type={graph_metadata.graph_type!r}")
          continue

        rows, cols = graph_metadata.height, graph_metadata.width
        filename = f"sample_{sample_idx:04d}.png"

        recon_fig = visualize_grid_reconstruction(reconstructed=reconstructed, signal=signal,
                                                  rows=rows, cols=cols, gwt_mode=configs.gwt_config.gwt_mode)
        tracker.save_figure(run_dir=run_dir, category="reconstruction", filename=filename, fig=recon_fig)

        coeffs_fig = visualize_grid_gwt_coefficients(coeffs=coeffs, signal=signal, scales=configs.gwt_config.scales,
                                                     rows=rows, cols=cols, gwt_mode=configs.gwt_config.gwt_mode)
        tracker.save_figure(run_dir=run_dir, category="coefficients", filename=filename, fig=coeffs_fig)

        mask_fig = visualize_grid_sparsification_mask(mask=sample_mask, rows=rows, cols=cols)
        tracker.save_figure(run_dir=run_dir, category="mask", filename=filename, fig=mask_fig)

  test_loss_mean, test_loss_std = tracker.summarize(test_losses)
  test_error_mean, test_error_std = tracker.summarize(test_errors)
  logging.info(f"[*] Test Loss: {test_loss_mean:.4f} +/- {test_loss_std:.4f} | "
              f"Test Relative Error: {test_error_mean:.4f} +/- {test_error_std:.4f}")

  # tracking run summary
  tracker.save_summary(
    run_dir=run_dir,
    summary=tracker.RunSummary(
      best_epoch=best_epoch,
      best_val_loss=best_val_loss,
      test_loss_mean=test_loss_mean,
      test_loss_std=test_loss_std,
      test_error_mean=test_error_mean,
      test_error_std=test_error_std,
    )
  )


if __name__ == "__main__":
  main()
