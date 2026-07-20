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
from steps.evaluation import run_gwcs_test_epoch
from model.nigwt import NeuralInverseGraphWaveletTransform
from model.loss import reconstruction_loss
from eval.utils import relative_error


REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
  torch.sparse.check_sparse_tensor_invariants.disable()
  debug: bool = True
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
                                                     gwt_config=gwt_config,
                                                     debug=debug)
  val_dataset = load_or_preprocess_turbulent_split(dataset_dir=dataset_dir / "valid",
                                                   raw_split_dir=raw_dir / "valid",
                                                   gwt_config=gwt_config,
                                                   debug=debug)
  test_dataset = load_or_preprocess_turbulent_split(dataset_dir=dataset_dir / "test",
                                                    raw_split_dir=raw_dir / "test",
                                                    gwt_config=gwt_config,
                                                    debug=debug)

  compression = resolve_gwcs_compression(target_compression_ratio=0.05, num_filters=train_dataset.num_filters)
  assert compression.keep_ratio is not None

  train_loader = DataLoader(dataset=train_dataset, batch_size=training_config.batch_size, shuffle=True,
                            generator=random_generator)
  val_loader = DataLoader(dataset=val_dataset, batch_size=training_config.batch_size, shuffle=False)
  test_loader = DataLoader(dataset=test_dataset, batch_size=training_config.batch_size, shuffle=False)

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
  tracker._log_model_info(model=model, hyperparams={
    "hidden_dim": model_config.hidden_dim,
    "conv_layer": model_config.conv_layer_type,
    "processor_layers": model_config.processor_num_layers,
    "keep_ratio": compression.keep_ratio,
  }, device=device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.lr,
                                weight_decay=training_config.weight_decay)
  lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                            factor=training_config.lr_scheduler_factor,
                                                            patience=training_config.lr_scheduler_patience,
                                                            min_lr=training_config.lr_scheduler_min_lr)
  tracker.save_run_metadata(
    run_dir=run_dir,
    metadata=tracker.RunMetadata(
      seed=seed,
      device=str(device),
      dataset_dir=str(train_dataset.storage_path.relative_to(REPO_ROOT)),
      config={"gwt_config": gwt_config, "model_config": model_config, "training_config": training_config},
      compression=compression,
    )
  )

  best_val_loss = float('inf')
  best_epoch = 0
  epochs_without_improvement = 0
  history: list[tracker.GWCSEpochRecord] = []
  for epoch in range(training_config.num_epochs):
    if epoch > 0:
      logging.info("-" * 60)
    epoch_start = time.perf_counter()
    train_loss = run_epoch_gwcs(loader=train_loader, model=model, device=device, loss_fn=reconstruction_loss,
                                keep_ratio=compression.keep_ratio, random_generator=random_generator,
                                optimizer=optimizer,
                                on_batch_end=lambda i, total, loss, batch_time: tracker.log_batch_progress(
                                  epoch=epoch, total_epochs=training_config.num_epochs, batch_idx=i,
                                  total_batches=total, loss=loss, batch_time=batch_time, prefix="Train batch"))
    val_loss = run_epoch_gwcs(loader=val_loader, model=model, device=device, loss_fn=reconstruction_loss,
                              keep_ratio=compression.keep_ratio, random_generator=random_generator,
                              on_batch_end=lambda i, total, loss, batch_time: tracker.log_batch_progress(
                                epoch=epoch, total_epochs=training_config.num_epochs, batch_idx=i,
                                total_batches=total, loss=loss, batch_time=batch_time, prefix="Val batch"))
    epoch_time = time.perf_counter() - epoch_start
    history.append(tracker.GWCSEpochRecord(epoch=epoch, train_loss=train_loss, val_loss=val_loss))
    logging.info(f"Epoch: {epoch + 1:5d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Time: {epoch_time:.2f}s")
    lr_scheduler.step(val_loss)
    if val_loss < best_val_loss - training_config.early_stopping_min_delta:
      best_val_loss = val_loss
      best_epoch = epoch
      epochs_without_improvement = 0
      checkpoint_path = run_dir / "checkpoint_best.pt"
      torch.save(model.state_dict(), checkpoint_path)
      logging.info(f"New best val loss: {best_val_loss:.4f} at epoch {best_epoch}; Model saved.")
    else:
      epochs_without_improvement += 1
      if epochs_without_improvement >= training_config.early_stopping_patience:
        logging.info(f"Early stopping at epoch: {epoch} "
                     f"(best val loss: {best_val_loss:.4f} at epoch {best_epoch})")
        break
  tracker.save_history(run_dir=run_dir, history=history)

  checkpoint_path = run_dir / "checkpoint_best.pt"
  logging.info(f"Loading best model from epoch {best_epoch} (val loss: {best_val_loss:.4f}) for testing.")
  model.load_state_dict(torch.load(checkpoint_path, map_location=device))

  test_results = run_gwcs_test_epoch(loader=test_loader, model=model, device=device, loss_fn=reconstruction_loss,
                                     error_fn=relative_error, keep_ratio=compression.keep_ratio,
                                     random_generator=random_generator)
  tracker.save_gwcs_test_figures(results=test_results, test_dataset=test_dataset, run_dir=run_dir,
                                 gwt_config=gwt_config, keep_ratio=compression.keep_ratio)
  test_loss_mean, test_loss_std = tracker.summarize([result.loss for result in test_results])
  test_error_mean, test_error_std = tracker.summarize([result.error for result in test_results])
  achieved_compression_ratio_mean, achieved_compression_ratio_std = tracker.summarize(
    [result.compression_ratio for result in test_results])
  logging.info(f"Test Loss: {test_loss_mean:.4f} +/- {test_loss_std:.4f} | "
              f"Test Relative Error: {test_error_mean:.4f} +/- {test_error_std:.4f}")

  tracker.save_summary(
    run_dir=run_dir,
    dataset_path=str(train_dataset.storage_path.relative_to(REPO_ROOT)),
    summary=tracker.RunSummary(
      best_epoch=best_epoch,
      best_val_loss=best_val_loss,
      test_loss_mean=test_loss_mean,
      test_loss_std=test_loss_std,
      test_error_mean=test_error_mean,
      test_error_std=test_error_std,
      achieved_compression_ratio_mean=achieved_compression_ratio_mean,
      achieved_compression_ratio_std=achieved_compression_ratio_std,
    )
  )


if __name__ == "__main__":
  main()
