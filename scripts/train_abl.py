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
      x_hat = forward_batch(model=model, batch=batch, keep_ratio=configs.keep_ratio, random_generator=random_generator)
      test_losses.append(reconstruction_loss(x_hat=x_hat, x=batch.x).item())
      test_errors.append(relative_error(reconstructed=x_hat, signal=batch.x))
  test_loss = sum(test_losses) / len(test_losses)
  test_error = sum(test_errors) / len(test_errors)
  logging.info(f"[*] Test Loss: {test_loss:.4f} | Test Relative Error: {test_error:.4f}")

  # tracking run summary
  tracker.save_summary(
    run_dir=run_dir,
    summary=tracker.RunSummary(
      best_epoch=best_epoch,
      best_val_loss=best_val_loss,
      test_loss=test_loss,
      test_error=test_error,
    )
  )


if __name__ == "__main__":
  main()
