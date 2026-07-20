import time
import logging
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

import tracker
from compression import resolve_sgae_compression
from configs.abl import default_abl_configs_sgae
from data.dataset import split_dataset
from steps.data_generation import load_or_generate_abl_dataset
from steps.training import run_epoch_sgae
from steps.evaluation import run_sgae_test_epoch
from model.sgae import SparseVariationalGraphAutoEncoder
from model.loss import reconstruction_loss
from scheduler import ste_temperature_at_epoch
from eval.utils import relative_error

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
  torch.sparse.check_sparse_tensor_invariants.disable()
  configs = default_abl_configs_sgae()
  torch.manual_seed(configs.seed)
  random_generator: torch.Generator | None = torch.Generator().manual_seed(configs.seed)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  # set to an existing run dir (e.g. REPO_ROOT / "runs" / "ABL_SGAE" / "<run_id>") to resume
  resume_from: Path | None = None
  dataset_dir = REPO_ROOT / "datasets" / "ABL"
  run_dir = tracker.resolve_run_dir(REPO_ROOT / "runs" / "ABL_SGAE", resume_from)
  tracker.setup_logging(run_dir)
  tracker.log_run_info(run_dir=run_dir, dataset_dir=dataset_dir, device=device, config=configs)

  # loading/generating dataset
  dataset = load_or_generate_abl_dataset(dataset_dir=dataset_dir,
                                         graph_config=configs.generation_config.graph_config,
                                         signal_config=configs.generation_config.signal_config,
                                         gwt_config=configs.generation_config.gwt_config,
                                         random_generator=random_generator)
  compression = resolve_sgae_compression(target_compression_ratio=configs.target_compression_ratio,
                                         num_channels=dataset.num_channels,
                                         latent_dim=configs.model_config.latent_dim)
  assert compression.sparsity_ratio is not None
  # creating data loaders
  train_dataset, val_dataset, test_dataset = split_dataset(dataset=dataset,
                                                           train_ratio=configs.training_config.train_ratio,
                                                           val_ratio=configs.training_config.val_ratio,
                                                           random_generator=random_generator)
  train_loader = DataLoader(dataset=train_dataset,
                            batch_size=configs.training_config.batch_size,
                            shuffle=True,
                            generator=random_generator)
  val_loader = DataLoader(dataset=val_dataset, batch_size=configs.training_config.batch_size, shuffle=False)
  test_loader = DataLoader(dataset=test_dataset, batch_size=configs.training_config.batch_size, shuffle=False)

  # initializing model, optimizer, and schedulers
  model = SparseVariationalGraphAutoEncoder(
    num_channels=dataset.num_channels,
    latent_dim=configs.model_config.latent_dim,
    activation_fn=configs.model_config.activation_fn,
    conv_layer_type=configs.model_config.conv_layer_type,
    num_layers=configs.model_config.num_layers,
    is_variational=configs.model_config.is_variational,
  ).to(device)
  tracker.log_sgae_model_info(model=model, configs=configs, sparsity_ratio=compression.sparsity_ratio, device=device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=configs.training_config.lr,
                                weight_decay=configs.training_config.weight_decay)
  lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                            factor=configs.training_config.lr_scheduler_factor,
                                                            patience=configs.training_config.lr_scheduler_patience,
                                                            min_lr=configs.training_config.lr_scheduler_min_lr)
  # tracking run metadata
  tracker.save_run_metadata(
    run_dir=run_dir,
    metadata=tracker.RunMetadata(
      seed=configs.seed,
      device=str(device),
      dataset_dir=str(dataset.storage_path.relative_to(REPO_ROOT)),
      config=configs,
      compression=compression,
    )
  )

  # training loop
  start_epoch, best_val_loss, best_epoch, epochs_without_improvement, history = tracker.init_training_state(
    run_dir=run_dir, resume_from=resume_from, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
    device=device)
  for epoch in range(start_epoch, configs.training_config.num_epochs):
    epoch_start = time.perf_counter()
    temperature = ste_temperature_at_epoch(epoch=epoch, mode=configs.model_config.ste_temperature_mode,
                                           temperature_init=configs.model_config.ste_temperature_init,
                                           temperature_final=configs.model_config.ste_temperature_final,
                                           anneal_epochs=configs.model_config.ste_temperature_anneal_epochs)
    # train
    train_loss, train_recon, train_kl = run_epoch_sgae(loader=train_loader, model=model, device=device,
                                                       sparsity_ratio=compression.sparsity_ratio,
                                                       temperature=temperature,
                                                       kl_beta=configs.model_config.kl_beta, optimizer=optimizer)
    # validation
    val_loss, val_recon, val_kl = run_epoch_sgae(loader=val_loader, model=model, device=device,
                                                 sparsity_ratio=compression.sparsity_ratio,
                                                 temperature=temperature,
                                                 kl_beta=configs.model_config.kl_beta)
    epoch_time = time.perf_counter() - epoch_start
    history.append(tracker.SGAEEpochRecord(epoch=epoch, train_loss=train_loss, train_recon=train_recon,
                                           train_kl=train_kl, val_loss=val_loss, val_recon=val_recon,
                                           val_kl=val_kl))
    logging.info(f"Epoch: {epoch:5d} | "
                 f"Train Loss: {train_loss:.4f} (recon {train_recon:.4f}, kl {train_kl:.4f}) | "
                 f"Val Loss: {val_loss:.4f} (recon {val_recon:.4f}, kl {val_kl:.4f}) | "
                 f"Temperature: {temperature:.4f} | "
                 f"Time: {epoch_time:.2f}s")
    # scheduling lr
    lr_scheduler.step(val_loss)
    # checkpointing best model so far
    if val_loss < best_val_loss - configs.training_config.early_stopping_min_delta:
      best_val_loss = val_loss
      best_epoch = epoch
      epochs_without_improvement = 0
      checkpoint_path = run_dir / "checkpoint_best.pt"
      torch.save(model.state_dict(), checkpoint_path)
      logging.info(f"New best val loss: {best_val_loss:.4f} at epoch {best_epoch}; Model saved.")
    else:
      # counting for early stopping
      epochs_without_improvement += 1
      if epochs_without_improvement >= configs.training_config.early_stopping_patience:
        logging.info(f"Early stopping at epoch: {epoch} "
                     f"(best val loss: {best_val_loss:.4f} at epoch {best_epoch})")
        break
    tracker.save_checkpoint(run_dir=run_dir, epoch=epoch, model=model, optimizer=optimizer,
                            lr_scheduler=lr_scheduler, best_val_loss=best_val_loss, best_epoch=best_epoch,
                            epochs_without_improvement=epochs_without_improvement, history=history)
  # tracking training history
  tracker.save_history(run_dir=run_dir, history=history)

  # loading best model before testing
  checkpoint_path = run_dir / "checkpoint_best.pt"
  logging.info(f"Loading best model from epoch {best_epoch} (val loss: {best_val_loss:.4f}) for testing.")
  model.load_state_dict(torch.load(checkpoint_path, map_location=device))

  # test
  test_temperature = ste_temperature_at_epoch(epoch=best_epoch, mode=configs.model_config.ste_temperature_mode,
                                              temperature_init=configs.model_config.ste_temperature_init,
                                              temperature_final=configs.model_config.ste_temperature_final,
                                              anneal_epochs=configs.model_config.ste_temperature_anneal_epochs)
  test_results = run_sgae_test_epoch(loader=test_loader, model=model, device=device, loss_fn=reconstruction_loss,
                                     error_fn=relative_error, sparsity_ratio=compression.sparsity_ratio,
                                     temperature=test_temperature, num_channels=dataset.num_channels,
                                     latent_dim=configs.model_config.latent_dim)
  tracker.save_sgae_test_figures(results=test_results, test_dataset=test_dataset, run_dir=run_dir)

  test_loss_mean, test_loss_std = tracker.summarize([result.loss for result in test_results])
  test_error_mean, test_error_std = tracker.summarize([result.error for result in test_results])
  achieved_compression_ratio_mean, achieved_compression_ratio_std = tracker.summarize(
    [result.compression_ratio for result in test_results])
  logging.info(f"Test Loss: {test_loss_mean:.4f} +/- {test_loss_std:.4f} | "
              f"Test Relative Error: {test_error_mean:.4f} +/- {test_error_std:.4f}")

  # tracking run summary
  tracker.save_summary(
    run_dir=run_dir,
    dataset_path=str(dataset.storage_path.relative_to(REPO_ROOT)),
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
