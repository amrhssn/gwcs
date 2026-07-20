from pathlib import Path

import pytest
import torch

import tracker

FIXED_INPUT = torch.arange(12.0).reshape(3, 4)
FIXED_TARGET = torch.zeros(3, 2)


def _build_model_optimizer_scheduler(seed: int = 0):
  torch.manual_seed(seed)
  model = torch.nn.Linear(4, 2)
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
  lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
  return model, optimizer, lr_scheduler


def _train_one_epoch(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> float:
  optimizer.zero_grad()
  loss = torch.nn.functional.mse_loss(model(FIXED_INPUT), FIXED_TARGET)
  loss.backward()
  optimizer.step()
  return loss.item()


def _run_epochs(run_dir, epochs, model, optimizer, lr_scheduler, best_val_loss, best_epoch,
                epochs_without_improvement, history):
  for epoch in epochs:
    train_loss = _train_one_epoch(model, optimizer)
    lr_scheduler.step(train_loss)
    history = history + [tracker.GWCSEpochRecord(epoch=epoch, train_loss=train_loss, val_loss=train_loss)]
    if train_loss < best_val_loss:
      best_val_loss, best_epoch, epochs_without_improvement = train_loss, epoch, 0
    else:
      epochs_without_improvement += 1
    tracker.save_checkpoint(run_dir=run_dir, epoch=epoch, model=model, optimizer=optimizer,
                            lr_scheduler=lr_scheduler, best_val_loss=best_val_loss, best_epoch=best_epoch,
                            epochs_without_improvement=epochs_without_improvement, history=history)
  return best_val_loss, best_epoch, epochs_without_improvement, history


def _as_scalar(value):
  return value.item() if torch.is_tensor(value) else value


def test_resolve_run_dir_creates_new_dir_when_resume_from_is_none(tmp_path: Path) -> None:
  run_dir = tracker.resolve_run_dir(tmp_path, None)
  assert run_dir.exists()
  assert run_dir.parent == tmp_path


def test_resolve_run_dir_returns_resume_from_unchanged(tmp_path: Path) -> None:
  existing = tmp_path / "20260101-000000"
  existing.mkdir()
  run_dir = tracker.resolve_run_dir(tmp_path, existing)
  assert run_dir == existing


def test_resolve_run_dir_does_not_create_resume_from_dir(tmp_path: Path) -> None:
  resume_from = tmp_path / "not_yet_created"
  run_dir = tracker.resolve_run_dir(tmp_path, resume_from)
  assert run_dir == resume_from
  assert not run_dir.exists()


def test_init_training_state_starts_fresh_without_resume(tmp_path: Path) -> None:
  model, optimizer, lr_scheduler = _build_model_optimizer_scheduler()
  start_epoch, best_val_loss, best_epoch, epochs_without_improvement, history = tracker.init_training_state(
    run_dir=tmp_path, resume_from=None, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
    device=torch.device("cpu"))
  assert start_epoch == 0
  assert best_val_loss == float('inf')
  assert best_epoch == 0
  assert epochs_without_improvement == 0
  assert history == []


def test_init_training_state_starts_fresh_when_no_checkpoint_exists_yet(tmp_path: Path) -> None:
  model, optimizer, lr_scheduler = _build_model_optimizer_scheduler()
  start_epoch, *_ = tracker.init_training_state(
    run_dir=tmp_path, resume_from=tmp_path, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
    device=torch.device("cpu"))
  assert start_epoch == 0


def test_save_checkpoint_writes_checkpoint_last(tmp_path: Path) -> None:
  model, optimizer, lr_scheduler = _build_model_optimizer_scheduler()
  tracker.save_checkpoint(run_dir=tmp_path, epoch=0, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
                          best_val_loss=1.0, best_epoch=0, epochs_without_improvement=0, history=[])
  assert (tmp_path / "checkpoint_last.pt").exists()


def test_load_checkpoint_restores_model_weights_and_scalars(tmp_path: Path) -> None:
  model_a, optimizer_a, lr_scheduler_a = _build_model_optimizer_scheduler(seed=1)
  _train_one_epoch(model_a, optimizer_a)
  tracker.save_checkpoint(run_dir=tmp_path, epoch=3, model=model_a, optimizer=optimizer_a,
                          lr_scheduler=lr_scheduler_a, best_val_loss=0.5, best_epoch=2,
                          epochs_without_improvement=1, history=[])

  model_b, optimizer_b, lr_scheduler_b = _build_model_optimizer_scheduler(seed=99)
  checkpoint = tracker.load_checkpoint(run_dir=tmp_path, model=model_b, optimizer=optimizer_b,
                                       lr_scheduler=lr_scheduler_b, device=torch.device("cpu"))

  for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
    assert torch.allclose(p_a, p_b)
  assert checkpoint["epoch"] == 3
  assert checkpoint["best_val_loss"] == 0.5
  assert checkpoint["best_epoch"] == 2
  assert checkpoint["epochs_without_improvement"] == 1


def test_load_checkpoint_restores_optimizer_state(tmp_path: Path) -> None:
  model_a, optimizer_a, lr_scheduler_a = _build_model_optimizer_scheduler()
  _train_one_epoch(model_a, optimizer_a)
  _train_one_epoch(model_a, optimizer_a)
  tracker.save_checkpoint(run_dir=tmp_path, epoch=1, model=model_a, optimizer=optimizer_a,
                          lr_scheduler=lr_scheduler_a, best_val_loss=1.0, best_epoch=0,
                          epochs_without_improvement=0, history=[])

  model_b, optimizer_b, lr_scheduler_b = _build_model_optimizer_scheduler()
  tracker.load_checkpoint(run_dir=tmp_path, model=model_b, optimizer=optimizer_b, lr_scheduler=lr_scheduler_b,
                          device=torch.device("cpu"))

  state_a = optimizer_a.state_dict()["state"]
  state_b = optimizer_b.state_dict()["state"]
  assert state_a.keys() == state_b.keys()
  for key in state_a:
    assert _as_scalar(state_a[key]["step"]) == _as_scalar(state_b[key]["step"])
    assert torch.allclose(state_a[key]["exp_avg"], state_b[key]["exp_avg"])
    assert torch.allclose(state_a[key]["exp_avg_sq"], state_b[key]["exp_avg_sq"])


def test_load_checkpoint_restores_lr_scheduler_state(tmp_path: Path) -> None:
  model_a, optimizer_a, lr_scheduler_a = _build_model_optimizer_scheduler()
  for val_loss in [1.0, 1.0, 1.0]:  # exceeds patience=1, forcing a bad-epoch streak and an lr reduction
    lr_scheduler_a.step(val_loss)
  tracker.save_checkpoint(run_dir=tmp_path, epoch=2, model=model_a, optimizer=optimizer_a,
                          lr_scheduler=lr_scheduler_a, best_val_loss=1.0, best_epoch=0,
                          epochs_without_improvement=2, history=[])

  model_b, optimizer_b, lr_scheduler_b = _build_model_optimizer_scheduler()
  tracker.load_checkpoint(run_dir=tmp_path, model=model_b, optimizer=optimizer_b, lr_scheduler=lr_scheduler_b,
                          device=torch.device("cpu"))

  assert lr_scheduler_b.state_dict() == lr_scheduler_a.state_dict()
  assert optimizer_b.param_groups[0]["lr"] == optimizer_a.param_groups[0]["lr"]


def test_init_training_state_resumes_scalar_state_and_history_from_checkpoint(tmp_path: Path) -> None:
  model_a, optimizer_a, lr_scheduler_a = _build_model_optimizer_scheduler()
  history = [tracker.GWCSEpochRecord(epoch=0, train_loss=1.0, val_loss=0.9),
             tracker.GWCSEpochRecord(epoch=1, train_loss=0.8, val_loss=0.5)]
  tracker.save_checkpoint(run_dir=tmp_path, epoch=1, model=model_a, optimizer=optimizer_a,
                          lr_scheduler=lr_scheduler_a, best_val_loss=0.5, best_epoch=1,
                          epochs_without_improvement=0, history=history)

  model_b, optimizer_b, lr_scheduler_b = _build_model_optimizer_scheduler()
  start_epoch, best_val_loss, best_epoch, epochs_without_improvement, resumed_history = tracker.init_training_state(
    run_dir=tmp_path, resume_from=tmp_path, model=model_b, optimizer=optimizer_b, lr_scheduler=lr_scheduler_b,
    device=torch.device("cpu"))

  assert start_epoch == 2
  assert best_val_loss == 0.5
  assert best_epoch == 1
  assert epochs_without_improvement == 0
  assert resumed_history == history


def test_checkpoint_round_trip_matches_uninterrupted_training(tmp_path: Path) -> None:
  """A run that crashes partway through and resumes must reach the same final state as one uninterrupted run."""
  num_epochs = 6
  crash_after = 3

  baseline_dir = tmp_path / "baseline"
  baseline_dir.mkdir()
  model, optimizer, lr_scheduler = _build_model_optimizer_scheduler(seed=7)
  baseline_best_val_loss, baseline_best_epoch, _, baseline_history = _run_epochs(
    baseline_dir, range(0, num_epochs), model, optimizer, lr_scheduler, float('inf'), 0, 0, [])

  resumed_dir = tmp_path / "resumed"
  resumed_dir.mkdir()
  model_1, optimizer_1, lr_scheduler_1 = _build_model_optimizer_scheduler(seed=7)
  _run_epochs(resumed_dir, range(0, crash_after), model_1, optimizer_1, lr_scheduler_1, float('inf'), 0, 0, [])

  # simulate a process restart: brand new model/optimizer/scheduler objects, state comes only from disk
  model_2, optimizer_2, lr_scheduler_2 = _build_model_optimizer_scheduler(seed=7)
  start_epoch, best_val_loss, best_epoch, epochs_without_improvement, history = tracker.init_training_state(
    run_dir=resumed_dir, resume_from=resumed_dir, model=model_2, optimizer=optimizer_2, lr_scheduler=lr_scheduler_2,
    device=torch.device("cpu"))
  assert start_epoch == crash_after

  resumed_best_val_loss, resumed_best_epoch, _, resumed_history = _run_epochs(
    resumed_dir, range(start_epoch, num_epochs), model_2, optimizer_2, lr_scheduler_2, best_val_loss, best_epoch,
    epochs_without_improvement, history)

  assert resumed_best_val_loss == pytest.approx(baseline_best_val_loss)
  assert resumed_best_epoch == baseline_best_epoch
  assert [r.epoch for r in resumed_history] == list(range(num_epochs))
  assert [r.train_loss for r in resumed_history] == pytest.approx([r.train_loss for r in baseline_history])
  for p_base, p_resumed in zip(model.parameters(), model_2.parameters()):
    assert torch.allclose(p_base, p_resumed)
