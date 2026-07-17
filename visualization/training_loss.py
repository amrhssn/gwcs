from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def moving_average(arr, window_size):
    if window_size is None or window_size <= 1 or len(arr) == 0:
        return np.asarray(arr), np.arange(len(arr))
    ma = np.convolve(arr, np.ones(window_size)/window_size, mode='valid')
    x = np.arange(window_size - 1, len(arr))
    return ma, x


def plot_train_val_losses(
    train_losses_history: dict,
    val_losses_history: dict,
    window_size: Optional[int] = 50,
    figsize=(12, 8),
    sparsity_scheduler_final_epoch: Optional[int] = None,
    suptitle: Optional[str] = None,
):
    """
    Visualize training and validation losses with optional moving average.

    Args:
        train_losses_history: dict with keys ['total','recon','coeffs_consistency','regularization'] -> list[float]
        val_losses_history:   dict with same keys as train_losses_history
        window_size: moving average window (set None or <=1 to disable)
        figsize: figure size
        sparsity_scheduler_final_epoch: draw a vertical dashed line if provided
        suptitle: overall figure title
    """
    keys = ['total', 'recon', 'coeffs_consistency', 'regularization']
    for d in (train_losses_history, val_losses_history):
        for k in keys:
            d.setdefault(k, [])

    epochs_train = len(train_losses_history['total'])
    epochs_val = len(val_losses_history['total'])

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    if suptitle:
        fig.suptitle(suptitle)

    # Panel 1: Total loss
    ax = axes[0]
    t_ma, t_x = moving_average(train_losses_history['total'], window_size)
    v_ma, v_x = moving_average(val_losses_history['total'], window_size)

    ax.plot(np.arange(epochs_train), train_losses_history['total'], color='tab:blue', alpha=0.25, label='Train Total (raw)')
    ax.plot(np.arange(epochs_val),   val_losses_history['total'],   color='tab:orange', alpha=0.25, label='Val Total (raw)')
    ax.plot(t_x, t_ma, color='tab:blue', linewidth=2, label=f'Train Total (MA{window_size})')
    ax.plot(v_x, v_ma, color='tab:orange', linewidth=2, label=f'Val Total (MA{window_size})')

    if sparsity_scheduler_final_epoch is not None:
        ax.axvline(x=sparsity_scheduler_final_epoch, color='red', linestyle='--', linewidth=1.5,
                   label=f'Scheduler Final Epoch ({sparsity_scheduler_final_epoch})')

    ax.set_ylabel('Total Loss')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    # Panel 2: Component losses
    ax = axes[1]
    colors = {
        'recon': 'tab:green',
        'coeffs_consistency': 'tab:purple',
        'regularization': 'tab:brown',
    }
    for k in ['recon', 'coeffs_consistency', 'regularization']:
        tr_raw = train_losses_history[k]
        va_raw = val_losses_history[k]
        tr_ma, tr_x = moving_average(tr_raw, window_size)
        va_ma, va_x = moving_average(va_raw, window_size)

        # raw
        ax.plot(np.arange(len(tr_raw)), tr_raw, color=colors[k], alpha=0.18, label=f'Train {k} (raw)')
        ax.plot(np.arange(len(va_raw)), va_raw, color=colors[k], alpha=0.18, linestyle='--', label=f'Val {k} (raw)')
        # smoothed
        ax.plot(tr_x, tr_ma, color=colors[k], linewidth=2, label=f'Train {k} (MA{window_size})')
        ax.plot(va_x, va_ma, color=colors[k], linewidth=2, linestyle='--', label=f'Val {k} (MA{window_size})')

    if sparsity_scheduler_final_epoch is not None:
        ax.axvline(x=sparsity_scheduler_final_epoch, color='red', linestyle='--', linewidth=1.5)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Component Losses')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', ncol=2, fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.97] if suptitle else None)
    plt.show()


def main():
    import os

    from paths import ROOT_PROJECT_PATH
    from data.utils_io import load_json

    # Data directories
    local_storage_dir = os.path.join(ROOT_PROJECT_PATH, "local_storage")
    base_data_dir = os.path.join(local_storage_dir, 'data', '2025-10-25_11-45-48')
    run_dir = os.path.join(local_storage_dir, 'runs', '2025-10-25_21-52-56')

    train_loss = load_json(filename='train_loss.json',
                           save_dir=run_dir)
    val_loss = load_json(filename='val_loss.json',
                         save_dir=run_dir)

    plot_train_val_losses(train_losses_history=train_loss,
                          val_losses_history=val_loss,
                          window_size=50,
                          figsize=(12, 8),
                          suptitle='Reconstruction Model Training Loss')
    pass


if __name__ == "__main__":
    main()
