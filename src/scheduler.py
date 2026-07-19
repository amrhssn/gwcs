def ste_temperature_at_epoch(
        epoch: int,
        mode: str,
        temperature_init: float,
        temperature_final: float,
        anneal_epochs: int,
) -> float:
  if mode == 'fixed':
    return temperature_init
  progress = min(1.0, epoch / anneal_epochs)
  return temperature_init * (temperature_final / temperature_init) ** progress
