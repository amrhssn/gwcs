from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from configs.common import GraphWaveletTransformConfig
from steps.turbulent_preprocessing import preprocess_turbulent_split

from tests.data.turbulent.conftest import HEIGHT, WIDTH, density_value


def test_batch_ids_identify_source_trajectory_and_timestep(
        tiny_raw_split_dir: Path,
        tmp_path: Path,
        gwt_config: GraphWaveletTransformConfig,
) -> None:
  dataset = preprocess_turbulent_split(raw_split_dir=tiny_raw_split_dir, gwt_config=gwt_config,
                                       samples_dir=tmp_path / "samples", height=HEIGHT, width=WIDTH)
  loader = DataLoader(dataset=dataset, batch_size=7, shuffle=True,
                      generator=torch.Generator().manual_seed(0))

  seen_keys = set()
  for batch in loader:
    for position in range(batch.num_graphs):
      example = batch.get_example(position)
      tcool = float(example.tcool)
      trajectory_id = int(example.trajectory_id)
      timestep = int(example.timestep)
      # node 0 is (h=0, w=0), so density_value(trajectory_id, timestep, 0, 0) is a direct function
      # of exactly the ids being checked here -- if PyG batching corrupted or misaligned these
      # per-sample ids relative to x (e.g. via the __inc__ node-offset heuristic silently triggering
      # on a name like "time_index"), this recomputed value would no longer match x.
      expected_density_at_node0 = density_value(trajectory_id, timestep, 0, 0)
      assert example.x[0, 0].item() == expected_density_at_node0, (
        f"tcool={tcool} trajectory_id={trajectory_id} timestep={timestep}: "
        f"x does not match the ids carried alongside it through batching"
      )
      seen_keys.add((tcool, trajectory_id, timestep))

  # every (tcool, trajectory, timestep) combination in the raw data must appear exactly once,
  # confirming ids survive shuffled multi-batch DataLoader iteration without loss or duplication.
  assert len(seen_keys) == len(dataset)
