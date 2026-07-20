from data.turbulent.graph import build_turbulent_graph

from tests.data.turbulent.conftest import HEIGHT, WIDTH


def test_node_layout_matches_row_major_flatten() -> None:
  graph = build_turbulent_graph(height=HEIGHT, width=WIDTH)
  num_nodes = HEIGHT * WIDTH
  assert graph.metadata.num_nodes == num_nodes

  edge_index = graph.edge_index
  neighbor_sets = [set() for _ in range(num_nodes)]
  for src, dst in edge_index.t().tolist():
    neighbor_sets[src].add(dst)

  for node_id in range(num_nodes):
    h, w = divmod(node_id, WIDTH)
    expected_neighbors = set()
    for dh in (-1, 0, 1):
      for dw in (-1, 0, 1):
        if dh == 0 and dw == 0:
          continue
        nh, nw = h + dh, w + dw
        if 0 <= nh < HEIGHT and 0 <= nw < WIDTH:
          expected_neighbors.add(nh * WIDTH + nw)
    assert neighbor_sets[node_id] == expected_neighbors, (
      f"node {node_id} (h={h}, w={w}): expected row-major (8-connected Moore) grid neighbors "
      f"{expected_neighbors}, got {neighbor_sets[node_id]} -- node layout does not match "
      f"row-major (h*width+w) flattening"
    )
