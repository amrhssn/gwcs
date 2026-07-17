import random
from typing import Tuple, List, Dict, Any


def split_train_val_test(
        graph_signal_list: List[Dict[str, Any]],
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        shuffle: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    assert train_ratio + val_ratio + test_ratio == 1.0
    if shuffle:
        random.shuffle(graph_signal_list)
    num_samples = len(graph_signal_list)

    train_last_index = int(train_ratio * num_samples)
    val_last_index = train_last_index + int(val_ratio * num_samples)

    train_graph_signal_list = graph_signal_list[:train_last_index]
    val_graph_signal_list = graph_signal_list[train_last_index:val_last_index]
    test_graph_signal_list = graph_signal_list[val_last_index:]
    return train_graph_signal_list, val_graph_signal_list, test_graph_signal_list
