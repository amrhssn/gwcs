import os

from the_well.data import WellDataset
import the_well.data.normalization

raw_dataset_dir = '/workspace/local_storage/datasets/turbulent_radiative_layer_2D'

well_base_path = os.path.dirname(raw_dataset_dir)
dataset = WellDataset(
    well_base_path=well_base_path,
    well_dataset_name='turbulent_radiative_layer_2D',
    well_split_name='train',
    n_steps_input=1,
    n_steps_output=0,
    use_normalization=True,
)

pass