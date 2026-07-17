#!/bin/bash
# Copyright 2020 Deepmind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Usage:
#     sh download_dataset.sh ${DATASET_NAME} ${OUTPUT_DIR}
# Example:
#     sh download_dataset.sh flag_simple /tmp/

# Datasets can be downloaded using the script download_dataset.sh. They contain a metadata file describing
# the available fields and their shape, and tfrecord datasets for train, valid and test splits.
# Dataset names match the naming in the paper. The following datasets are available:
# airfoil
# cylinder_flow
# deforming_plate
# flag_minimal
# flag_simple
# flag_dynamic
# flag_dynamic_sizing
# sphere_simple
# sphere_dynamic
# sphere_dynamic_sizing

# flag_minimal is a truncated version of flag_simple, and is only used for integration tests.
# flag_dynamic_sizing and sphere_dynamic_sizing can be used to learn the sizing field.
# These datasets have the same structure as the other datasets, but contain the meshes in their
# state before remeshing, and define a matching sizing_field target for each mesh.

set -e

DATASET_NAME="${1}"
OUTPUT_DIR="${2}/${DATASET_NAME}"

BASE_URL="https://storage.googleapis.com/dm-meshgraphnets/${DATASET_NAME}/"

mkdir -p ${OUTPUT_DIR}
for file in meta.json train.tfrecord valid.tfrecord test.tfrecord
do
curl -L -C - "${BASE_URL}${file}" -o "${OUTPUT_DIR}/${file}"
done