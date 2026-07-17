from pathlib import Path

# This paths.py lives at: <project_root>/scripts/turbulent_radiative_layer_2D/paths.py
# parents[2] traverses: turbulent_radiative_layer_2D → scripts → <project_root>
ROOT_PROJECT_PATH = Path(__file__).resolve().parents[2]

print(f"ROOT_PROJECT_PATH resolved to: {ROOT_PROJECT_PATH}")