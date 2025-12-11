from pathlib import Path

DATASET_KAGGLEHUB_PATH = "emmarex/plantdisease/versions/1"
DATASET_PATH = (
    Path.home()
    / Path(".cache/kagglehub/datasets")
    / DATASET_KAGGLEHUB_PATH
    / "PlantVillage"
)
