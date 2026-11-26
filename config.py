from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


APP_CONFIG = load_yaml(ROOT_DIR / "configs" / "app_config.yaml")
MODEL_CONFIG = load_yaml(ROOT_DIR / "configs" / "model_config.yaml")
