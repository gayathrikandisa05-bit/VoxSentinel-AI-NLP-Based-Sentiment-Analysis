from pathlib import Path

import yaml


def test_expected_directories_exist():
    root = Path(__file__).resolve().parents[1]
    required = ["app/core", "app/data", "app/models", "app/services", "app/api", "app/dashboard", "configs", "tests", "docs"]
    assert all((root / path).is_dir() for path in required)


def test_app_config_is_valid():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "app_config.yaml").read_text(encoding="utf-8"))
    weights = config["cps"]["weights"]
    assert abs(sum(weights.values()) - 1.0) < 1e-9
