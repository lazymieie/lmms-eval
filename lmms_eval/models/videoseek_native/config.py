from pathlib import Path

import yaml

from . import VENDORED_UPSTREAM_COMMIT


PACKAGE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = PACKAGE_DIR / "config"


def _load_yaml(name: str) -> dict:
    with open(CONFIG_DIR / name, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


general_config = _load_yaml("general.yaml")
prompts_config = _load_yaml("prompts.yaml")


def build_agent_config(overrides: dict | None = None) -> dict:
    config = dict(general_config)
    config.update(prompts_config)
    if overrides:
        config.update({key: value for key, value in overrides.items() if value is not None})
    config["vendored_upstream_commit"] = VENDORED_UPSTREAM_COMMIT
    return config
