"""Load and merge pipeline configuration from config.yaml + optional overrides."""

import pathlib
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class Config:
    paths:             dict = field(default_factory=dict)
    reference_probes:  list = field(default_factory=list)
    alphafold_targets: dict = field(default_factory=dict)
    filter:            dict = field(default_factory=dict)
    hmmer:             dict = field(default_factory=dict)
    embedding:         dict = field(default_factory=dict)
    clustering:        dict = field(default_factory=dict)
    subclustering:     dict = field(default_factory=dict)
    tree:              dict = field(default_factory=dict)
    structure:         dict = field(default_factory=dict)
    catalytic_motifs:  dict = field(default_factory=dict)

    # Derived path helpers
    @property
    def data_dir(self) -> pathlib.Path:
        return pathlib.Path(self.paths.get("data_dir", "data"))

    @property
    def results_dir(self) -> pathlib.Path:
        return pathlib.Path(self.paths.get("results_dir", "results"))

    def probe_colours(self) -> dict[str, str]:
        return {p["id"]: p["colour"] for p in self.reference_probes}

    def probe_labels(self) -> dict[str, str]:
        return {p["id"]: p["label"] for p in self.reference_probes}


def load_config(path: str | pathlib.Path = "config.yaml",
                overrides: dict[str, Any] | None = None) -> Config:
    """Load config.yaml, apply optional dict overrides, return Config object."""
    with open(path) as f:
        data = yaml.safe_load(f)
    if overrides:
        _deep_merge(data, overrides)
    return Config(**{k: v for k, v in data.items() if k in Config.__dataclass_fields__})


def _deep_merge(base: dict, override: dict) -> None:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
