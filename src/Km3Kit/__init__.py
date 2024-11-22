"""Top-level package for Km3Kit."""

__author__ = """Amid Nayerhoda"""
__email__ = 'anayerhoda@km3net.de'
__version__ = '0.0.1'


import os

__all__ = [
    "DF_",
    "load_dst",
    "Loader",
    "load_branches_config",
    "report_time_interval",
    "report_memory_usage",
    "BRANCHES_YML",
    "DATASET_REGISTRY_YML",
]

# Exposing data-loading and processing functions
from .IO.io import DF_
from .IO.rootio import load_dst

# Exposing YAML utility functions and classes
from .utils.yaml_utils import Loader, load_branches_config

# Exposing performance monitoring tools
from .utils.tools import report_time_interval, report_memory_usage

# Paths to YAML configuration files
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")
BRANCHES_YML = os.path.join(CONFIG_DIR, "branches.yml")  # Branch structure configuration
DATASET_REGISTRY_YML = os.path.join(CONFIG_DIR, "dataset_registry.yml")  # Dataset registry configuration

# Optional: Validate that configuration files exist
for path in [BRANCHES_YML, DATASET_REGISTRY_YML]:
    if not os.path.exists(path):
        print(f"Warning: Configuration file not found: {path}")