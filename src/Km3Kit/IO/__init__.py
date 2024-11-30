# src/Km3Kit/IO/__init__.py

# Expose key modules or functions for ease of import
from .io import get_dataset_version, process_dfs
from .rootio import load_dst, pd_dataFrame
from .fitsio import create_fits_file

__all__ = ["get_dataset_version", "process_dfs", "load_dst", "pd_dataFrame", "create_fits_file"]