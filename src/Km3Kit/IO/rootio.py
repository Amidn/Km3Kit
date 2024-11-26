import uproot
import awkward as ak
import numpy as np
import pandas as pd
import time
from Km3Kit.utils.tools import report_time_interval, report_memory_usage
from ..utils.yml_utils import Loader, load_branches_config


def load_dst(E_branches, T_branches, file_paths, verbose=False):
    E_tree = "E"
    T_tree = "T"

    # Initialize a variable to track maximum memory usage
    max_memory = 0

    start = time.time()
    max_memory = report_memory_usage("initial state", max_memory, verbose)
    if verbose:
        report_time_interval(start, "initial state - time elapsed", verbose)

    # Initialize an empty DataFrame
    DF_MC = pd.DataFrame()

    ################################################################
    for branch in E_branches:
        E_array = uproot.dask(
            {file_path: E_tree for file_path in file_paths},
            filter_name=branch,
            library="ak",
        )
        max_memory = report_memory_usage(f"E_array ({branch}) loaded", max_memory, verbose)
        if verbose:
            report_time_interval(start, f"E_array ({branch}) loaded - time elapsed", verbose)

        E_data = E_array.compute()
        max_memory = report_memory_usage(f"E_data ({branch}) computed", max_memory, verbose)
        if verbose:
            report_time_interval(start, f"E_data ({branch}) computed - time elapsed", verbose)

        # Extract the last part of the branch name
        column_name = branch.split("/")[-1]

        # Add current branch to DataFrame
        DF_MC[column_name] = E_data[column_name].to_list()
        max_memory = report_memory_usage(f"E_tree ({branch}) added to DataFrame", max_memory, verbose)
        if verbose:
            report_time_interval(start, f"E_tree ({branch}) added to DataFrame - time elapsed", verbose)

        # Clear E_data to free memory
        del E_data

    ##################################################################
    # Load and process `T_tree` incrementally
    for branch in T_branches:
        T_array = uproot.dask(
            {file_path: T_tree for file_path in file_paths},
            filter_name=branch,
            library="ak",
        )
        max_memory = report_memory_usage(f"T_array ({branch}) loaded", max_memory, verbose)
        if verbose:
            report_time_interval(start, f"T_array ({branch}) loaded - time elapsed", verbose)

        T_data = T_array.compute()
        max_memory = report_memory_usage(f"T_data ({branch}) computed", max_memory, verbose)
        if verbose:
            report_time_interval(start, f"T_data ({branch}) computed - time elapsed", verbose)

        # Extract the last part of the branch name
        column_name = branch.split("/")[-1]

        # Add current branch to DataFrame
        DF_MC[column_name] = T_data[column_name].to_list()
        max_memory = report_memory_usage(f"T_tree ({branch}) added to DataFrame", max_memory, verbose)
        if verbose:
            report_time_interval(start, f"T_tree ({branch}) added to DataFrame - time elapsed", verbose)

        # Clear T_data to free memory
        del T_data

    ######################################################
    max_memory = report_memory_usage("All trees added to DataFrame", max_memory, verbose)
    if verbose:
        report_time_interval(start, "All trees added to DataFrame - time elapsed", verbose)

    if verbose:
        # Print the first 10 rows
        print(DF_MC.head(10))

    # Report final process time (always)
    report_time_interval(start, "total process time", True)

    # Report maximum memory usage
    if verbose:
        print(f">>>>>>>>>>>>>>>>>>>> Maximum memory usage during process: {max_memory:.2f} MB")

    return DF_MC


def pd_dataFrame(dataset_name="arca21_bdt", branches_config_path="config/branches.yml", data_type="data", verbose=False):
    """
    Loads data into a Pandas DataFrame based on the dataset name, branch configuration, and data type.

    Args:
        dataset_name (str): Name of the dataset to load from the dataset registry.
        branches_config_path (str): Path to the branches configuration YAML file.
        data_type (str): Type of data to load ("data", "muon", or "neutrino").
        verbose (bool): Whether to enable verbose logging.

    Returns:
        pd.DataFrame: The resulting DataFrame with the loaded data.
    """
    # Step 1: Load branch configurations
    if verbose:
        print(f"Loading branch configurations from {branches_config_path}...")
    branches = load_branches_config(branches_config_path)
    E_branches = branches["E_branches"]
    T_branches = branches["T_branches"]

    if verbose:
        print("Branch structures loaded:")
        print(f"E_branches: {E_branches}")
        print(f"T_branches: {T_branches}")

    # Step 2: Load dataset paths
    if verbose:
        print(f"Loading dataset registry for dataset: {dataset_name}...")
    loader_instance = Loader.readYML(dataset_name, verbose=verbose)
    if not loader_instance:
        raise ValueError(f"Dataset '{dataset_name}' not found in the registry.")

    # Retrieve paths
    data_dict, muon_dict, neutrino_dict = loader_instance.read_paths(verbose=verbose)

    # Step 3: Select file paths based on data type
    if data_type == "data":
        file_paths = data_dict["data"].split("\n")
    elif data_type == "muon":
        file_paths = muon_dict["muon"].split("\n")
    elif data_type == "neutrino":
        file_paths = neutrino_dict["neutrino"].split("\n")
    else:
        raise ValueError(f"Invalid data type '{data_type}'. Must be 'data', 'muon', or 'neutrino'.")

    if verbose:
        print(f"File paths to be processed for '{data_type}':")
        print(file_paths)

    # Step 4: Call load_dst to load data into a DataFrame
    df = load_dst(E_branches, T_branches, file_paths, verbose=verbose)

    if verbose:
        print(f"DataFrame successfully created for data type '{data_type}'. Shape: {df.shape}")

    return df

