import os
import pandas as pd
from Km3Kit.utils.yml_utils import readConfigs


def load_saved_files(dataset_name, verbose=True):
    """
    Loads previously saved HDF5 files (data, muon, neutrino) into Pandas DataFrames.

    Args:
        dataset_name (str): Name of the dataset to load (e.g., "arca21").
        saving_dir (str): Directory where the HDF5 files are saved.
        verbose (bool): If True, enables detailed logging.

    Returns:
        dict: A dictionary containing the DataFrames for 'data', 'muon', and 'neutrino'.
    """
    if verbose:
        print("Reading configuration for saving directories...")
    configs = readConfigs(verbose=verbose)
        
        # Validate configuration keys
    try:
        saving_dir = configs["FileConfig"]["Saving_Dir"]
    except KeyError as e:
        raise KeyError(f"Missing configuration key: {e}. Check 'FileConfig' in your config YAML file.")

    # Ensure saving directory exists
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)
        if verbose:
            print(f"Created directory: {saving_dir}")



    # Construct file paths
    data_h5 = os.path.join(saving_dir, f"{dataset_name}_Data.h5")
    muon_h5 = os.path.join(saving_dir, f"{dataset_name}_Muon.h5")            
    neutrino_h5 = os.path.join(saving_dir, f"{dataset_name}_Neutrino.h5")

    # Check if files exist
    if verbose:
        print(f"Looking for saved files in {saving_dir}...")

    for file in [data_h5, muon_h5, neutrino_h5]:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Expected file not found: {file}")

    if verbose:
        print("All files found. Loading into DataFrames...")

    # Load the HDF5 files into Pandas DataFrames
    df_data = pd.read_hdf(data_h5, key="data")
    df_muon = pd.read_hdf(muon_h5, key="muon")
    df_neutrino = pd.read_hdf(neutrino_h5, key="neutrino")

    if verbose:
        print("Files successfully loaded into DataFrames.")

    # Return DataFrames in a dictionary
    return {"data": df_data, "muon": df_muon, "neutrino": df_neutrino}