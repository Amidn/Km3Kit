
import astropy.units as u
from astropy.coordinates import SkyCoord
from gammapy.data import EventList
from .rootio import load_dst, pd_dataFrame
from ..utils.yml_utils import Loader, load_branches_config
from utils.yml_utils import readConfigs

import os
from .rootio import pd_dataFrame
from ..utils.yml_utils import readConfigs

def process_dfs(dataset_name, branches_config_path="config/branches.yml", save_pd=None, verbose=True):
    """
    Loads data, muon, and neutrino datasets into Pandas DataFrames and optionally saves them as HDF5 files.

    Args:
        dataset_name (str): Name of the dataset in the registry.
        branches_config_path (str): Path to the branches configuration YAML file.
        save_pd (bool): If True, saves the DataFrames as HDF5 files.
        verbose (bool): If True, enables detailed logging.

    Returns:
        dict: A dictionary containing the DataFrames for 'data', 'muon', and 'neutrino'.
    """
    # Load the DataFrames
    if verbose:
        print("Loading data, muon, and neutrino datasets...")
    df_data = pd_dataFrame(dataset_name, branches_config_path, data_type="data", verbose=verbose)
    df_muon = pd_dataFrame(dataset_name, branches_config_path, data_type="muon", verbose=verbose)
    df_neutrino = pd_dataFrame(dataset_name, branches_config_path, data_type="neutrino", verbose=verbose)

    if save_pd:
        # Read configurations for saving directory
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

        # Construct file paths for saving
        data_h5 = os.path.join(saving_dir, "Data.h5")
        muon_h5 = os.path.join(saving_dir, "Muon.h5")
        neutrino_h5 = os.path.join(saving_dir, "Neutrino.h5")

        # Save DataFrames to HDF5
        if verbose:
            print(f"Saving data to {data_h5}, {muon_h5}, and {neutrino_h5}...")
        df_data.to_hdf(data_h5, key="data", mode="w")
        df_muon.to_hdf(muon_h5, key="data", mode="w")
        df_neutrino.to_hdf(neutrino_h5, key="data", mode="w")

        if verbose:
            print("Files successfully saved.")

        # Add the new dataset information to the registry
        if verbose:
            print("Updating the dataset registry...")
        addDataSetToRegistry(
            name=f"{dataset_name}_converted_2pd",
            data_type="HDF5",
            comment="Converted dataset into HDF5 format.",
            directory=saving_dir,
            data_name="Data.h5",
            muon_name="Muon.h5",
            neutrino_name="Neutrino.h5",
            verbose=verbose
        )

    # Return DataFrames for further processing
    return {"data": df_data, "muon": df_muon, "neutrino": df_neutrino}



def read(dataset_name, usage_tag, recreate=False, verbose=False, yml_data_registry_path="config/dataset_registry.yml"):
    """
    Chooses a dataset based on the given name and usage tag.
    Searches the YAML registry for preprocessed versions (Pandas or FITS).

    Args:
        dataset_name (str): The name of the dataset (e.g., "arca21").
        usage_tag (str): The intended usage ("GammaPy" or "KM3Net").
        recreate (bool): If True, forces the use of the original dataset without checking conversions.
        verbose (bool): If True, prints detailed logs.
        yml_data_registry_path (str): Path to the YAML registry file.

    Returns:
        str: Name of the chosen dataset.
    """
    # If recreate is True, return the original dataset name
    if recreate:
        if verbose:
            print(f"Recreate is True. Returning original dataset: {dataset_name}")
        return dataset_name

    # Load the dataset names and types from the YAML registry
    datasets = Loader.list_datasets_and_types(yml_registry_path=yml_data_registry_path, verbose=verbose)

    # Build the names of the preprocessed datasets
    dataset_name_fits = dataset_name + "_converted_2Fits"
    dataset_name_pd = dataset_name + "_converted_2pd"

    # Check the usage tag and look for the appropriate preprocessed dataset
    if usage_tag == "GammaPy":
        for name, _ in datasets:
            if name == dataset_name_fits:
                if verbose:
                    print(f"Found preprocessed FITS dataset: {dataset_name_fits}")
                return dataset_name_fits
        if verbose:
            print(f"Preprocessed FITS dataset not found. Returning original dataset: {dataset_name}")
        return dataset_name

    elif usage_tag == "KM3Net":
        for name, _ in datasets:
            if name == dataset_name_pd:
                if verbose:
                    print(f"Found preprocessed Pandas dataset: {dataset_name_pd}")
                return dataset_name_pd
        if verbose:
            print(f"Preprocessed Pandas dataset not found. Returning original dataset: {dataset_name}")
        return dataset_name

    else:
        raise ValueError(f"Invalid usage_tag: {usage_tag}. Must be 'GammaPy' or 'KM3Net'.")
    
        
