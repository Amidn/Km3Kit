
import astropy.units as u
import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
from gammapy.data import EventList
from Km3Kit.IO.fitsio import create_fits_file
from Km3Kit.IO.hdf5io import load_saved_files
from .rootio import load_dst, pd_dataFrame
from Km3Kit.utils.yml_utils import Loader, load_branches_config, readConfigs, add_dataset_to_registry
from utils.tools import  diagnose_dataframe
from astropy.io import fits
import io
import os


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


    if verbose:
        print("Loading  muon,  dataset...")
    df_muon = pd_dataFrame(dataset_name, branches_config_path, data_type="muon", verbose=verbose)

    if save_pd:
        muon_h5 = os.path.join(saving_dir, str(dataset_name) + "_converted_" + "Muon.h5")
        if verbose:
            print(f"Saving data to {muon_h5} ...")
            print("\nDiagnosing 'muon' DataFrame:")
            diagnose_dataframe(df_muon)  # Diagnose the muon DataFrame
        df_muon.to_hdf(muon_h5, key="data", mode="w")


    if verbose:
        print("Loading neutrino dataset...")
    df_neutrino = pd_dataFrame(dataset_name, branches_config_path, data_type="neutrino", verbose=verbose)
    if save_pd:
        neutrino_h5 = os.path.join(saving_dir, str(dataset_name) + "_converted_" + "Neutrino.h5")
        if verbose:
            print(f"Saving data to  {neutrino_h5} ...")
            print("\nDiagnosing 'neutrino' DataFrame:")
            diagnose_dataframe(df_neutrino)  # Diagnose the neutrino DataFrame
        df_neutrino.to_hdf(neutrino_h5, key="data", mode="w")


    if verbose:
        print("Loading data dataset...")
    df_data = pd_dataFrame(dataset_name, branches_config_path, data_type="data", verbose=verbose)
        
    if save_pd:          
        data_h5 = os.path.join(saving_dir,str(dataset_name) + "_converted_" + "Data.h5")
        if verbose:
            print(f"Saving data to {data_h5} ...")
            print("\nDiagnosing 'data' DataFrame:")
            diagnose_dataframe(df_data)  # Diagnose the data DataFrame
        df_data.to_hdf(data_h5, key="data", mode="w")


    if save_pd:
        if verbose:
            print("Files successfully saved.")

        # Add the new dataset information to the registry
        if verbose:
            print("Updating the dataset registry...")

        add_dataset_to_registry(
            name=f"{dataset_name}_converted_2pd",
            data_type="HDF5",
            comment="Converted dataset into HDF5 format.",
            directory_path=saving_dir,
            data_name = data_h5 ,
            muon_name = muon_h5,
            neutrino_name = neutrino_h5,
            verbose=verbose
        )

    # Return DataFrames for further processing
    return {"data": df_data, "muon": df_muon, "neutrino": df_neutrino}




def get_dataset_version(dataset_name, usage_tag, recreate=False, verbose=False, yml_registry_path="config/dataset_registry.yml"):
    """
    Chooses a dataset based on the given name and usage tag.
    Searches the YAML registry for preprocessed versions (Pandas or FITS).

    Args:
        dataset_name (str): The name of the dataset (e.g., "arca21").
        usage_tag (str): The intended usage ("MMAA" or "KM3Net"). MMAA: "Multi-Messenger Astronomy Analysis
        recreate (bool): If True, forces the use of the original dataset without checking conversions.
        verbose (bool): If True, prints detailed logs.
        yml_data_registry_path (str): Path to the YAML registry file.

    Returns:
        str: Name of the chosen dataset.
    """

    # Load the dataset names and types from the YAML registry
    datasets_list = Loader.list_datasets_and_types( verbose = verbose)

    # Build the names of the preprocessed datasets
    dataset_name_fits = dataset_name + "_converted_2Fits"
    dataset_name_pd = dataset_name + "_converted_2pd"
    
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


    
    # If recreate is True, return the original dataset name
    if recreate:
        if verbose:
            print(f"Recreate is True. Returning original dataset: {dataset_name}")
        datasets = process_dfs(dataset_name= dataset_name, save_pd=True, verbose=True) #1
        df_data = datasets["data"]
        create_fits_file("config/fits_config.yml", df_data, saving_dir)


    # Check the usage tag and look for the appropriate preprocessed dataset
    if usage_tag == "MMAA":
        if any(item[0] == dataset_name_fits for item in datasets_list):
            if verbose:
                print(f"Found preprocessed FITS dataset: {dataset_name_fits}")
            return dataset_name_fits
        
        if any(item[0] == dataset_name_pd for item in datasets_list):
            if verbose:
                print(f"Found preprocessed h5 dataset: {dataset_name_pd}")
            datasets = load_saved_files(dataset_name_pd,  verbose=True)
            df_data = datasets["data"]
            create_fits_file("config/fits_config.yml", df_data, saving_dir)
            return dataset_name_fits
        
        if any(item[0] == dataset_name for item in datasets_list):
            datasets = process_dfs(dataset_name= dataset_name, save_pd=True, verbose=True) 
            df_data = datasets["data"]
            create_fits_file(dataset_name ,"config/fits_config.yml", df_data, saving_dir)
            return dataset_name_fits
            

    elif usage_tag == "KM3Net":
        for name, _ in datasets:
            if name == dataset_name_pd:
                if verbose:
                    print(f"Found preprocessed Pandas dataset: {dataset_name_pd}")
                return dataset_name_pd
            else:
                if verbose:
                    print(f"Preprocessed Pandas dataset not found. Converting dataset: {dataset_name} to h5")
                datasets = process_dfs(dataset_name= dataset_name, save_pd=True, verbose=True) #
                return dataset_name_pd

    else:
        raise ValueError(f"Invalid usage_tag: {usage_tag}. Must be 'GammaPy' or 'KM3Net'.")
    
        

