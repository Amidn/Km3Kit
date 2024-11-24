
import astropy.units as u
from astropy.coordinates import SkyCoord
from gammapy.data import EventList
from .rootio import load_dst, pd_dataFrame
from ..utils.yml_utils import Loader, load_branches_config

def NEWFUNC(dataset_name=None, branches_config_path="config/branches.yml", save_dir = None ,  verbose=True):
    df_data = pd_dataFrame( dataset_name, branches_config_path , data_type="data",  verbose=True)
    df_muon = pd_dataFrame( dataset_name, branches_config_path , data_type="muon",  verbose=True)
    df_neutrino = pd_dataFrame( dataset_name, branches_config_path , data_type="neutrino",  verbose=True)






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
    
        
