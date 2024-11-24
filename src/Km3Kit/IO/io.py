
import astropy.units as u
from astropy.coordinates import SkyCoord
from gammapy.data import EventList
from .rootio import load_dst
from ..utils.yml_utils import Loader, load_branches_config





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
    
        



"""

def Read(input, verbose=True):
    loader_instance = Loader.readYML(input, verbose=verbose)
    if loader_instance is None:
        raise ValueError(f"Dataset '{input}' not found or could not be loaded.")
    
    # Retrieve data, neutrino, and muon files
    data_files, neutrino_files, muon_files = loader_instance.read_paths(input)
    type_ = loader_instance.get_data_type()
    
    # Verbose output for debugging
    if verbose:
        print("Input:", input)
        print("Data files:", data_files)
        print("Neutrino files:", neutrino_files)
        print("Muon files:", muon_files)
        print("Data type:", type_)
    
    # Check if the data type is ROOT and proceed with reading
    if type_ == "ROOT":

        start_time1 = time.time()
        DATA_ = readroot(data_files)
        if verbose:
            timeto(start_time1, "reading data")

        start_time = time.time()
        NEUTRINO_ = readroot(neutrino_files)
        if verbose:
            timeto(start_time, "reading neutrino")

        start_time = time.time()
        MUON_ = readroot(muon_files)
        if verbose:
            timeto(start_time, "reading muons")
        
        # Print the order of outputs if verbose is enabled
        if verbose:
            print("ORDER: DATA, NEUTRINO, MUON")
            timeto(start_time1, "reading all data")
        return DATA_, NEUTRINO_, MUON_
    
        
        """