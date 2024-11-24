
from .rootio import load_dst
from ..utils.yml_utils import Loader, load_branches_config


def DF_(dataset_name="arca21_bdt", branches_config_path="config/branches.yml", data_type="data", verbose=False):
    """
    Loads data into a Pandas DataFrame based on the dataset name, branch configuration, and data type.

    Args:
        dataset_name (str): Name of the dataset to load from the dataset registry.
        branches_config_path (str): Path to the branches configuration YAML file.
        data_type (str): Type of data to load ("data", "muon", or "nu").
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
    elif data_type == "nu":
        file_paths = neutrino_dict["neutrino"].split("\n")
    else:
        raise ValueError(f"Invalid data type '{data_type}'. Must be 'data', 'muon', or 'nu'.")

    if verbose:
        print(f"File paths to be processed for '{data_type}':")
        print(file_paths)

    # Step 4: Call load_dst to load data into a DataFrame
    df = load_dst(E_branches, T_branches, file_paths, verbose=verbose)

    if verbose:
        print(f"DataFrame successfully created for data type '{data_type}'. Shape: {df.shape}")

    return df




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