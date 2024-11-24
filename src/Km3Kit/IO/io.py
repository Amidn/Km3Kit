
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