
from IO.yml_Loader import Loader
from IO.rootio import readroot
from .yml_Loader import Loader

def Read(input, verbose=True):
    loader_instance = Loader.readYML(input, verbose=verbose)
    if loader_instance is None:
        raise ValueError(f"Dataset '{input}' not found or could not be loaded.")
    
    # Retrieve data, neutrino, and muon files
    data_files, neutrino_files, muon_files = loader_instance.read_paths(input)
    type_ = Loader.get_data_type(input)
    
    # Verbose output for debugging
    if verbose:
        print("Input:", input)
        print("Data files:", data_files)
        print("Neutrino files:", neutrino_files)
        print("Muon files:", muon_files)
        print("Data type:", type_)
    
    # Check if the data type is ROOT and proceed with reading
    if type_ == "ROOT":
        DATA_ = readroot(data_files)
        NEUTRINO_ = readroot(neutrino_files)
        MUON_ = readroot(muon_files)
        
        # Print the order of outputs if verbose is enabled
        if verbose:
            print("ORDER: DATA, NEUTRINO, MUON")
        
        return DATA_, NEUTRINO_, MUON_
    