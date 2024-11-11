
from IO.yml_Loader import Loader
from IO.rootio import readroot


def Read(input, verbose=False):
    # Retrieve data, neutrino, and muon files
    data_files, neutrino_files, muon_files = Loader.read_paths(input)
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
    