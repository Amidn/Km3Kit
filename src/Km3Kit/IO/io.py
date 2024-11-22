
from Km3Kit.utils.yml_Loader import Loader
from IO.rootio import readroot
from ..utils.yml_Loader import Loader
import time 
from utils.tools import report_time_interval as timeto
from rootio import load_dst
from config.config_loader import load_branches_config
# Load branches using the default path
config = load_branches_config()

E_branches = config["E_branches"]
T_branches = config["T_branches"]

# Assuming you have an instance of Loader for a specific dataset
loader_instance = Loader.readYML("arca21_bdt", verbose=True)

# Retrieve paths as dictionaries
data_dict, muon_dict, neutrino_dict = loader_instance.read_paths(verbose=True)

def dataframe(df_, E_branches, T_branches ):
     # Pass these to your function
    DF_data = load_dst(E_branches, T_branches, df_, verbose=True)
    return DF_data


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