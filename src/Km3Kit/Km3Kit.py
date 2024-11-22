from utils.yaml_utils import Loader, load_branches_config

# Load dataset configuration
dataset_name = "arca21_bdt"
loader_instance = Loader.readYML(dataset_name, verbose=True)

# Retrieve paths for the dataset
data_dict, muon_dict, neutrino_dict = loader_instance.read_paths(verbose=True)

# Load branch configurations
branches = load_branches_config("config/branches.yml")
E_branches = branches["E_branches"]
T_branches = branches["T_branches"]

# Print results
print("E_branches:", E_branches)
print("T_branches:", T_branches)
print("Data Paths:", data_dict["data"])