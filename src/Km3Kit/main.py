from Km3Kit import  BRANCHES_YML, DATASET_REGISTRY_YML

print(BRANCHES_YML)  # Output: /path/to/Km3Kit/config/branches.yml

# Import the function
from utils.yml_utils import readConfigs

# Read the configuration
configs = readConfigs()

# Access specific configuration values
saving_dir = configs["FileConfig"]["Saveing_Dir"]

print(f"The saving directory is: {saving_dir}")

print ("------------------")
from Km3Kit import pd_dataFrame

# Load the data from the ROOT files listed in the dataset_registry.yml
df_data = pd_dataFrame(
    dataset_name="arca21_bdt",
    branches_config_path="config/branches.yml",
    data_type="data",
    verbose=True
)

# Display the first few rows of the resulting DataFrame
print(df_data.head())

