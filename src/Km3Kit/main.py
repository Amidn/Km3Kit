
from utils.yml_utils import readConfigs

# Read the configuration
Configs = readConfigs()  # Should return a dictionary

print("Raw configs:", Configs)  # To inspect the parsed data
print(type(Configs))  # To confirm the data type
# Access the desired value
saving_dir = Configs["FileConfig"]["Saving_Dir"]
print(f"Saving directory: {saving_dir}")





print ("------------------")
from Km3Kit import  BRANCHES_YML, DATASET_REGISTRY_YML
from Km3Kit import pd_dataFrame
'''
# Load the data from the ROOT files listed in the dataset_registry.yml
df_data = pd_dataFrame(
    dataset_name="arca21_bdt",
    branches_config_path="config/branches.yml",
    data_type="data",
    verbose=True
)

# Display the first few rows of the resulting DataFrame
print(df_data.head())'''


