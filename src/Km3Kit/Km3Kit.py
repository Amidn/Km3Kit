from Km3Kit import DF_, BRANCHES_YML, DATASET_REGISTRY_YML

print(BRANCHES_YML)  # Output: /path/to/Km3Kit/config/branches.yml

# Use DF_ to load a DataFrame
df = DF_(dataset_name="arca21_bdt", branches_config_path=BRANCHES_YML, data_type="data", verbose=True)
print(df.head())