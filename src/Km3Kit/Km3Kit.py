from io.io import DF_

# Example usage
dataset_name = "arca21_bdt"
branches_config_path = "config/branches.yml"

# Load data (e.g., "data", "muon", or "nu")
df = DF_(dataset_name, branches_config_path, data_type="data", verbose=True)

# Display the DataFrame
print("Loaded DataFrame:")
print(df.head())