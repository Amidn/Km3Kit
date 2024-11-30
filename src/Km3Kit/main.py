
from Km3Kit import process_dfs, get_dataset_version


f = get_dataset_version("arca21_bdt", "MMAA")
print (f)


"""
# Load datasets and save them
datasets = process_dfs(dataset_name="arca21_bdt", save_pd=True, verbose=True)

# Access individual DataFrames
df_data = datasets["data"]
df_muon = datasets["muon"]
df_neutrino = datasets["neutrino"]

# Print the first few rows of each
print("DataFrame (Data):")
print(df_data.head())

print("DataFrame (Muon):")
print(df_muon.head())

print("DataFrame (Neutrino):")
print(df_neutrino.head())"""