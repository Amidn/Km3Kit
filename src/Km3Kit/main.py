


import os
import pandas as pd
from Km3Kit.IO.fitsio import create_fits_file
#from Km3Kit import process_dfs, get_dataset_version

df_data = pd.read_hdf("/sps/km3net/users/amid/DataSets/Km3KitProductions/arca21_bdt_converted_Data.h5", key="data")
print(df_data.columns)
print(df_data.head())
create_fits_file("arca21_bdt" ,"config/fits_configs.yml", df_data, "/sps/km3net/users/amid/DataSets/Km3KitProductions/")
print ("Done")
#f = get_dataset_version("arca21_bdt", "MMAA")
#print (f)


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




