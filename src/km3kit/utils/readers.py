from datetime import datetime
import pandas as pd
import yaml
import tables
import argparse
from astropy.coordinates import SkyCoord, Galactic
import astropy.units as u
import numpy as np
import os
import ROOT
import aa


class RooTLoader:
    def __init__(self, name=None, date_added=None, data_type=None, comment=None, paths=None):
        self.name = name
        self.date_added = date_added
        self.data_type = data_type
        self.comment = comment
        self.paths = paths
    @classmethod
    def read_dataset_info(cls, file_path, dataset_name, verbose=False):
        """Reads the YAML file and returns the details of the requested dataset."""
        try:
            with open(file_path, 'r') as file:
                data = yaml.safe_load(file)
                for entry in data:
                    if entry['name'] == dataset_name:
                        dataset_info = cls(
                            name=entry['name'],
                            date_added=entry['date_added'],
                            data_type=entry['data_type'],
                            comment=entry['comment'],
                            paths=entry['path']  # Adjust if no trailing space
                        )
                        if verbose:
                            dataset_info.print_dataset_info()
                        return dataset_info
        except FileNotFoundError:
            print(f"The file {file_path} does not exist.")
        except yaml.YAMLError as exc:
            print(f"Error reading YAML file: {exc}")

        print(f"Dataset with name '{dataset_name}' not found.")
        return None
    
    def print_dataset_info(self):
        """Prints detailed information of a dataset."""
        print("Detailed information of a dataset:")
        print("Name:", self.name)
        print("Date Added:", self.date_added)
        print("Data Type:", self.data_type)
        print("Comment:", self.comment)
        print("Paths:")
        for data_type, paths_list in self.paths.items():
            print(" ", data_type, ":")
            for path in paths_list:
                print("   -", path)
        print("End of information of a dataset")


    def list_all_paths(self, verbose=False):
        """Prints all key-value pairs for 'neutrino', 'muon', and 'data' paths."""
        categories = {'neutrino': 'allnufiles', 'muon': 'allmufiles', 'data': 'alldatafiles'}
        lists = {}

        for category, var_name in categories.items():
            paths = self.paths.get(category, [])
            file_list = f"{var_name}={{\n "
            for item in paths:
                name, path = next(iter(item.items()))
                channel = f"\"{name}\" : ROOT.EventFile(\"{path}\")"
                file_list += f"{channel}, \n"
            file_list += "}\n"
            lists[var_name] = file_list
            if verbose:
                print(file_list)

        return lists['allnufiles'], lists['allmufiles'], lists['alldatafiles']
    
    def read(self):
        allnufiles, allmufiles, alldatafiles = self.list_all_paths()
        print (allnufiles, allmufiles, alldatafiles)
        nu_r = exec(allnufiles)
        mu_r = exec(allmufiles)
        da_r = exec(alldatafiles)
        return da_r, nu_r, mu_r

        
     
    
     




class h5Loader:
    def __init__(self, name=None, date_added=None, data_type=None, comment=None, paths=None):
        self.name = name
        self.date_added = date_added
        self.data_type = data_type
        self.comment = comment
        self.paths = paths

    @classmethod
    def read_dataset_info(cls, file_path, dataset_name, verbose=False):
        """Reads the YAML file and returns the details of the requested dataset."""
        try:
            with open(file_path, 'r') as file:
                data = yaml.safe_load(file)
                for entry in data:
                    if entry['name'] == dataset_name:
                        dataset_info = cls(
                            name=entry['name'],
                            date_added=entry['date_added'],
                            data_type=entry['data_type'],
                            comment=entry['comment'],
                            paths=entry['path']  # Adjust if no trailing space
                        )
                        if verbose:
                            dataset_info.print_dataset_info()
                        return dataset_info
        except FileNotFoundError:
            print(f"The file {file_path} does not exist.")
        except yaml.YAMLError as exc:
            print(f"Error reading YAML file: {exc}")

        print(f"Dataset with name '{dataset_name}' not found.")
        return None

    def print_dataset_info(self):
        """Prints detailed information of a dataset."""
        print("Detailed information of a dataset:")
        print("Name:", self.name)
        print("Date Added:", self.date_added)
        print("Data Type:", self.data_type)
        print("Comment:", self.comment)
        print("Paths:")
        for data_type, paths_list in self.paths.items():
            print(" ", data_type, ":")
            for path in paths_list:
                print("   -", path)
        print("End of information of a dataset")

        
    
    
def Sort(df_files, remove_id=None):
    """
    Concatenates multiple HDF5 files into a single DataFrame, optionally removing rows with a specific identifier.

    Parameters:
    - df_files (list of str): The file paths of HDF5 files to concatenate.
    - remove_id (int or float, optional): The identifier of runs to remove from the DataFrame. Defaults to None.

    Returns:
    - pandas.DataFrame: The concatenated DataFrame, possibly with specified runs removed.
    """
    
    # Ensure df_files is a list or a similar iterable
    if not isinstance(df_files, (list, tuple)):
        raise ValueError("df_files should be a list or tuple of file paths.")
    
    # Generator expression to read all DataFrames
    all_Dfs = (pd.read_hdf(file) for file in df_files)
    
    # Concatenate all DataFrames into one
    DF = pd.concat(all_Dfs, ignore_index=True, sort=False)
    
    # Remove rows where 'run_id' matches 'remove_id', if specified
    if remove_id is not None and isinstance(remove_id, (int, float)):
        removed_count = DF[DF['run_id'] == remove_id].shape[0]
        DF = DF[DF['run_id'] != remove_id]
        if removed_count > 0:
            print(f" > >> >>>Run {remove_id} is removed ({removed_count} rows).")
    
    return DF
 
    
    
