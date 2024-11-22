import uproot
import awkward as ak
import numpy as np
import pandas as pd
import time
from .tools import report_time_interval, report_memory_usage




def load_dst(E_branches, T_branches, file_paths, verbose=False):
    E_tree = "E"
    T_tree = "T"

    # Initialize a variable to track maximum memory usage
    max_memory = 0

    start = time.time()
    max_memory = report_memory_usage("initial state", max_memory, verbose)
    if verbose:
        report_time_interval(start, "initial state - time elapsed", verbose)

    # Initialize an empty DataFrame
    DF_MC = pd.DataFrame()

    ################################################################
    for branch in E_branches:
        E_array = uproot.dask(
            {file_path: E_tree for file_path in file_paths},
            filter_name=branch,
            library="ak",
        )
        max_memory = report_memory_usage(f"E_array ({branch}) loaded", max_memory, verbose)
        if verbose:
            report_time_interval(start, f"E_array ({branch}) loaded - time elapsed", verbose)

        E_data = E_array.compute()
        max_memory = report_memory_usage(f"E_data ({branch}) computed", max_memory, verbose)
        if verbose:
            report_time_interval(start, f"E_data ({branch}) computed - time elapsed", verbose)

        # Extract the last part of the branch name
        column_name = branch.split("/")[-1]

        # Add current branch to DataFrame
        DF_MC[column_name] = E_data[column_name].to_list()
        max_memory = report_memory_usage(f"E_tree ({branch}) added to DataFrame", max_memory, verbose)
        if verbose:
            report_time_interval(start, f"E_tree ({branch}) added to DataFrame - time elapsed", verbose)

        # Clear E_data to free memory
        del E_data

    ##################################################################
    # Load and process `T_tree` incrementally
    for branch in T_branches:
        T_array = uproot.dask(
            {file_path: T_tree for file_path in file_paths},
            filter_name=branch,
            library="ak",
        )
        max_memory = report_memory_usage(f"T_array ({branch}) loaded", max_memory, verbose)
        if verbose:
            report_time_interval(start, f"T_array ({branch}) loaded - time elapsed", verbose)

        T_data = T_array.compute()
        max_memory = report_memory_usage(f"T_data ({branch}) computed", max_memory, verbose)
        if verbose:
            report_time_interval(start, f"T_data ({branch}) computed - time elapsed", verbose)

        # Extract the last part of the branch name
        column_name = branch.split("/")[-1]

        # Add current branch to DataFrame
        DF_MC[column_name] = T_data[column_name].to_list()
        max_memory = report_memory_usage(f"T_tree ({branch}) added to DataFrame", max_memory, verbose)
        if verbose:
            report_time_interval(start, f"T_tree ({branch}) added to DataFrame - time elapsed", verbose)

        # Clear T_data to free memory
        del T_data

    ######################################################
    max_memory = report_memory_usage("All trees added to DataFrame", max_memory, verbose)
    if verbose:
        report_time_interval(start, "All trees added to DataFrame - time elapsed", verbose)

    if verbose:
        # Print the first 10 rows
        print(DF_MC.head(10))

    # Report final process time (always)
    report_time_interval(start, "total process time", True)

    # Report maximum memory usage
    if verbose:
        print(f">>>>>>>>>>>>>>>>>>>> Maximum memory usage during process: {max_memory:.2f} MB")

    return DF_MC






''' 
def readroot(file_list, verbose=True, step_size="100 MB"):
    if verbose:
        print("Reading files in batches:", file_list)

    # Initialize empty dictionaries to store data
    file_content = {"headerTree": [], "E": [], "T": []}

    for file in file_list:
        if verbose:
            print(f"Processing file: {file}")

        # Read "headerTree" in batches
        try:
            if verbose:
                print("Iterating over 'headerTree'")
            for batch in uproot.iterate(f"{file}:headerTree", step_size=step_size, library="np"):
                file_content["headerTree"].append(batch)
        except Exception as e:
            print(f"Error iterating 'headerTree' in {file}: {e}")

        # Read "E" in batches
        try:
            if verbose:
                print("Iterating over 'E'")
            for batch in uproot.iterate(f"{file}:E", step_size=step_size, library="np"):
                file_content["E"].append(batch)
        except Exception as e:
            print(f"Error iterating 'E' in {file}: {e}")

        # Read "T" in batches
        try:
            if verbose:
                print("Iterating over 'T'")
            for batch in uproot.iterate(f"{file}:T", step_size=step_size, library="ak"):
                file_content["T"].append(batch)
        except Exception as e:
            print(f"Error iterating 'T' in {file}: {e}")

    if verbose:
        print("Completed reading files in batches.")

    return file_content
'''


'''
def readroot(file_list, verbose=True):
    file_content = {"headerTree": [], "E": [], "T": []}

    if verbose:
        print("Reading files:", file_list)

    for file in file_list:
        if verbose:
            print(f"Processing file: {file}")

        # Read "headerTree"
        try:
            if verbose:
                print("Attempting to read 'headerTree'")
            tree = uproot.open(f"{file}:headerTree")
            data = tree.arrays(library="np")
            file_content["headerTree"].append(data)
        except Exception as e:
            print(f"Error reading 'headerTree' in {file}: {e}")

        # Read "E"
        try:
            if verbose:
                print("Attempting to read 'E'")
            tree = uproot.open(f"{file}:E")
            data = tree.arrays(library="np")
            file_content["E"].append(data)
        except Exception as e:
            print(f"Error reading 'E' in {file}: {e}")

        # Read "T"
        try:
            if verbose:
                print("Attempting to read 'T'")
            tree = uproot.open(f"{file}:T")
            data = tree.arrays(library="ak")
            file_content["T"].append(data)
        except Exception as e:
            print(f"Error reading 'T' in {file}: {e}")

    if verbose:
        print("Completed reading files.")

    return file_content

'''



'''
def readroot(file_list, verbose=True):
    file_content = {}
    
    if verbose:
        print("Reading files:", file_list)
    
    # Attempt to read "headerTree"
    try:
        if verbose:
            print("Attempting to read 'headerTree'")
        file_content["headerTree"] = uproot.concatenate([f"{file}:headerTree" for file in file_list])
    except Exception as e:
        print(f"Error reading 'headerTree': {e}")
        file_content["headerTree"] = None

    # Attempt to read "E"
    try:
        if verbose:
            print("Attempting to read 'E'")
        file_content["E"] = uproot.concatenate([f"{file}:E" for file in file_list] , library="np")
    except Exception as e:
        print(f"Error reading 'E': {e}")
        file_content["E"] = None

    # Attempt to read "T"
    try:
        if verbose:
            print("Attempting to read 'T'")
        file_content["T"] = uproot.concatenate([f"{file}:T" for file in file_list], library="ak")
    except Exception as e:
        print(f"Error reading 'T': {e}")
        file_content["T"] = None

    # Final output in verbose mode
    if verbose:
        print("Completed reading. Contents:", file_content.keys())

    return file_content

'''



"""
def readroot(file_list, branches_headerTree=None, branches_E = None, branches_T=None, verbose=True):
    file_content = {}
    
    if verbose:
        print("Reading files:", file_list)
    
    # Define default branches if not provided
    if branches_headerTree is None:
        branches_headerTree = ['livetime_s']  # Replace with actual branch names
    if branches_E is None:
        branches_E = ['Evt/run_id']
    if branches_T is None:
        branches_T = ['coords/trackfit_dec', 'coords/showerfit_ra']
    
    # Attempt to read "headerTree"
    try:
        if verbose:
            print("Attempting to read 'headerTree'")
        file_content["headerTree"] = uproot.concatenate(
            [f"{file}:headerTree" for file in file_list],
            expressions=branches_headerTree,
            library="ak"
        )
    except Exception as e:
        print(f"Error reading 'headerTree': {e}")
        file_content["headerTree"] = None

    # Attempt to read "E"
    try:
        if verbose:
            print("Attempting to read 'E'")
        file_content["E"] = uproot.concatenate(
            [f"{file}:E" for file in file_list],
            expressions=branches_E,
            library="np"
        )
    except Exception as e:
        print(f"Error reading 'E': {e}")
        file_content["E"] = None

    # Attempt to read "T"
    try:
        if verbose:
            print("Attempting to read 'T'")
        file_content["T"] = uproot.concatenate(
            [f"{file}:T" for file in file_list],
            expressions=branches_T,
            library="ak"
        )
    except Exception as e:
        print(f"Error reading 'T': {e}")
        file_content["T"] = None

    # Final output in verbose mode
    if verbose:
        print("Completed reading. Contents:", file_content.keys())

    return file_content

"""