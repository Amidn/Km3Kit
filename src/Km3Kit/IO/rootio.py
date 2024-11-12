import uproot
import awkward as ak
import numpy as np



def readroot(file_list, verbose=True):
    file_content = {}

    if verbose:
        print("Setting up lazy loading for files:", file_list)

    try:
        if verbose:
            print("Setting up lazy arrays for 'headerTree'")
        file_content["headerTree"] = uproot.lazy([f"{file}:headerTree" for file in file_list])
    except Exception as e:
        print(f"Error setting up 'headerTree': {e}")
        file_content["headerTree"] = None

    try:
        if verbose:
            print("Setting up lazy arrays for 'E'")
        file_content["E"] = uproot.lazy([f"{file}:E" for file in file_list])
    except Exception as e:
        print(f"Error setting up 'E': {e}")
        file_content["E"] = None

    try:
        if verbose:
            print("Setting up lazy arrays for 'T'")
        file_content["T"] = uproot.lazy([f"{file}:T" for file in file_list])
    except Exception as e:
        print(f"Error setting up 'T': {e}")
        file_content["T"] = None

    if verbose:
        print("Lazy arrays are set up. Data will be loaded when computations are performed.")

    return file_content


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