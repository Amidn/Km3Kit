import uproot
import awkward as ak

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
        file_content["T"] = uproot.concatenate([f"{file}:T" for file in file_list], library="np")
    except Exception as e:
        print(f"Error reading 'T': {e}")
        file_content["T"] = None

    # Final output in verbose mode
    if verbose:
        print("Completed reading. Contents:", file_content.keys())

    return file_content