import uproot


def readroot(file_list):
    file_content = {}
    try:
        file_content["headerTree"] = uproot.concatenate([f"{file}:headerTree" for file in file_list])
    except Exception as e:
        print(f"Error reading 'headerTree': {e}")
        file_content["headerTree"] = None

    try:
        file_content["E"] = uproot.concatenate([f"{file}:E" for file in file_list])
    except Exception as e:
        print(f"Error reading 'E': {e}")
        file_content["E"] = None

    try:
        file_content["T"] = uproot.concatenate([f"{file}:T" for file in file_list])
    except Exception as e:
        print(f"Error reading 'T': {e}")
        file_content["T"] = None

    return file_content