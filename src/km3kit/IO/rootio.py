import uproot
from yml_Loader import Loader

def readroot(file_list):
    file_content = {}
    file_content["headerTree"] = uproot.concatenate([f"{file}:headerTree" for file in file_list])
    file_content["E"]          =  uproot.concatenate([f"{file}:E" for file in file_list])
    file_content["T"]          =  uproot.concatenate([f"{file}:T" for file in file_list])
    return file_content