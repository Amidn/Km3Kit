
from yml_Loader import Loader
from rootio import readroot


def Read(input):
    data_files, neutrino_files, muon_files = Loader.read(input)
    type_ = Loader.get_data_type(input)
    if type_ == "ROOT":
        DATA_     = readroot(data_files)
        NEUTRINO_ = readroot(neutrino_files)
        MUON_     = readroot(muon_files)
        print ("ORDER: DATA, NEUOTRINO, MUON")
        return DATA_ , NEUTRINO_, MUON_


    