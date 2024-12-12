import numpy as np
from ctypes import cdll
import ctypes

lib = None

def load_lib():
    """Load the shared library for atmospheric flux calculations."""
    global lib
    if lib is None:
        lib = cdll.LoadLibrary('./libfluxfunctions.so')
        lib.get_conventional_flux.restype = ctypes.c_double
        lib.get_prompt_flux.restype = ctypes.c_double

def atmospheric_flux(**kwargs):
    """Calculate the atmospheric neutrino flux using a provided method.

    Args:
        mc_type (int): PDG encoding for particle type.
        energy (float): Energy of particle in GeV.
        zenith (float): Zenith angle in rad.
        method (callable): The method from the shared library for flux calculation.

    Returns:
        float or array: The calculated flux.
    """
    try:
        return kwargs["method"](ctypes.c_int(int(kwargs["mc_type"])),
                                ctypes.c_double(kwargs["energy"]),
                                ctypes.c_double(kwargs["zenith"]))
    except TypeError:
        f = []
        for i in kwargs["energy"].index:
            f.append(kwargs["method"](ctypes.c_int(int(kwargs["mc_type"][i])),
                                      ctypes.c_double(kwargs["energy"][i]),
                                      ctypes.c_double(kwargs["zenith"][i])))
        return np.array(f, float)

def atmospheric_conventional(**kwargs):
    """Calculate the conventional atmospheric neutrino flux."""
    load_lib()
    return atmospheric_flux(**kwargs, method=lib.get_conventional_flux)

def atmospheric_prompt(**kwargs):
    """Calculate the prompt atmospheric neutrino flux."""
    load_lib()
    return atmospheric_flux(**kwargs, method=lib.get_prompt_flux)