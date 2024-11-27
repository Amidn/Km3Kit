
import astropy.units as u
from astropy.coordinates import SkyCoord
from gammapy.data import EventList
from .rootio import load_dst, pd_dataFrame
from ..utils.yml_utils import Loader, load_branches_config , readConfigs, add_dataset_to_registry
from utils.tools import  diagnose_dataframe
import pandas as pd
import numpy as np
from astropy.io import fits
import io
import os


def process_dfs(dataset_name, branches_config_path="config/branches.yml", save_pd=None, verbose=True):
    """
    Loads data, muon, and neutrino datasets into Pandas DataFrames and optionally saves them as HDF5 files.

    Args:
        dataset_name (str): Name of the dataset in the registry.
        branches_config_path (str): Path to the branches configuration YAML file.
        save_pd (bool): If True, saves the DataFrames as HDF5 files.
        verbose (bool): If True, enables detailed logging.

    Returns:
        dict: A dictionary containing the DataFrames for 'data', 'muon', and 'neutrino'.
    """
    # Load the DataFrames
    if verbose:
            print("Reading configuration for saving directories...")
    configs = readConfigs(verbose=verbose)
        
        # Validate configuration keys
    try:
        saving_dir = configs["FileConfig"]["Saving_Dir"]
    except KeyError as e:
        raise KeyError(f"Missing configuration key: {e}. Check 'FileConfig' in your config YAML file.")

    # Ensure saving directory exists
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)
        if verbose:
            print(f"Created directory: {saving_dir}")


    if verbose:
        print("Loading data dataset...")
    df_data = pd_dataFrame(dataset_name, branches_config_path, data_type="data", verbose=verbose)
        
    if save_pd:
                
        data_h5 = os.path.join(saving_dir,str(dataset_name) + "_converted_" + "Data.h5")
        if verbose:
            print(f"Saving data to {data_h5} ...")
            print("\nDiagnosing 'data' DataFrame:")
            diagnose_dataframe(df_data)  # Diagnose the data DataFrame
        df_data.to_hdf(data_h5, key="data", mode="w")


    if verbose:
        print("Loading  muon,  dataset...")
    df_muon = pd_dataFrame(dataset_name, branches_config_path, data_type="muon", verbose=verbose)

    if save_pd:
        muon_h5 = os.path.join(saving_dir, str(dataset_name) + "_converted_" + "Muon.h5")
        if verbose:
            print(f"Saving data to {muon_h5} ...")
            print("\nDiagnosing 'muon' DataFrame:")
            diagnose_dataframe(df_muon)  # Diagnose the muon DataFrame
        df_muon.to_hdf(muon_h5, key="data", mode="w")


    if verbose:
        print("Loading neutrino dataset...")
    df_neutrino = pd_dataFrame(dataset_name, branches_config_path, data_type="neutrino", verbose=verbose)
    if save_pd:
        neutrino_h5 = os.path.join(saving_dir, str(dataset_name) + "_converted_" + "Neutrino.h5")
        if verbose:
            print(f"Saving data to  {neutrino_h5} ...")
            print("\nDiagnosing 'neutrino' DataFrame:")
            diagnose_dataframe(df_neutrino)  # Diagnose the neutrino DataFrame
        df_neutrino.to_hdf(neutrino_h5, key="data", mode="w")

    if save_pd:
        if verbose:
            print("Files successfully saved.")

        # Add the new dataset information to the registry
        if verbose:
            print("Updating the dataset registry...")

        add_dataset_to_registry(
            name=f"{dataset_name}_converted_2pd",
            data_type="HDF5",
            comment="Converted dataset into HDF5 format.",
            directory_path=saving_dir,
            data_name="Data.h5",
            muon_name="Muon.h5",
            neutrino_name="Neutrino.h5",
            verbose=verbose
        )

    # Return DataFrames for further processing
    return {"data": df_data, "muon": df_muon, "neutrino": df_neutrino}


def read(dataset_name, usage_tag, recreate=False, verbose=False, yml_data_registry_path="config/dataset_registry.yml"):
    """
    Chooses a dataset based on the given name and usage tag.
    Searches the YAML registry for preprocessed versions (Pandas or FITS).

    Args:
        dataset_name (str): The name of the dataset (e.g., "arca21").
        usage_tag (str): The intended usage ("GammaPy" or "KM3Net").
        recreate (bool): If True, forces the use of the original dataset without checking conversions.
        verbose (bool): If True, prints detailed logs.
        yml_data_registry_path (str): Path to the YAML registry file.

    Returns:
        str: Name of the chosen dataset.
    """
    # If recreate is True, return the original dataset name
    if recreate:
        if verbose:
            print(f"Recreate is True. Returning original dataset: {dataset_name}")
        return dataset_name

    # Load the dataset names and types from the YAML registry
    datasets = Loader.list_datasets_and_types(yml_registry_path=yml_data_registry_path, verbose=verbose)

    # Build the names of the preprocessed datasets
    dataset_name_fits = dataset_name + "_converted_2Fits"
    dataset_name_pd = dataset_name + "_converted_2pd"

    # Check the usage tag and look for the appropriate preprocessed dataset
    if usage_tag == "GammaPy":
        for name, _ in datasets:
            if name == dataset_name_fits:
                if verbose:
                    print(f"Found preprocessed FITS dataset: {dataset_name_fits}")
                return dataset_name_fits
        if verbose:
            print(f"Preprocessed FITS dataset not found. Returning original dataset: {dataset_name}")
        return dataset_name

    elif usage_tag == "KM3Net":
        for name, _ in datasets:
            if name == dataset_name_pd:
                if verbose:
                    print(f"Found preprocessed Pandas dataset: {dataset_name_pd}")
                return dataset_name_pd
        if verbose:
            print(f"Preprocessed Pandas dataset not found. Returning original dataset: {dataset_name}")
        return dataset_name

    else:
        raise ValueError(f"Invalid usage_tag: {usage_tag}. Must be 'GammaPy' or 'KM3Net'.")
    
        


def dataframe_to_fits(df, output_path=None, save_fits=False):
    """
    Convert a Pandas DataFrame into a FITS file with appropriate headers and structure.

    Parameters:
        df (pd.DataFrame): Input DataFrame with columns like 'log10_Erec', 'ra_deg', 'dec_deg', 'mjd'.
        output_path (str, optional): Path to save the generated FITS file if save_fits=True.
        save_fits (bool): If True, save the FITS file to disk. Default is False.

    Returns:
        HDUList or None:
        - If save_fits=False, returns the FITS HDUList object for in-memory use.
        - If save_fits=True, saves the FITS file to the specified output_path and returns None.
    """
    # Step 1: Convert DataFrame columns to required formats
    try:
        energy = df['trks.E']  
        run_id = df["run_id"]
        event_id = df["id"]
        Type = df["type"] 
        ra_values = df['ra_deg']
        dec_values = df['dec_deg']
        time_values = df['mjd']  # MJD time in days

        bdt_columns = df['bdt_trk'].apply(pd.Series)
        bdt_columns.columns = ['BDT_trk_0', 'BDT_trk_1']
        BDT_trk_0 = df['BDT_trk_0']
        BDT_trk_1 = df['BDT_trk_1']

    except KeyError as e:
        raise ValueError(f"Missing required column in DataFrame: {e}")

    # Step 2: Create FITS columns
    columns = [
        fits.Column(name='ENERGY', format='E', unit='GeV', array=energy),
        fits.Column(name='RUN_ID', format='D', unit='d', array=run_id),
        fits.Column(name='EVENT_ID', format='D', unit='d', array=event_id),
        fits.Column(name='TYPE', format='D', unit='d', array=Type),

        fits.Column(name='RA', format='E', unit='deg', array=ra_values),
        fits.Column(name='DEC', format='E', unit='deg', array=dec_values), 	

        fits.Column(name='TIME', format='D', unit='d', array=time_values),
        fits.Column(name='BDT_trk0', format='D', unit='d', array= BDT_trk_0 ),
        fits.Column(name='BDT_trk1', format='D', unit='d', array= BDT_trk_1 )
    ]

    # Step 3: Create HDUs
    primary_hdu = fits.PrimaryHDU()  # Primary header
    events_hdu = fits.BinTableHDU.from_columns(columns, name='EVENTS')  # Event data HDU

    # Step 4: Add metadata to headers
    header = events_hdu.header
    header['XTENSION'] = 'BINTABLE'
    header['BITPIX'] = 8
    header['NAXIS'] = 2
    header['NAXIS1'] = 154  # Length of one row in bytes (example value; update as needed)
    header['NAXIS2'] = len(ra_values)  # Number of rows
    header['TFIELDS'] = len(columns)  # Number of fields
    header['DATE'] = 'NONE'
    header['DATE-OBS'] = 'NONE'
    header['DATE-END'] = 'NONE'
    header['TELESCOP'] = 'None'
    header['INSTRUME'] = 'None'
    header['OBSERVER'] = 'Arca21'
    header['EQUINOX'] = 2000.0
    header['RADECSYS'] = 'FK5'
    header['MJDREFI'] = 0.0
    header['MJDREFF'] = 0.00074287037037037  # Fraction of day
    header['TIMEUNIT'] = 's'
    header['TIMESYS'] = 'TT'
    header['DSTYP2'] = 'TIME'
    header['DSUNI2'] = 'd'
    header['DSTYP4'] = 'ENERGY'
    header['DSUNI4'] = 'MeV'
    header['DSVAL4'] = '10000:2000000'

    # Step 5: Create HDUList
    hdul = fits.HDUList([primary_hdu, events_hdu])

    if save_fits:
        # Save to file if requested
        if output_path is None:
            raise ValueError("Output path must be specified if save_fits=True.")
        hdul.writeto(output_path, overwrite=True)
        print(f"FITS file saved to: {output_path}")
        return None
    else:
        # Keep in memory
        print("FITS file created in memory.")
        return hdul