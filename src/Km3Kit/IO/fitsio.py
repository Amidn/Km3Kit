import pandas as pd
import yaml
from astropy.io import fits
import numpy as np
from Km3Kit.utils.yml_utils import add_dataset_to_registry
from Km3Kit.utils.yml_utils import readConfigs
import os

def validate_header(header):
    """
    Validates and converts FITS header values to their correct types.
    Args:
        header (dict): The header dictionary to validate.

    Returns:
        dict: The validated header dictionary.
    """
    validated_header = {}
    for key, value in header.items():
        if isinstance(value, str) and value.isdigit():
            # Convert strings that represent integers to integers
            validated_header[key] = int(value)
        elif value in {"T", "F"}:
            # Convert boolean-like strings to bool
            validated_header[key] = True if value == "T" else False
        else:
            # Keep other values as they are
            validated_header[key] = value
    return validated_header


def process_fits_config_multi(yaml_config, df):
    """
    Processes the YAML configuration for multiple extensions (Primary, Events, GTI) and dynamically generates headers and columns.

    Args:
        yaml_config (dict): The configuration dictionary read from the YAML file.
        df (pd.DataFrame): The input Pandas DataFrame.

    Returns:
        tuple: Primary header, Event HDU, GTI HDU.
    """
    # Process Primary Header
    primary_header = yaml_config["Primary"]

    # Process Event Header
    event_header = validate_header(yaml_config["event_header"])
    columns_config = yaml_config["columns"]

    event_columns = []
    ttype_index = 1  # Start with index 1 for TTYPE, TFORM, TUNIT

    for df_col, col_info in columns_config.items():
        fits_name, fits_type, fits_unit, conversion = col_info

        # Fetch data from DataFrame
        try:
            data = df[df_col]
        except KeyError:
            raise ValueError(f"Column '{df_col}' not found in DataFrame.")

        # Apply conversion if specified
        if conversion:
            conversion_factor = eval(str(conversion))
            data = data * conversion_factor

        # Create FITS column
        if fits_unit is not None:
            fits_col = fits.Column(name=fits_name, format=fits_type, unit=fits_unit, array=data)
        else:
            fits_col = fits.Column(name=fits_name, format=fits_type, array=data)
        event_columns.append(fits_col)

        # Add TTYPE, TFORM, TUNIT dynamically to the event header
        event_header[f"TTYPE{ttype_index}"] = fits_name
        event_header[f"TFORM{ttype_index}"] = fits_type
        if fits_unit:
            event_header[f"TUNIT{ttype_index}"] = fits_unit
        ttype_index += 1

    # Dynamically compute NAXIS2 (number of rows) and TFIELDS (number of fields)
    event_header["NAXIS2"] = len(df)  # Number of rows in the DataFrame
    event_header["TFIELDS"] = len(columns_config)  # Number of columns

    # Process GTI Header (Static)
    gti_header = validate_header(yaml_config["GTI_header"])

    # Process GTI Columns (example: START and STOP times)
    gti_columns = [
        fits.Column(name="START", format="D", unit="s", array=[0.0] * len(df)),
        fits.Column(name="STOP", format="D", unit="s", array=[0.0] * len(df)),
    ]

    # Dynamically compute NAXIS2 and TFIELDS for GTI
    gti_header["NAXIS2"] = len(df)
    gti_header["TFIELDS"] = len(gti_columns)

    return primary_header, event_columns, event_header, gti_columns, gti_header


def create_fits_file(dataset_name, yaml_path, df, output_path, verbose = False):
    """
    Creates a FITS file with Primary, Event, and GTI extensions based on the YAML configuration and DataFrame.

    Args:
        yaml_path (str): Path to the YAML configuration file.
        df (pd.DataFrame): The input Pandas DataFrame.
        output_path (str): Path to save the FITS file.
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

    # Read YAML configuration
    with open(yaml_path, 'r') as file:
        yaml_config = yaml.safe_load(file)

    # Process FITS configuration
    primary_header, event_columns, event_header, gti_columns, gti_header = process_fits_config_multi(yaml_config, df)

    # Create Primary HDU
    primary_hdu = fits.PrimaryHDU(header=fits.Header(primary_header))

    # Create Event HDU
    event_hdu = fits.BinTableHDU.from_columns(event_columns, header=fits.Header(event_header), name="EVENTS")

    # Create GTI HDU
    gti_hdu = fits.BinTableHDU.from_columns(gti_columns, header=fits.Header(gti_header), name="GTI")

    # Write FITS file
    hdul = fits.HDUList([primary_hdu, event_hdu, gti_hdu])
    hdul.writeto(output_path, overwrite=True)
    print(f"FITS file created at {output_path}")
    data_fits = os.path.join(saving_dir,str(dataset_name) + "_converted_" + "Data.fits")
    muon_fits = os.path.join(saving_dir,str(dataset_name) + "_converted_" + "muon_fits")
    neutrino_fits = os.path.join(saving_dir,str(dataset_name) + "_converted_" + "neutrino_fits")
    add_dataset_to_registry(
                    name=f"{dataset_name}_converted_2fitst",
                    data_type="FITS",
                    comment="Converted dataset into FITS format.",
                    directory_path=saving_dir,
                    data_name = data_fits ,
                    muon_name = muon_fits,
                    neutrino_name = neutrino_fits,
                    verbose=verbose
                )
    print(f"FITS file path added  to {saving_dir}")

    
