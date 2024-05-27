from astropy.table import Table
import astropy.units as u
import os
import uproot
import awkward as ak
from gammapy.data import EventList
from astropy.time import Time
import numpy as np


class DataReader:
    
    def __init__(self, file_path):
        self.file_path = file_path

    def readDst(self, verbose=True):
        if verbose:
            print(f"Reading {self.file_path}")

        if os.path.exists(self.file_path):
            return uproot.open(self.file_path)
        else:
            print(f"File {self.file_path} not found.")
            return None

    def branch_check(self, branch):
        # Define the dictionary with paths to DST branches
        paths = {
            "Evt_path": "E/Evt",
            "hits_path": "E/Evt/hits",
            "mchits_path": "E/Evt/mc_hits",
            "trks_path": "E/Evt/trks",
            "mctrks_path": "E/Evt/mc_trks",
            "Coords_path": "T/coords",
            "sum_mc_trks_path": "T/sum_mc_trks",
            "sum_mc_nu_path": "T/sum_mc_nu",
            "sum_mc_ev_path": "T/sum_mc_evt",
            "sum_hits_path": "T/sum_hits",
            "sum_mc_hits_path": "T/sum_mc_hits"
        }
        if branch not in paths:
            print(f"Branch {branch} is not in paths dictionary.")
            return None
        return paths[branch]

    def trkList(self, DataMC, verbose=True):
        DST_ = self.readDst(verbose=verbose)
        if DST_ is None:
            raise FileNotFoundError(f"File {self.file_path} could not be read.")

        if verbose:
            print("Create an Astropy Table with required columns")

        data_ = Table()

        if DataMC == "trk":
            coords_ = DST_[self.branch_check("Coords_path")]
            if coords_ is None:
                raise ValueError("Coords_path data not found.")
            mjd = coords_["mjd"].array()
            ra = coords_["trackfit_ra"].array() * (180.0 / np.pi)
            dec = coords_["trackfit_dec"].array() * (180.0 / np.pi)

            Id_ = DST_[self.branch_check("Evt_path")]
            if Id_ is None:
                raise ValueError("Evt_path data not found.")
            run_id = Id_["run_id"].array()

            trk_ = DST_[self.branch_check("trks_path")]
            if trk_ is None:
                raise ValueError("trks_path data not found.")
            Energy_ = trk_["trks.E"].array() * 1e3
            like_ = trk_["trks.lik"].array()
            pdg_ = trk_["trks.type"].array()
            length_ = trk_["trks.len"].array()

        elif DataMC == "mc_trk":
            coords_ = DST_[self.branch_check("Coords_path")]
            if coords_ is None:
                raise ValueError("Coords_path data not found.")
            mjd = coords_["mjd"].array()
            ra = coords_["trackfit_ra"].array() * (180.0 / np.pi)
            dec = coords_["trackfit_dec"].array() * (180.0 / np.pi)

            Id_ = DST_[self.branch_check("Evt_path")]
            if Id_ is None:
                raise ValueError("Evt_path data not found.")
            run_id = Id_["run_id"].array()

            trk_ = DST_[self.branch_check("mctrks_path")]
            if trk_ is None:
                raise ValueError("mctrks_path data not found.")
            Energy_ = trk_["mc_trks.E"].array() * 1.e3
            like_ = trk_["mc_trks.lik"].array()
            pdg_ = trk_["mc_trks.type"].array()
            length_ = trk_["mc_trks.len"].array()
        else:
            print("Unknown DataMC type requested")
            return None

        # Convert awkward arrays to numpy arrays before adding to Astropy Table
        data_['RA'] = ak.to_numpy(ra) * u.deg
        data_['DEC'] = ak.to_numpy(dec) * u.deg
        data_['ENERGY'] = ak.to_numpy(Energy_) * u.MeV
        data_['TIME'] = ak.to_numpy(mjd) * u.day
        data_["EVENT_ID"] = ak.to_numpy(run_id) * u.dimensionless_unscaled
        data_["lik"] = ak.to_numpy(like_) * u.dimensionless_unscaled
        data_["PDG"] = ak.to_numpy(pdg_) * u.dimensionless_unscaled
        data_["length"] = ak.to_numpy(length_) * u.m

        if verbose:
            print("Setting meta information for the table")

        meta = {
            'MJDREF': 0,
            'XTENSION': 'BINTABLE',
            'BITPIX': 8,
            'NAXIS': 2,
            'NAXIS1': len(data_.columns),
            'NAXIS2': len(data_),
            'TFIELDS': len(data_.columns),
            'DATE': 'NONE',
            'DATE-OBS': 'NONE',
            'DATE-END': 'NONE',
            'DSTYP2': 'TIME',
            'DSUNI2': 'd',  # Modified to days
            'DSVAL2': 'TABLE',
            'MJDREFI': 0.0,
            'CHECKSUM': 'UH9VUE8UUE8UUE8U',
            'DATASUM': 'None',
            'TELESCOP': 'None',
            'INSTRUME': 'None',
            'EQUINOX': 2000.0,
            'RADECSYS': 'FK5',
            'OBSERVER': 'Arca21',
            'EXTNAME': 'EVENTS',
            'HDUCLAS1': 'EVENTS',
            'MJDREFF': 0.00074287037037037,
            'TIMEUNIT': 's',
            'TIMEZERO': 0.0,
            'TIMESYS': 'TT',
            'DSTYP3': 'BIT_MASK(EVENT_TYPE,3,P8R2)',
            'DSUNI3': 'DIMENSIONLESS',
            'DSVAL3': '1:1',
            'DSTYP4': 'ENERGY',
            'DSUNI4': 'MeV',
            'DSVAL4': '10000:2000000'
        }

        if verbose:
            print("Table is created")

        return data_, meta



class CustomEventList(EventList):
    @classmethod
    def from_table_meta(cls, table_, meta_):
        """
        Create EventList from an Astropy Table and meta information.
        
        Parameters
        ----------
        table : `astropy.table.Table`
            Table containing event data.
        meta : dict
            Meta information dictionary.
        
        Returns
        -------
        event_list : `CustomEventList`
            Event list object.
        """
      #  event_list = cls(table_)
      #  event_list.meta = meta_
       # table = table_.meta.update(meta_)
       # return cls(table=table)
        event_list = cls(table=table_)
        event_list.meta = meta_
        return event_list




# This class needs to be tested
class hff2fits:
    def __init__(self, filepath, **kwargs):
        self.filepath = filepath

    def read_key(self, verbose=False):
        if verbose:
            print(f"Reading {self.filepath}")
        with pd.HDFStore(self.filepath, mode='r') as store:
            keys = store.keys()
            if not keys:
                print("No keys found in the HDF5 file.")
                return None, []
            if verbose:
                print("Keys found in the HDF5 file:")
                for key in keys:
                    print(key)
        return keys
    
    def read_columns(self):
        keys = self.read_key()
        if not keys:
            return None  # No keys found, return None
        key = str(keys[0])  # Assuming you want to read the first key
        # Read only the headers (column names) without loading the data
        columns = pd.read_hdf(self.filepath, key=key, stop=0).columns.tolist()
        return columns

    def get_entries(self, num_entries=10, column=None):
        keys = self.read_key()
        if not keys:
            print("No keys found.")
            return None
        
        key = str(keys[0])  # Assuming you want to read the first key
        
        if column is None:
            # If no specific column is specified, read all columns
            custom_entries = pd.read_hdf(self.filepath, key=key, start=0, stop=num_entries)
        else:
            # Read all columns and then select the specified column
            if column not in pd.read_hdf(self.filepath, key=key, stop=0).columns:
                print(f"Column '{column}' not found in the dataset.")
                return None
            custom_entries = pd.read_hdf(self.filepath, key=key, start=0, stop=num_entries)[column]
        
        return custom_entries


    def create_column_dicts(self):
        columns = self.read_columns()
        if not columns:
            return None  # No columns found, return None
        column_dicts = [{col: idx} for idx, col in enumerate(columns)]
        return column_dicts

    def print_column_dicts(self):
        column_dicts = self.create_column_dicts()
        if not column_dicts:
            print("No column dictionaries found.")
        else:
            for idx, column_dict in enumerate(column_dicts):
                print(f"Dictionary {idx + 1}:")
                for column_name in column_dict:
                    print(column_name)
                print()
  
    def read_data(self, key=None, verbose=False):
        if key is None:
            keys = self.read_key()
            if not keys:
                return None  # No keys found, return None
            key = str(keys[0])  # Assuming you want to read the first key
        else:
            keys = self.read_key()
            if key not in keys:
                print(f"Key '{key}' not found in the HDF5 file.")
                return None
        if verbose:
            print(f"Reading {self.filepath} with key '{key}'")
        dataframe = pd.read_hdf(self.filepath, key=key)
        if verbose: 
            print ("Read successful!")
        return dataframe 
    
    def create_fits(self,  output_path='your_file.fits'):
        df_ = self.read_data()
        E_ = 1000.0 * pow(10,df_['log10_Erec']) # To MeV
        ra_values = df_['ra_deg']
        dec_values = df_['dec_deg']
        Time_ = df_['mjd']

        columns = [
            fits.Column(name='ENERGY', format='E', unit='MeV', array= E_),
            fits.Column(name='RA', format='E', unit='deg', array=ra_values),
            fits.Column(name='DEC', format='E', unit='deg', array=dec_values),
            fits.Column(name='TIME', format='D', unit='d', array=Time_)
            #fits.Column(name='EVENT_ID', format='J', array=np.array([1823040, 550833, 1353175, 9636241, 11233188, 14156811, 14140569, 15688393], dtype=np.int32)),
            #fits.Column(name='RUN_ID', format='J', array=np.array([239571670, 239577663, 239577663, 239601276, 239606871, 239618329, 239629788, 239629788], dtype=np.int32)),
            ]

        # Create HDUs
        primary_hdu = fits.PrimaryHDU()
        tb_hdu = fits.BinTableHDU.from_columns(columns, name='EVENTS')
        
        # Create HDU list and write to file
        hdul = fits.HDUList([primary_hdu, tb_hdu])
        header = hdul[1].header
        header['XTENSION'] = 'BINTABLE'
        header['BITPIX'] = 8
        header['NAXIS'] = 2
        header['NAXIS1'] = 154
        header['NAXIS2'] = len(ra_values)
        header['TFIELDS'] = len(columns)
      #  header['PCOUNT'] = 0
        #header['GCOUNT'] = 1
       # header['DIFRSP0'] = 'NONE'
       # header['DIFRSP1'] = 'NONE'
       # header['DIFRSP2'] = 'NONE'
       # header['DIFRSP3'] = 'NONE'
       # header['DIFRSP4'] = 'NONE'
        # later we can modify the DATE info based on Arca observations
        header['DATE'] = 'NONE'
        header['DATE-OBS'] ='NONE'
        header['DATE-END'] ='NONE'

      #  header['EXTVER'] = 1
      #  header['DSTYP1'] = 'BIT_MASK(EVENT_CLASS,128,P8R2)'
      #  header['DSUNI1'] = 'DIMENSIONLESS'
      #  header['DSVAL1'] = '1:1'    

        header['DSTYP2'] = 'TIME'
        header['DSUNI2'] = 'd' # Modified to days
        header['DSVAL2'] = 'TABLE'
        header['MJDREFI'] = 0.0
        #header['DSREF2'] = ':GTI'
        header['CHECKSUM'] = 'UH9VUE8UUE8UUE8U'
        header['DATASUM'] = 'None'
        header['TELESCOP'] = 'None'
        header['INSTRUME'] = 'None'
        header['EQUINOX'] = 2000.0
        header['RADECSYS'] = 'FK5'
        header['OBSERVER'] = 'Arca21'
        header['EXTNAME'] = 'EVENTS'
        header['HDUCLAS1'] = 'EVENTS'
        header['MJDREFF'] = 0.00074287037037037
        header['TIMEUNIT'] = 's'
        header['TIMEZERO'] = 0.0
        header['TIMESYS'] = 'TT'
        header['DSTYP3'] = 'BIT_MASK(EVENT_TYPE,3,P8R2)'
        header['DSUNI3'] = 'DIMENSIONLESS'
        header['DSVAL3'] = '1:1'
        header['DSTYP4'] = 'ENERGY'
        header['DSUNI4'] = 'MeV'
        header['DSVAL4'] = '10000:2000000'
        hdul.writeto(output_path, overwrite=True)
        print(f"FITS file created: {output_path}")
        return True


