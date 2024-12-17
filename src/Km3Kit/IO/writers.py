from gammapy.data import Observation, GTI, EventList, DataStore
from gammapy.irf import EffectiveAreaTable2D, EnergyDispersion2D, PSF3D, Background2D
from astropy.time import Time
from astropy.coordinates import EarthLocation
import astropy.units as u
from gammapy.data.pointing import FixedPointingInfo
from astropy.table import Table
import os

class Km3NetDataBuilder:
    def __init__(
        self,
        aeff_file,
        edisp_file,
        psf_file,
        bkg_file,
        events_file,
        output_file,
        datastore_dir,
        reference_time="2000-01-01T00:00:00",
        geolon=16.10,
        geolat=36.27,
        altitude=-3000,
        obs_id=21,
        ra_pnt=0.0,
        dec_pnt=0.0,
    ):
        self.aeff_file = aeff_file
        self.edisp_file = edisp_file
        self.psf_file = psf_file
        self.bkg_file = bkg_file
        self.events_file = events_file
        self.output_file = output_file
        self.datastore_dir = datastore_dir
        self.reference_time = Time(reference_time, scale="utc")
        self.geolon = geolon
        self.geolat = geolat
        self.altitude = altitude
        self.obs_id = obs_id
        self.ra_pnt = ra_pnt
        self.dec_pnt = dec_pnt
        
    def load_irfs(self):
        # Load IRFs
        self.aeff = EffectiveAreaTable2D.read(self.aeff_file)
        self.psf = PSF3D.read(self.psf_file)
        self.edisp = EnergyDispersion2D.read(self.edisp_file, hdu="EDISP_2D")
        self.bkg_mu = Background2D.read(self.bkg_file)
    
    def load_events(self):
        # Load event list and extract times
        self.event_list = EventList.read(self.events_file)
        self.event_times = self.event_list.table["TIME"]
        
        self.tstart_mjd = self.event_times.min()
        self.tstop_mjd = self.event_times.max()
        self.tstart_seconds = (self.tstart_mjd - self.reference_time.mjd) * 86400
        self.tstop_seconds = (self.tstop_mjd - self.reference_time.mjd) * 86400
        
        # Update event metadata
        self.event_list.table.meta["TIMEUNIT"] = "s"
        self.event_list.table.meta["TSTART"] = self.tstart_seconds
        self.event_list.table.meta["TSTOP"] = self.tstop_seconds
        self.event_list.table.meta["GEOLON"] = self.geolon
        self.event_list.table.meta["GEOLAT"] = self.geolat
        self.event_list.table.meta["ALTITUDE"] = self.altitude
        self.event_list.table.meta["DEADC"] = 1.0
        
    def create_gti(self):
        self.gti = GTI.create(
            start=self.tstart_seconds * u.s,
            stop=self.tstop_seconds * u.s,
            reference_time=self.reference_time,
        )
    
    def create_pointing(self):
        # Fixed pointing info
        self.pointing = FixedPointingInfo.from_fits_header(
            {"RA_PNT": self.ra_pnt, "DEC_PNT": self.dec_pnt, "RADECSYS": "FK5"}
        )
        self.location = EarthLocation(lat=self.geolat * u.deg, lon=self.geolon * u.deg, height=self.altitude * u.m)
        
    def create_observation(self):
        # Load IRFs and events
        self.load_irfs()
        self.load_events()
        
        # Create GTI and Pointing
        self.create_gti()
        self.create_pointing()
        
        # Create Observation
        self.observation = Observation(
            obs_id=self.obs_id,
            pointing=self.pointing,
            gti=self.gti,
            aeff=self.aeff,
            edisp=self.edisp,
            psf=self.psf,
            bkg=self.bkg_mu,
            events=self.event_list,
        )
        
        # Save the observation
        self.observation.write(self.output_file, overwrite=True)
        print(f"Observation with events saved to {self.output_file}")
        
    def create_datastore(self):
        # Ensure datastore directory
        os.makedirs(self.datastore_dir, exist_ok=True)
        
        # Create HDU-index table
        hdu_table = Table(
            names=["OBS_ID", "HDU_TYPE", "HDU_CLASS", "FILE_DIR", "FILE_NAME", "HDU_NAME"],
            dtype=["i4", "S20", "S20", "S20", "S20", "S20"],
        )
        hdu_table.add_row([self.obs_id, "events", "events", "./", os.path.basename(self.output_file), "EVENTS"])
        hdu_table.add_row([self.obs_id, "gti", "gti", "./", os.path.basename(self.output_file), "GTI"])
        hdu_table.add_row([self.obs_id, "aeff", "aeff_2d", "./", os.path.basename(self.output_file), "EFFECTIVE AREA"])
        hdu_table.add_row([self.obs_id, "edisp", "edisp_2d", "./", os.path.basename(self.output_file), "ENERGY DISPERSION"])
        # For a PSF3D IRF, standard HDU name should be "PSF"
        hdu_table.add_row([self.obs_id, "psf", "psf_3d", "./", os.path.basename(self.output_file), "PSF"])
        hdu_table.add_row([self.obs_id, "bkg", "bkg_2d", "./", os.path.basename(self.output_file), "BACKGROUND"])
        
        hdu_table.write(os.path.join(self.datastore_dir, "hdu-index.fits.gz"), overwrite=True)
        
        # Create OBS-index table
        # Use approximate times and pointings; in a real scenario these should come from the data
        obs_table = Table(
            names=["OBS_ID", "RA_PNT", "DEC_PNT", "ZEN_PNT", "TSTART", "TSTOP", "ONTIME", "LIVETIME", "DEADC"],
            dtype=["i4", "f8", "f8", "f8", "f8", "f8", "f8", "f8", "f8"],
        )
        
        # Convert TSTART/TSTOP in MJD for obs-index
        # If tstart_seconds are relative to reference_time, convert back to absolute MJD
        tstart_abs_mjd = self.reference_time.mjd + (self.tstart_seconds / 86400.0)
        tstop_abs_mjd = self.reference_time.mjd + (self.tstop_seconds / 86400.0)
        
        # Dummy values for ZEN_PNT, ONTIME, LIVETIME can be replaced with actual computed values
        # ONTIME and LIVETIME in seconds, TSTART and TSTOP in MJD
        obs_table.add_row([
            self.obs_id, 
            self.ra_pnt, 
            self.dec_pnt, 
            0.0, 
            tstart_abs_mjd, 
            tstop_abs_mjd, 
            self.tstop_seconds - self.tstart_seconds, 
            self.tstop_seconds - self.tstart_seconds, 
            1.0
        ])
        
        obs_table.write(os.path.join(self.datastore_dir, "obs-index.fits.gz"), overwrite=True)
        
        # Verify DataStore
        datastore = DataStore.from_dir(self.datastore_dir)
        print("DataStore created successfully:")
        print(datastore)

        # Optionally load Observation from DataStore
        observations = datastore.get_observations([self.obs_id])
        print(observations)


# Example usage:
# builder = Km3NetDataBuilder(
#     aeff_file="aeff.fits",
#     edisp_file="edisp.fits",
#     psf_file="psf.fits",
#     bkg_file="bkg_mu.fits",
#     events_file="arca21.fits",
#     output_file="km3net_observation_with_events.fits",
#     datastore_dir="km3net_datastore"
# )
# builder.create_observation()
# builder.create_datastore()