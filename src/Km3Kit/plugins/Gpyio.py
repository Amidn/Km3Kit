import astropy.units as u
from astropy.coordinates import SkyCoord

from gammapy.data import EventList
from gammapy.catalog import SourceCatalog3FHL



def read_eve_cat(event_path, catalog_path):
    """
    Reads event and catalog data from the specified paths.

    Parameters:
    event_path (str): Path to the event FITS file.
    catalog_path (str): Path to the 3FHL catalog FITS file.

    Returns:
    tuple: A tuple containing the event list and the catalog objects.
    """
    events = EventList.read(event_path)
    catalog = SourceCatalog3FHL(catalog_path)
    return events, catalog

# Example usage:
